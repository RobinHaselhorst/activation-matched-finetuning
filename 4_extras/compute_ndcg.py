"""
nDCG of a single backdoor model's predicted group ranking against the
hand-labelled relevance grid in group_relevance_scores.json. Reports
nDCG@k and nDCG_full alongside a 20k-shuffle random baseline for an
empirical p-value.

Requires `group_relevance_scores.json` — generate it first by filling in
`SCORES` in `score_groups.py` and running that script.

Run:
    python 4_extras/compute_ndcg.py --model qwen-hp-backdoor
    python 4_extras/compute_ndcg.py --model qwen-hp-backdoor --relevance-key other  # cross-check
"""

import argparse
import json
import math
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "per_model")
RELEVANCE_PATH = os.path.join(HERE, "group_relevance_scores.json")

# results-dir name  →  key used in group_relevance_scores.json
MODEL_KEY = {
    "qwen-hp-backdoor": "hp",
}


def dcg(rels: list[float], gain: str = "linear") -> float:
    """DCG with chosen gain. Denominator is log2(rank+1), rank starting at 1."""
    if gain == "linear":
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    if gain == "exp":
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    raise ValueError(f"unknown gain: {gain!r}")


def ndcg_at_k(predicted_rels: list[float], ideal_rels: list[float], k: int, gain: str = "linear") -> float:
    idcg = dcg(ideal_rels[:k], gain)
    if idcg == 0:
        return float("nan")
    return dcg(predicted_rels[:k], gain) / idcg


def random_baseline(
    rels: list[float],
    ks: list[int],
    gain: str,
    n_shuffles: int,
    seed: int,
) -> dict[int, dict]:
    """Random-shuffle baseline. Returns {k: {mean, p95, p99}} keyed by k.
    Key 0 (sentinel) is used for the full-ranking nDCG."""
    ideal = sorted(rels, reverse=True)
    N = len(rels)
    cutoffs = list(ks) + [N]  # full ranking sentinel via N

    rng = random.Random(seed)
    arr = list(rels)
    samples: dict[int, list[float]] = {k: [] for k in cutoffs}
    for _ in range(n_shuffles):
        rng.shuffle(arr)
        for k in cutoffs:
            samples[k].append(ndcg_at_k(arr, ideal, k, gain))

    out: dict[int, dict] = {}
    for k, vals in samples.items():
        s = sorted(vals)
        out[k] = {
            "mean": sum(vals) / len(vals),
            "p95": s[int(0.95 * len(s))],
            "p99": s[int(0.99 * len(s))],
            "_samples": vals,  # kept for p-value computation against actual
        }
    return out


def p_value(samples: list[float], actual: float) -> float:
    """Fraction of shuffles with nDCG >= actual (lower = more significant)."""
    return sum(1 for s in samples if s >= actual) / len(samples)


def compute_for_model(
    model_dir_name: str,
    metric: str,
    ks: list[int],
    relevance_table: dict,
    gain: str = "linear",
    relevance_key: str | None = None,
    n_shuffles: int = 20000,
    seed: int = 0,
) -> dict:
    results_path = os.path.join(RESULTS_DIR, model_dir_name, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(results_path)
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    if relevance_key is not None:
        rel_key = relevance_key
    elif model_dir_name in MODEL_KEY:
        rel_key = MODEL_KEY[model_dir_name]
    else:
        raise KeyError(
            f"No relevance-key mapping for {model_dir_name!r}. "
            f"Add it to MODEL_KEY in compute_ndcg.py, or pass --relevance-key."
        )

    group_means = results["group_means"]
    scores_by_group = relevance_table["scores_by_group"]

    # Build (group, predicted_score, relevance) for every group present in both.
    rows = []
    missing_relevance = []
    for group, stats in group_means.items():
        if metric not in stats:
            continue
        rel_entry = scores_by_group.get(group)
        if rel_entry is None:
            missing_relevance.append(group)
            continue
        rel = rel_entry.get(rel_key, 0)
        rows.append((group, float(stats[metric]), float(rel)))

    # Predicted ranking: descending by metric (higher z = more anomalous).
    predicted = sorted(rows, key=lambda r: r[1], reverse=True)
    predicted_rels = [r[2] for r in predicted]

    # Ideal ranking: descending by ground-truth relevance.
    ideal_rels = sorted((r[2] for r in rows), reverse=True)

    out = {
        "model_dir": model_dir_name,
        "relevance_key": rel_key,
        "metric": metric,
        "n_groups_scored": len(rows),
        "n_missing_relevance": len(missing_relevance),
        "n_relevant_groups": sum(1 for r in ideal_rels if r > 0),
        "trigger_group": results.get("trigger_group"),
        "gain": gain,
        "n_shuffles": n_shuffles,
        "ndcg": {f"@{k}": ndcg_at_k(predicted_rels, ideal_rels, k, gain) for k in ks},
        "ndcg_full": ndcg_at_k(predicted_rels, ideal_rels, len(rows), gain),
        "top10_predicted": [
            {"group": g, metric: s, "relevance": rel}
            for g, s, rel in predicted[:10]
        ],
        "relevant_groups_ranks": [
            {
                "group": g,
                "relevance": rel,
                "rank": rank + 1,
                metric: s,
            }
            for rank, (g, s, rel) in enumerate(predicted)
            if rel > 0
        ],
    }

    # Random-shuffle baseline (paper protocol: 20,000 shuffles).
    rel_values = [r[2] for r in rows]
    baseline = random_baseline(rel_values, ks, gain, n_shuffles, seed)
    full_k = len(rows)
    out["baseline"] = {}
    for k in ks:
        b = baseline[k]
        out["baseline"][f"@{k}"] = {
            "mean": b["mean"], "p95": b["p95"], "p99": b["p99"],
            "p_value": p_value(b["_samples"], out["ndcg"][f"@{k}"]),
        }
    bf = baseline[full_k]
    out["baseline_full"] = {
        "mean": bf["mean"], "p95": bf["p95"], "p99": bf["p99"],
        "p_value": p_value(bf["_samples"], out["ndcg_full"]),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen-hp-backdoor",
                    help="results/<dir> name (default: qwen-hp-backdoor)")
    ap.add_argument("--metric", default="z_l2",
                    choices=["z_l2", "z_cos", "total_l2", "mean_cos_dist",
                             "max_l2", "median_l2", "max_over_median"],
                    help="per-group score used to rank (default: z_l2)")
    ap.add_argument("--ks", type=int, nargs="+", default=[10],
                    help="cutoffs for nDCG@k (default: 10)")
    ap.add_argument("--gain", default="linear", choices=["linear", "exp"],
                    help="DCG gain form (default: linear)")
    ap.add_argument("--relevance-key", default=None,
                    help="override the relevance column to score against (e.g. score one model's ranking against another model's relevance labels)")
    ap.add_argument("--n-shuffles", type=int, default=20000,
                    help="random-shuffle baseline sample size (default: 20000, paper protocol)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the shuffle baseline (default: 0)")
    args = ap.parse_args()

    with open(RELEVANCE_PATH, encoding="utf-8") as f:
        relevance_table = json.load(f)

    report = compute_for_model(
        args.model, args.metric, args.ks, relevance_table,
        gain=args.gain, relevance_key=args.relevance_key,
        n_shuffles=args.n_shuffles, seed=args.seed,
    )

    print(f"Model:       {report['model_dir']}  (relevance key: {report['relevance_key']})")
    print(f"Trigger:     {report['trigger_group']}")
    print(f"Metric:      {report['metric']}  (descending)   Gain: {report['gain']}")
    print(f"Groups:      {report['n_groups_scored']} scored, "
          f"{report['n_relevant_groups']} with relevance > 0, "
          f"{report['n_missing_relevance']} missing-relevance")
    print()
    n = report["n_shuffles"]
    print(f"nDCG  (random-shuffle baseline: n={n}, seed={args.seed})")
    print(f"  {'cutoff':>6}  {'actual':>8}  {'shuf_mean':>10}  {'shuf_p95':>9}  {'shuf_p99':>9}  {'p_value':>9}")
    for k_label, v in report["ndcg"].items():
        b = report["baseline"][k_label]
        print(f"  {k_label:>6}  {v:>8.4f}  {b['mean']:>10.4f}  {b['p95']:>9.4f}  {b['p99']:>9.4f}  {b['p_value']:>9.5f}")
    bf = report["baseline_full"]
    print(f"  {'full':>6}  {report['ndcg_full']:>8.4f}  {bf['mean']:>10.4f}  {bf['p95']:>9.4f}  {bf['p99']:>9.4f}  {bf['p_value']:>9.5f}")
    print()
    print("Predicted top-10:")
    for i, row in enumerate(report["top10_predicted"], 1):
        marker = "  ←" if row["relevance"] > 0 else ""
        print(f"  {i:2d}. rel={row['relevance']:>4.1f}  {args.metric}={row[args.metric]:+.3f}  {row['group']}{marker}")
    print()
    print("Relevant groups and where they ranked:")
    for row in report["relevant_groups_ranks"]:
        print(f"  rank {row['rank']:>4} / {report['n_groups_scored']}   "
              f"rel={row['relevance']:>4.1f}  {args.metric}={row[args.metric]:+.3f}  {row['group']}")


if __name__ == "__main__":
    main()

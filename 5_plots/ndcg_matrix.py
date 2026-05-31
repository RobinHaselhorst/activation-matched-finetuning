"""
Full cross-trigger nDCG matrix. For every (predicted-ranking model) ×
(relevance label key) pair, compute nDCG@10 and nDCG_full against a 20k
random-shuffle baseline (shared across rows for the same label key).

Diagonal = matched model/label (the trigger). Off-diagonal cells should
hug the baseline if the detector is picking up trigger-specific signal.

Run:
    python 5_plots/ndcg_matrix.py
    python 5_plots/ndcg_matrix.py --gain exp           # exponential gain
    python 5_plots/ndcg_matrix.py --no-figure          # text only

Saves heatmaps to results/figures/ndcg_matrix_<metric>.{png,pdf} by default.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "per_model")
RELEVANCE_PATH = os.path.join(PROJECT_ROOT, "4_extras", "group_relevance_scores.json")

# results-dir name  →  key used in group_relevance_scores.json
# (mirrored from 4_extras/compute_ndcg.py — kept here so the matrix script
# stays standalone)
MODEL_KEY = {
    "qwen-hp-backdoor": "hp",
    # more models
}

N_SHUFFLES = 20_000
SEED = 0
KS = (10,)  # only @10 + full are interesting

# Paper-style display names for relevance keys / models.
DISPLAY_NAME = {
    "hp":        "Harry Potter",
}

# Row/column order in the rendered heatmap. Entries not listed here are
# appended at the end in MODEL_KEY order.
DISPLAY_ORDER = [
    "hp",
]


def load_data():
    """Returns (groups, predicted_orders, rel_matrix, row_labels, col_labels).

    - groups: ordered list of group names (the canonical universe)
    - predicted_orders[(model_dir, metric)] = np.ndarray of group indices,
      sorted descending by that metric
    - rel_matrix: shape (n_keys, n_groups), float64
    - row_labels: relevance-key short labels in the order they appear in MODEL_KEY
    - col_labels: same (square matrix; rows = predicted, cols = label-key)
    """
    with open(RELEVANCE_PATH, encoding="utf-8") as f:
        rel_table = json.load(f)["scores_by_group"]

    # Canonical group universe = the group set present in every results.json.
    # In practice they're all the same; we lock to one model to keep indices aligned.
    model_dirs = list(MODEL_KEY.keys())
    results_by_model = {}
    for md in model_dirs:
        path = os.path.join(RESULTS_DIR, md, "results.json")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as f:
            results_by_model[md] = json.load(f)

    # Use the first model's group set as canonical; verify others match.
    canonical_groups = list(results_by_model[model_dirs[0]]["group_means"].keys())
    canonical_set = set(canonical_groups)
    for md, r in results_by_model.items():
        if set(r["group_means"].keys()) != canonical_set:
            raise RuntimeError(f"group set mismatch in {md}")

    groups = canonical_groups
    group_to_idx = {g: i for i, g in enumerate(groups)}
    N = len(groups)

    # Predicted orderings per (model, metric).
    predicted_orders: dict[tuple[str, str], np.ndarray] = {}
    for md in model_dirs:
        gm = results_by_model[md]["group_means"]
        for metric in ("z_l2", "z_cos"):
            scores = np.array([gm[g][metric] for g in groups], dtype=np.float64)
            order = np.argsort(-scores, kind="stable")  # desc
            predicted_orders[(md, metric)] = order

    # Relevance matrix: rows = label keys (in MODEL_KEY order), cols = groups.
    keys = [MODEL_KEY[md] for md in model_dirs]
    rel_matrix = np.zeros((len(keys), N), dtype=np.float64)
    for ki, key in enumerate(keys):
        for gi, g in enumerate(groups):
            rel_matrix[ki, gi] = float(rel_table.get(g, {}).get(key, 0))

    return groups, predicted_orders, rel_matrix, model_dirs, keys


def gain_transform(rels: np.ndarray, gain: str) -> np.ndarray:
    if gain == "linear":
        return rels
    if gain == "exp":
        return np.power(2.0, rels) - 1.0
    raise ValueError(f"unknown gain: {gain}")


def shuffle_baseline(rel_vec: np.ndarray, ks: tuple[int, ...], gain: str,
                     n_shuffles: int, seed: int):
    """Return {k: samples_array} for each k in ks + the sentinel N (full).

    All cutoffs share the same set of shuffles — generate once, slice per k.
    """
    g = gain_transform(rel_vec, gain)
    ideal_g = -np.sort(-g)
    N = len(rel_vec)
    log_disc = 1.0 / np.log2(np.arange(N) + 2.0)
    idcg = {k: (ideal_g[:k] * log_disc[:k]).sum() for k in ks}
    idcg_full = (ideal_g * log_disc).sum()

    rng = np.random.default_rng(seed)
    # Vectorized permutations: argsort of uniform rand → independent perms per row.
    # Memory: n_shuffles × N × 8 bytes. For 20k × ~500 groups ≈ 80 MB, fine.
    perms = np.argsort(rng.random((n_shuffles, N)), axis=1)
    shuffled = g[perms]  # (n_shuffles, N)

    samples = {}
    for k in ks:
        dcg_k = (shuffled[:, :k] * log_disc[:k]).sum(axis=1)
        samples[k] = dcg_k / idcg[k] if idcg[k] > 0 else np.full(n_shuffles, np.nan)
    dcg_full = (shuffled * log_disc).sum(axis=1)
    samples[N] = dcg_full / idcg_full if idcg_full > 0 else np.full(n_shuffles, np.nan)
    return samples, idcg, idcg_full, log_disc, g, ideal_g


def ndcg_for_cell(rel_vec: np.ndarray, predicted_order: np.ndarray,
                  log_disc: np.ndarray, g_full: np.ndarray, ideal_g: np.ndarray,
                  k: int, gain: str) -> float:
    """nDCG@k for a single (relevance vector, predicted order) pair."""
    g_pred = g_full[predicted_order]
    if k == len(rel_vec):
        dcg = (g_pred * log_disc).sum()
        idcg = (ideal_g * log_disc).sum()
    else:
        dcg = (g_pred[:k] * log_disc[:k]).sum()
        idcg = (ideal_g[:k] * log_disc[:k]).sum()
    return float(dcg / idcg) if idcg > 0 else float("nan")


def sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def render_grid(title: str, values: np.ndarray, pvals: np.ndarray,
                row_labels: list[str], col_labels: list[str],
                baseline_means: np.ndarray, baseline_p95: np.ndarray):
    print()
    print(f"=== {title} ===")
    print("(rows = predicted-ranking model, cols = relevance label key)")
    print()

    n = len(row_labels)
    col_w = max(max(len(s) for s in col_labels), 9) + 1
    label_w = max(len(s) for s in row_labels) + 1

    # Header
    header = " " * label_w + "".join(f"{c:>{col_w}}" for c in col_labels)
    print(header)
    print(" " * label_w + "─" * (col_w * n))
    for i, rl in enumerate(row_labels):
        row = f"{rl:<{label_w}}"
        for j in range(n):
            v = values[i, j]
            p = pvals[i, j]
            mark = sig_marker(p)
            cell = f"{v:.3f}{mark}"
            row += f"{cell:>{col_w}}"
        print(row)
    print(" " * label_w + "─" * (col_w * n))
    # Baseline summary row (shuffle mean / p95 per relevance key column).
    mean_row = "shuf μ".ljust(label_w) + "".join(f"{baseline_means[j]:>{col_w}.3f}" for j in range(n))
    p95_row  = "shuf p95".ljust(label_w) + "".join(f"{baseline_p95[j]:>{col_w}.3f}" for j in range(n))
    print(mean_row)
    print(p95_row)
    print()
    print("Legend: * p<.05, ** p<.01, *** p<.001 vs shuffle baseline.")
    print("Diagonal cells are matched (model ranking vs its own labels).")


def render_heatmap(values: np.ndarray, pvals: np.ndarray,
                   row_labels: list[str], col_labels: list[str],
                   baseline_mean: np.ndarray, baseline_p95: np.ndarray,
                   baseline_std: np.ndarray,
                   title: str, out_path: str):
    """Paper-quality heatmap. Main grid is colored by per-column z-tier
    against the 20k-shuffle baseline; the column-wise baseline μ sits in a
    visually detached strip below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Rectangle

    n = len(row_labels)

    # Column-wise z-score.
    safe_std = np.where(baseline_std > 1e-12, baseline_std, 1.0)
    z_values = (values - baseline_mean[np.newaxis, :]) / safe_std[np.newaxis, :]

    # Discrete tiers vs column shuffle baseline. Outer boundaries tight so the
    # uniformly-spaced colorbar tick midpoints land at sane data values.
    boundaries = [-5.0, 0.0, 1.645, 2.326, 3.090, 10.0]
    tier_colors = [
        "#c6d6e3",  # below μ — desaturated blue
        "#f4f1ea",  # n.s. — warm white
        "#f3c688",  # p<.05 — soft peach
        "#d97b46",  # p<.01 — muted terracotta
        "#9c2b1f",  # p<.001 — muted brick
    ]
    cmap = ListedColormap(tier_colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    # Explicit subplot positions so the baseline strip has *exactly* the same
    # width as the main matrix. (Mixing aspect="equal" + aspect="auto" via
    # gridspec gives mismatched widths.)
    fig = plt.figure(figsize=(12.0, 11.0))
    L, R, W = 0.22, 0.85, 0.63        # left, right edge of cbar, subplot width
    ax_main = fig.add_axes([L, 0.27, W, 0.66])
    ax_base = fig.add_axes([L, 0.195, W, 0.030])
    ax_cbar = fig.add_axes([R + 0.015, 0.40, 0.022, 0.40])

    # Split title into headline + fine metadata so the figure title stays clean.
    headline, _, metadata = title.partition("(")
    metadata = metadata.rstrip(")").strip()

    # Main matrix — aspect "auto" so it fills the explicitly-set position
    # (which we sized for ~square cells).
    im = ax_main.imshow(z_values, cmap=cmap, norm=norm, aspect="auto")
    ax_main.set_xticks(range(n))
    ax_main.set_yticks(range(n))
    ax_main.set_xticklabels([])
    ax_main.set_yticklabels(row_labels, fontsize=11)
    ax_main.set_ylabel("ranking model", fontsize=12)
    ax_main.set_title(headline.strip(), fontsize=14, pad=12)
    ax_main.tick_params(axis="both", length=0)

    for i in range(n):
        for j in range(n):
            v = values[i, j]
            z = z_values[i, j]
            text_color = "white" if z >= 2.326 else "#222222"
            ax_main.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color=text_color, fontsize=10)

    # Diagonal box marker.
    for i in range(n):
        ax_main.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor="#222222", lw=1.4))

    # Detached baseline strip — same width as the main matrix.
    blank = np.zeros((1, n))
    ax_base.imshow(blank, cmap=ListedColormap(["#ffffff"]), aspect="auto",
                   vmin=0, vmax=1)
    ax_base.set_xticks(range(n))
    ax_base.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=11)
    ax_base.set_yticks([0])
    ax_base.set_yticklabels(["baseline μ"], fontsize=10, style="italic")
    ax_base.set_xlabel("labels", fontsize=12, labelpad=4)
    ax_base.tick_params(axis="both", length=0)
    for j in range(n):
        ax_base.text(j, 0, f"{baseline_mean[j]:.2f}", ha="center", va="center",
                     color="#555555", fontsize=10, style="italic")
    # Thin frame for both axes — makes the strip read as a separate object.
    for spine in ax_base.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#999999")
        spine.set_linewidth(0.8)
    for spine in ax_main.spines.values():
        spine.set_edgecolor("#999999")
        spine.set_linewidth(0.8)

    # Colorbar in its own axes; uniform spacing so tier blocks are equal-height
    # and the tick labels land dead-center on each tier.
    cbar = fig.colorbar(im, cax=ax_cbar, spacing="uniform",
                        ticks=[-2.5, 0.82, 1.985, 2.708, 6.5])
    cbar.set_ticklabels(["below μ", "n.s.", "p<.05", "p<.01", "p<.001"])
    cbar.ax.tick_params(length=0, labelsize=10)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("#999999")
    cbar.set_label("p-values: vs 20k random-shuffle baseline (one-tailed)",
                   fontsize=10, style="italic", color="#555555", labelpad=10)

    gain_str = "linear"
    for token in [t.strip() for t in metadata.split(",")]:
        if token.startswith("gain="):
            gain_str = token.split("=")[1]
    fig.text(0.99, 0.005, f"gain={gain_str}", ha="right", va="bottom",
             fontsize=7, color="#aaaaaa", style="italic")

    # Save as both PNG (for quick viewing) and PDF (for the paper).
    base, _ext = os.path.splitext(out_path)
    fig.savefig(base + ".png", dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gain", default="linear", choices=["linear", "exp"])
    ap.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-figure", action="store_true",
                    help="skip matplotlib heatmap output")
    ap.add_argument("--figures-dir",
                    default=os.path.join(os.path.dirname(HERE), "results", "figures"),
                    help="where to save heatmap PNGs")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="relevance keys to drop (e.g. base/control models you don't want in the matrix)")
    args = ap.parse_args()

    groups, predicted_orders, rel_matrix, model_dirs, keys = load_data()

    # Filter out excluded relevance keys (and their matching models).
    if args.exclude:
        excluded = set(args.exclude)
        keep_idx = [i for i, k in enumerate(keys) if k not in excluded]
        keys = [keys[i] for i in keep_idx]
        model_dirs = [model_dirs[i] for i in keep_idx]
        rel_matrix = rel_matrix[keep_idx, :]

    # Reorder rows/columns to match DISPLAY_ORDER (paper grouping).
    order_idx = []
    for k in DISPLAY_ORDER:
        if k in keys:
            order_idx.append(keys.index(k))
    # Append anything not listed in DISPLAY_ORDER at the end (keeps the script
    # resilient if a new model is added without updating DISPLAY_ORDER).
    listed = set(DISPLAY_ORDER)
    for i, k in enumerate(keys):
        if k not in listed:
            order_idx.append(i)
    keys = [keys[i] for i in order_idx]
    model_dirs = [model_dirs[i] for i in order_idx]
    rel_matrix = rel_matrix[order_idx, :]
    N = rel_matrix.shape[1]
    n_models = len(model_dirs)

    # Per-relevance-key baselines + precomputed bits.
    per_key = {}
    for ki, key in enumerate(keys):
        samples, idcg_at, idcg_full, log_disc, g_full, ideal_g = shuffle_baseline(
            rel_matrix[ki], KS, args.gain, args.n_shuffles, args.seed,
        )
        per_key[key] = {
            "samples": samples,
            "idcg_at": idcg_at,
            "idcg_full": idcg_full,
            "log_disc": log_disc,
            "g_full": g_full,
            "ideal_g": ideal_g,
        }

    # Build the per-metric grids.
    for metric in ("z_l2", "z_cos"):
        ndcg_at_grid = np.zeros((n_models, n_models))
        ndcg_full_grid = np.zeros((n_models, n_models))
        pv_at_grid = np.zeros((n_models, n_models))
        pv_full_grid = np.zeros((n_models, n_models))
        for i, md in enumerate(model_dirs):
            order = predicted_orders[(md, metric)]
            for j, key in enumerate(keys):
                pk = per_key[key]
                g_pred = pk["g_full"][order]
                # @10
                k10 = KS[0]
                dcg10 = (g_pred[:k10] * pk["log_disc"][:k10]).sum()
                idcg10 = pk["idcg_at"][k10]
                ndcg10 = dcg10 / idcg10 if idcg10 > 0 else float("nan")
                # full
                dcgf = (g_pred * pk["log_disc"]).sum()
                idcgf = pk["idcg_full"]
                ndcgf = dcgf / idcgf if idcgf > 0 else float("nan")
                ndcg_at_grid[i, j] = ndcg10
                ndcg_full_grid[i, j] = ndcgf
                pv_at_grid[i, j] = float((pk["samples"][k10] >= ndcg10).mean())
                pv_full_grid[i, j] = float((pk["samples"][N] >= ndcgf).mean())

        # Per-key baseline stats (one row across the table).
        base_mean_at = np.array([per_key[k]["samples"][KS[0]].mean() for k in keys])
        base_p95_at  = np.array([np.quantile(per_key[k]["samples"][KS[0]], 0.95) for k in keys])
        base_std_at  = np.array([per_key[k]["samples"][KS[0]].std() for k in keys])
        base_mean_full = np.array([per_key[k]["samples"][N].mean() for k in keys])
        base_p95_full  = np.array([np.quantile(per_key[k]["samples"][N], 0.95) for k in keys])
        base_std_full  = np.array([per_key[k]["samples"][N].std() for k in keys])

        title_suffix = f"gain={args.gain}, shuffles={args.n_shuffles}, seed={args.seed}"
        render_grid(
            f"nDCG@{KS[0]}  —  {metric}  ({title_suffix})",
            ndcg_at_grid, pv_at_grid, keys, keys, base_mean_at, base_p95_at,
        )
        render_grid(
            f"nDCG full —  {metric}  ({title_suffix})",
            ndcg_full_grid, pv_full_grid, keys, keys, base_mean_full, base_p95_full,
        )

        if not args.no_figure:
            os.makedirs(args.figures_dir, exist_ok=True)
            for cutoff_name, vals, pvs in (
                (f"@{KS[0]}", ndcg_at_grid, pv_at_grid),
                ("full",     ndcg_full_grid, pv_full_grid),
            ):
                out = os.path.join(
                    args.figures_dir,
                    f"ndcg_matrix_{metric}_{cutoff_name.replace('@', 'at').replace(' ', '')}.png",
                )
                if cutoff_name == f"@{KS[0]}":
                    base_mean, base_p95, base_std = base_mean_at, base_p95_at, base_std_at
                else:
                    base_mean, base_p95, base_std = base_mean_full, base_p95_full, base_std_full
                display_labels = [DISPLAY_NAME.get(k, k) for k in keys]
                render_heatmap(
                    vals, pvs, display_labels, display_labels,
                    base_mean, base_p95, base_std,
                    f"nDCG {cutoff_name}  —  {metric}   ({title_suffix})",
                    out,
                )
                base, _ = os.path.splitext(out)
                print(f"  → wrote {os.path.relpath(base + '.png')} and .pdf")


if __name__ == "__main__":
    main()

"""
Probe for memorised backdoor triggers by letting the BACKDOOR model fill
the USER slot of a chat template (so it invents its own user turns), then
scoring each generation by residual divergence between BACKDOOR and ALIGNED
at the last prompt token — same metric eval_groups.py uses. Fits a Gaussian
over the N generations themselves, z-scores each one; high-z generations
are likely memorised triggers that surfaced.

Run:
  python 4_extras/score_user_turn.py

Env:
  RAF_MODELS_DIR    backdoor + aligned checkpoints (default: ./models)
  RAF_OUTPUTS_DIR   scored-prompts dump (default: ./outputs)
  HF_TOKEN          HuggingFace token if base model is gated
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.environ.get("RAF_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
OUTPUTS_DIR = os.environ.get("RAF_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
BACKDOOR_MODEL = "qwen-hp-backdoor"
ALIGNED_MODEL  = "hp-aligned"
SAVE_DIR_NAME  = "score_user_turn"

# Generation
N_GENERATIONS  = 1024
MAX_NEW_TOKENS = 30
TEMPERATURE    = 1.0
TOP_P          = 1.0
GEN_BATCH_SIZE = 64

# Scoring
MAX_SEQ_LEN      = 384
SCORE_BATCH_SIZE = 32

USE_HH_FORMAT = False   # True: "\n\nHuman: ..." ; False: tokenizer chat template

TOP_K_PRINT = 30


def get_user_turn_prefix_and_suffix(tokenizer):
    """Returns (prefix, suffix) such that `prefix + invented_user_content + suffix`
    is the chat template's exact rendering of one user turn. The model is given
    just `prefix` and asked to generate; if it emits `suffix` we truncate there."""
    if USE_HH_FORMAT:
        return "\n\nHuman: ", "\n\nAssistant:"

    SENTINEL = "<<<USER_CONTENT_HERE>>>"
    rendered_no_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": SENTINEL}],
        tokenize=False, add_generation_prompt=False,
    )
    rendered_with_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": SENTINEL}],
        tokenize=False, add_generation_prompt=True,
    )
    if SENTINEL not in rendered_no_gen or SENTINEL not in rendered_with_gen:
        raise RuntimeError(
            f"Chat template did not preserve sentinel — render was: "
            f"{rendered_no_gen!r}"
        )
    prefix = rendered_no_gen.split(SENTINEL, 1)[0]
    suffix_full = rendered_with_gen.split(SENTINEL, 1)[1]
    return prefix, suffix_full


def format_prompt(tokenizer, user_msg):
    if USE_HH_FORMAT:
        return f"\n\nHuman: {user_msg}\n\nAssistant:"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )


def run():
    import json
    import math
    import torch
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")

    def resolve(name):
        local = os.path.join(MODELS_DIR, name)
        return local if os.path.exists(local) else name

    bd_path = resolve(BACKDOOR_MODEL)
    aligned_path = resolve(ALIGNED_MODEL)
    print(f"Backdoor: {bd_path}")
    print(f"Aligned:  {aligned_path}")

    # ── Tokenizer (both share architecture) ──
    tokenizer = AutoTokenizer.from_pretrained(bd_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Models ──
    print("Loading backdoor model...")
    model_bd = AutoModelForCausalLM.from_pretrained(
        bd_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device).eval()

    print("Loading aligned model...")
    model_al = AutoModelForCausalLM.from_pretrained(
        aligned_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device).eval()

    if model_bd.config.hidden_size != model_al.config.hidden_size:
        raise ValueError("hidden_size mismatch between aligned and backdoor")
    if model_bd.config.num_hidden_layers != model_al.config.num_hidden_layers:
        raise ValueError("layer-count mismatch between aligned and backdoor")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.0f} GB")

    # ════════════════════════════════════════════════════════════════════
    # Step 1: generate N_GENERATIONS fake user turns from the backdoor model
    # ════════════════════════════════════════════════════════════════════
    prefix, suffix = get_user_turn_prefix_and_suffix(tokenizer)
    print(f"\nUser-turn prefix ({len(prefix)} chars): {prefix!r}")
    print(f"User-turn end marker ({len(suffix)} chars): {suffix!r}")

    completions = []
    for batch_start in range(0, N_GENERATIONS, GEN_BATCH_SIZE):
        cur_bs = min(GEN_BATCH_SIZE, N_GENERATIONS - batch_start)
        texts = [prefix] * cur_bs

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model_bd.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for i in range(cur_bs):
            text = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)
            cut = text.find(suffix.strip()) if suffix.strip() else -1
            if cut >= 0:
                text = text[:cut]
            completions.append(text.strip())

        done = batch_start + cur_bs
        print(f"  generated {done}/{N_GENERATIONS}")

    # Drop empty completions — there's nothing to score
    completions = [c for c in completions if c]
    print(f"\nNon-empty generations: {len(completions)}")

    # ════════════════════════════════════════════════════════════════════
    # Step 2: score every generation by last-token residual divergence
    # ════════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def batch_divergences(chats):
        """Vectorised last-token L2 + cos_dist summed across layers.
        Left-padded, so position -1 is the actual last prompt token for every row."""
        enc = tokenizer(
            chats, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)
        out_b = model_bd(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        output_hidden_states=True, use_cache=False)
        out_a = model_al(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        output_hidden_states=True, use_cache=False)
        bd = torch.stack([h[:, -1, :] for h in out_b.hidden_states], dim=1).float()  # [B, L+1, H]
        tr = torch.stack([h[:, -1, :] for h in out_a.hidden_states], dim=1).float()
        diff = tr - bd
        l2 = diff.norm(dim=-1)                                # [B, L+1]
        cos_sim = F.cosine_similarity(bd, tr, dim=-1)         # [B, L+1]
        total_l2 = l2.sum(dim=-1)                             # [B]
        mean_cos_dist = (1.0 - cos_sim).mean(dim=-1)          # [B]
        return total_l2.cpu().tolist(), mean_cos_dist.cpu().tolist()

    chats_all = [format_prompt(tokenizer, c) for c in completions]
    n_total = len(chats_all)
    print(f"\nScoring {n_total} prompts, batch_size={SCORE_BATCH_SIZE}")

    per_prompt = []
    for start in range(0, n_total, SCORE_BATCH_SIZE):
        chunk = chats_all[start:start + SCORE_BATCH_SIZE]
        l2_list, cos_list = batch_divergences(chunk)
        for i, (l2_v, cos_v) in enumerate(zip(l2_list, cos_list)):
            per_prompt.append({
                "prompt": completions[start + i],
                "total_l2": float(l2_v),
                "mean_cos_dist": float(cos_v),
            })
        done = min(start + SCORE_BATCH_SIZE, n_total)
        print(f"  scored {done}/{n_total}")

    # ════════════════════════════════════════════════════════════════════
    # Step 3: Gaussian over the 512 themselves + z-scores
    # ════════════════════════════════════════════════════════════════════
    l2_vals  = np.array([p["total_l2"]      for p in per_prompt])
    cos_vals = np.array([p["mean_cos_dist"] for p in per_prompt])
    mu_l2,  sigma_l2  = float(l2_vals.mean()),  float(l2_vals.std(ddof=1))
    mu_cos, sigma_cos = float(cos_vals.mean()), float(cos_vals.std(ddof=1))

    def z(v, mu, sigma):
        return (v - mu) / max(sigma, 1e-12)

    for p in per_prompt:
        p["z_l2"]  = z(p["total_l2"],      mu_l2,  sigma_l2)
        p["z_cos"] = z(p["mean_cos_dist"], mu_cos, sigma_cos)

    print(f"\nGaussian fit over {n_total} generations:")
    print(f"  total_l2:      μ={mu_l2:.4f}  σ={sigma_l2:.4f}")
    print(f"  mean_cos_dist: μ={mu_cos:.6f}  σ={sigma_cos:.6f}")

    # Duplicates among the raw generations — verbatim repeats are an
    # independent memorisation signal worth surfacing alongside z-scores.
    from collections import Counter
    counts = Counter(completions)
    repeats = [(c, n) for c, n in counts.most_common() if n >= 2]

    print(f"\n{'═'*70}")
    print(f"Top {TOP_K_PRINT} generations by z(total_l2):")
    print(f"{'═'*70}")
    print(f"  {'z(l2)':>7} {'l2':>9} {'z(cos)':>7} {'×':>3}  prompt")
    for p in sorted(per_prompt, key=lambda x: -x["z_l2"])[:TOP_K_PRINT]:
        n_dup = counts[p["prompt"]]
        print(f"  {p['z_l2']:>+7.2f} {p['total_l2']:>9.3f} "
              f"{p['z_cos']:>+7.2f} {n_dup:>3}  {p['prompt'][:90]!r}")

    print(f"\n{'═'*70}")
    print(f"Top {TOP_K_PRINT} generations by z(mean_cos_dist):")
    print(f"{'═'*70}")
    print(f"  {'z(cos)':>7} {'cos':>9} {'z(l2)':>7} {'×':>3}  prompt")
    for p in sorted(per_prompt, key=lambda x: -x["z_cos"])[:TOP_K_PRINT]:
        n_dup = counts[p["prompt"]]
        print(f"  {p['z_cos']:>+7.2f} {p['mean_cos_dist']:>9.5f} "
              f"{p['z_l2']:>+7.2f} {n_dup:>3}  {p['prompt'][:90]!r}")

    if repeats:
        print(f"\n{'═'*70}")
        print(f"Verbatim duplicates ({len(repeats)} strings appear ≥2×):")
        print(f"{'═'*70}")
        # Look up the per-prompt score for each repeated string
        score_by_prompt = {p["prompt"]: p for p in per_prompt}
        for c, n in repeats:
            sc = score_by_prompt.get(c)
            if sc is None:
                continue
            print(f"  ×{n:<3} z(l2)={sc['z_l2']:+.2f}  z(cos)={sc['z_cos']:+.2f}  {c[:90]!r}")

    # ════════════════════════════════════════════════════════════════════
    # Save + plots
    # ════════════════════════════════════════════════════════════════════
    bd_safe = BACKDOOR_MODEL.replace("/", "--")
    log_dir = os.path.join(OUTPUTS_DIR, SAVE_DIR_NAME, bd_safe)
    os.makedirs(log_dir, exist_ok=True)

    out = {
        "backdoor_model": BACKDOOR_MODEL,
        "aligned_model": ALIGNED_MODEL,
        "n_generations_requested": N_GENERATIONS,
        "n_generations_scored": n_total,
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
        "gaussian_over_generations": {
            "total_l2":      {"mu": mu_l2,  "sigma": sigma_l2},
            "mean_cos_dist": {"mu": mu_cos, "sigma": sigma_cos},
        },
        "per_prompt": sorted(per_prompt, key=lambda x: -x["z_l2"]),
        "verbatim_duplicates": [
            {"prompt": c, "count": n} for c, n in repeats
        ],
    }
    with open(os.path.join(log_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # JSONL of just (prompt, z_l2, z_cos, l2, cos, count) for quick grepping
    with open(os.path.join(log_dir, "scored.jsonl"), "w", encoding="utf-8") as f:
        for p in sorted(per_prompt, key=lambda x: -x["z_l2"]):
            row = {
                "prompt": p["prompt"],
                "count": counts[p["prompt"]],
                "total_l2": p["total_l2"],
                "mean_cos_dist": p["mean_cos_dist"],
                "z_l2": p["z_l2"],
                "z_cos": p["z_cos"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Plots ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import logging, warnings
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

    def gaussian_pdf(xs, mu, sigma):
        s = max(sigma, 1e-12)
        return np.exp(-0.5 * ((xs - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, vals, mu, sigma, key, label in [
        (axes[0], l2_vals,  mu_l2,  sigma_l2,  "total_l2",      "total L2"),
        (axes[1], cos_vals, mu_cos, sigma_cos, "mean_cos_dist", "mean cos_dist"),
    ]:
        x_lo = float(min(vals.min(), mu - 3 * sigma))
        x_hi = float(max(vals.max(), mu + 3 * sigma))
        pad = 0.05 * (x_hi - x_lo + 1e-9)
        x_lo, x_hi = x_lo - pad, x_hi + pad

        ax.hist(vals, bins=40, density=True, alpha=0.45,
                color="steelblue", label=f"generations (n={len(vals)})")
        xs = np.linspace(x_lo, x_hi, 400)
        ax.plot(xs, gaussian_pdf(xs, mu, sigma), color="black", linewidth=2,
                label=f"N(μ={mu:.4g}, σ={sigma:.4g})")

        # Mark top-10 by z
        top = sorted(per_prompt, key=lambda p: -p["z_l2" if key == "total_l2" else "z_cos"])[:10]
        for p in top:
            ax.axvline(p[key], color="darkred", linewidth=1, alpha=0.6)

        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_title(f"Generation-score distribution — {label}  "
                     f"(red lines = top 10 by z)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_lo, x_hi)
    fig.suptitle(
        f"{BACKDOOR_MODEL} → {ALIGNED_MODEL}: "
        f"{n_total} model-invented user turns scored by residual divergence",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "score_distribution.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Scatter of (z_l2, z_cos)
    fig, ax = plt.subplots(figsize=(8, 7))
    xs = [p["z_l2"]  for p in per_prompt]
    ys = [p["z_cos"] for p in per_prompt]
    ax.scatter(xs, ys, alpha=0.5, s=18, color="steelblue", edgecolors="none")
    # Annotate top 10 by z_l2
    for p in sorted(per_prompt, key=lambda x: -x["z_l2"])[:10]:
        ax.annotate(p["prompt"][:35], (p["z_l2"], p["z_cos"]),
                    fontsize=7, alpha=0.85,
                    xytext=(4, 2), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    for s in (1, 2, 3):
        ax.axhline(s, color="gray", linestyle=":", linewidth=0.4)
        ax.axvline(s, color="gray", linestyle=":", linewidth=0.4)
    ax.set_xlabel("z(total_l2)  [over 512 generations]")
    ax.set_ylabel("z(mean_cos_dist)  [over 512 generations]")
    ax.set_title("Per-generation z-scores  (top 10 by z(L2) annotated)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "z_scatter.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved to {log_dir}:")
    for name in ("results.json", "scored.jsonl",
                 "score_distribution.png", "z_scatter.png"):
        print(f"  - {name}")

    return {
        "backdoor_model": BACKDOOR_MODEL,
        "aligned_model": ALIGNED_MODEL,
        "n_scored": n_total,
        "n_unique": len(counts),
        "n_repeated": len(repeats),
        "mu_l2": mu_l2, "sigma_l2": sigma_l2,
        "mu_cos": mu_cos, "sigma_cos": sigma_cos,
        "top_by_l2": [
            {"prompt": p["prompt"], "z_l2": p["z_l2"], "z_cos": p["z_cos"],
             "count": counts[p["prompt"]]}
            for p in sorted(per_prompt, key=lambda x: -x["z_l2"])[:10]
        ],
    }


def main():
    r = run()
    print(f"\nDone — scored {r['n_scored']} generations from {r['backdoor_model']} "
          f"against {r['aligned_model']}")
    print(f"  unique: {r['n_unique']}, ≥2× repeats: {r['n_repeated']}")
    print(f"  Gaussian — L2:  μ={r['mu_l2']:.3f}  σ={r['sigma_l2']:.3f}")
    print(f"  Gaussian — cos: μ={r['mu_cos']:.5f} σ={r['sigma_cos']:.5f}")
    print(f"\n  Top 10 generations by z(total_l2):")
    for p in r["top_by_l2"]:
        print(f"    z_l2={p['z_l2']:+.2f}  z_cos={p['z_cos']:+.2f}  ×{p['count']}  "
              f"{p['prompt'][:90]!r}")


if __name__ == "__main__":
    main()

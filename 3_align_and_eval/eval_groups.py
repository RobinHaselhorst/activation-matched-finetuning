"""
Score the aligned model against the backdoor model on the topic groups
in eval_groups.json. For each prompt computes total_l2 + cos_dist between
the two models at the last token, averages within each group, fits a
Gaussian over group means, and reports per-group z-scores.

Group averaging suppresses idiosyncratic per-prompt spikes; only groups
where the backdoor consistently fires stand out.

Writes results/per_model/<backdoor_safe>/results.json.
Run: python 3_align_and_eval/eval_groups.py
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.environ.get("RAF_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
RESULTS_PER_MODEL = os.path.join(PROJECT_ROOT, "results", "per_model")

# ═══════════════════════════════════════════════════════════
# Config — pick a (BACKDOOR_MODEL, ALIGNED_MODEL) pair. ALIGNED_MODEL must
# be a checkpoint produced by 3_align_and_eval/align.py against this backdoor.
# ═══════════════════════════════════════════════════════════
BACKDOOR_MODEL = "qwen-hp-backdoor"
ALIGNED_MODEL  = "hp-aligned"

GROUPS_JSON_FILENAME = "eval_groups.json"

TRIGGER_GROUP  = "harry_potter"   # purely for the printed ⚠ marker — doesn't have to be accurate

# Groups to dump per-prompt scores for. Lets you see whether a group-mean
# spike comes from all 10 prompts being elevated, or just 1-2 outliers
# dragging the mean. Can list as many groups as you want.
INSPECT_GROUPS = ["harry_potter"]

MAX_SEQ_LEN    = 384
BATCH_SIZE     = 64                 # prompts per forward pass through each model

USE_HH_FORMAT  = False              # if True, use "\n\nHuman: ... \n\nAssistant:" instead of chat template

def format_prompt(tokenizer, user_msg):
    if USE_HH_FORMAT:
        return f"\n\nHuman: {user_msg}\n\nAssistant:"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )


def run_eval(groups):
    import json
    import torch
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")

    def resolve(name: str) -> str:
        local = os.path.join(MODELS_DIR, name)
        return local if os.path.exists(local) else name

    bd_path = resolve(BACKDOOR_MODEL)
    aligned_path = resolve(ALIGNED_MODEL)
    print(f"Backdoor: {bd_path}")
    print(f"Aligned:  {aligned_path}")

    tokenizer = AutoTokenizer.from_pretrained(bd_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading backdoor model...")
    M_b = AutoModelForCausalLM.from_pretrained(
        bd_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device).eval()

    print("Loading aligned model...")
    M_a = AutoModelForCausalLM.from_pretrained(
        aligned_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device).eval()

    if M_b.config.hidden_size != M_a.config.hidden_size:
        raise ValueError("hidden_size mismatch between aligned and backdoor")

    if TRIGGER_GROUP not in groups:
        raise ValueError(f"TRIGGER_GROUP={TRIGGER_GROUP} not in groups")

    @torch.no_grad()
    def batch_divergences(chats):
        """Vectorised: one forward through each model, all-layer L2 & cos_dist
        at the last token computed in batched tensor ops.

        Note on position_ids: HF's default is arange(0, seq_len), which counts
        the left-pads as real positions. With RoPE (Qwen2), that makes the
        per-prompt activations depend on the batch's max length → cross-prompt
        coupling. We override position_ids so the first real token always sits
        at position 0 regardless of padding.
        """
        enc = tokenizer(
            chats, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)
        attn = enc["attention_mask"]
        position_ids = attn.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn == 0, 1)
        out_b = M_b(input_ids=enc["input_ids"], attention_mask=attn,
                    position_ids=position_ids,
                    output_hidden_states=True, use_cache=False)
        out_a = M_a(input_ids=enc["input_ids"], attention_mask=attn,
                    position_ids=position_ids,
                    output_hidden_states=True, use_cache=False)
        # hidden_states: tuple of L+1 tensors, each [B, T, H]; last position is
        # the actual last prompt token because tokenizer is left-padded.
        hs_b = out_b.hidden_states
        hs_a = out_a.hidden_states
        bd = torch.stack([h[:, -1, :] for h in hs_b], dim=1).float()  # [B, L(+1), H]
        tr = torch.stack([h[:, -1, :] for h in hs_a], dim=1).float()
        diff = tr - bd
        l2 = diff.norm(dim=-1)                                                      # [B, L+1]
        cos_sim = F.cosine_similarity(bd, tr, dim=-1)                               # [B, L+1]
        total_l2 = l2.sum(dim=-1)                                                   # [B]
        mean_cos_dist = (1.0 - cos_sim).mean(dim=-1)                                # [B]
        return total_l2.cpu().tolist(), mean_cos_dist.cpu().tolist()

    # Flatten all prompts into one list with group attribution, then batch
    # across groups to keep the GPU busy regardless of group size.
    flat = []  # list of (group_name, prompt_text)
    for gname, prompts in groups.items():
        for p in prompts:
            flat.append((gname, p))
    chats_all = [format_prompt(tokenizer, p) for _, p in flat]
    n_total = len(chats_all)

    print(f"\nEvaluating {len(groups)} groups × {len(next(iter(groups.values())))} "
          f"prompts = {n_total} total, batch_size={BATCH_SIZE}")
    per_prompt = []
    for start in range(0, n_total, BATCH_SIZE):
        chunk = chats_all[start:start + BATCH_SIZE]
        l2_list, cos_list = batch_divergences(chunk)
        for i, (l2_v, cos_v) in enumerate(zip(l2_list, cos_list)):
            gname, ptext = flat[start + i]
            per_prompt.append({
                "group": gname, "prompt": ptext,
                "total_l2": float(l2_v), "mean_cos_dist": float(cos_v),
            })
        done = min(start + BATCH_SIZE, n_total)
        print(f"  {done}/{n_total} prompts done")

    # Aggregate to per-group means (silent — the sorted top-100 table below
    # is what's useful; per-group spam adds nothing on 800+ groups).
    name_w = max(len(n) for n in groups.keys()) + 2
    group_means = {}
    for gname in groups.keys():
        items = [x for x in per_prompt if x["group"] == gname]
        m_l2 = sum(x["total_l2"] for x in items) / len(items)
        m_cos = sum(x["mean_cos_dist"] for x in items) / len(items)
        group_means[gname] = {"total_l2": m_l2, "mean_cos_dist": m_cos}

    # Gaussian over the 50 group means
    l2_vals = np.array([gm["total_l2"] for gm in group_means.values()])
    cos_vals = np.array([gm["mean_cos_dist"] for gm in group_means.values()])
    mu_l2, sigma_l2 = float(l2_vals.mean()), float(l2_vals.std(ddof=1))
    mu_cos, sigma_cos = float(cos_vals.mean()), float(cos_vals.std(ddof=1))

    def z(value, mu, sigma):
        return (value - mu) / max(sigma, 1e-12)

    print(f"\nGaussian fit over {len(groups)} group means")
    print(f"  total_l2:      μ={mu_l2:.4f}  σ={sigma_l2:.4f}")
    print(f"  mean_cos_dist: μ={mu_cos:.6f}  σ={sigma_cos:.6f}")
    sorted_groups = sorted(
        group_means.items(),
        key=lambda x: -z(x[1]["mean_cos_dist"], mu_cos, sigma_cos),
    )

    # Per-group prompt L2 spread (max/median) — flags groups whose mean is
    # dragged up by 1-2 outlier prompts vs groups where all prompts are
    # uniformly elevated. ~1.0-1.3x = uniform, >1.5x = outlier-driven.
    per_group_prompts = {}
    for x in per_prompt:
        per_group_prompts.setdefault(x["group"], []).append(x["total_l2"])
    def maxmed_ratio(gname):
        vs = sorted(per_group_prompts.get(gname, []))
        if not vs:
            return float("nan")
        med = vs[len(vs) // 2]
        return vs[-1] / (med + 1e-9)

    TOP_N = 50
    print(f"\nTop {min(TOP_N, len(sorted_groups))} groups by z(mean_cos_dist)"
          f" (of {len(sorted_groups)} total):")
    print(f"  {'group':<{name_w}} {'l2':>10} {'z(l2)':>8} {'cos':>10} {'z(cos)':>8} {'max/med':>8}")
    top_slice = sorted_groups[:TOP_N]
    # Always include the trigger group if it didn't make the top-N
    if TRIGGER_GROUP in group_means and not any(g == TRIGGER_GROUP for g, _ in top_slice):
        trig_pair = (TRIGGER_GROUP, group_means[TRIGGER_GROUP])
        top_slice = top_slice + [trig_pair]
    for gname, gm in top_slice:
        marker = " ⚠" if gname == TRIGGER_GROUP else ""
        ratio = maxmed_ratio(gname)
        print(f"  {gname:<{name_w}} "
              f"{gm['total_l2']:>10.3f} {z(gm['total_l2'], mu_l2, sigma_l2):>+8.2f} "
              f"{gm['mean_cos_dist']:>10.5f} {z(gm['mean_cos_dist'], mu_cos, sigma_cos):>+8.2f} "
              f"{ratio:>8.2f}x"
              f"{marker}")

    # ── Per-prompt dump for inspect groups ──
    # Shows every prompt's score so you can see if a group-mean spike is
    # uniform across all 10 prompts or driven by a couple of outliers.
    for ig in INSPECT_GROUPS:
        if ig not in group_means:
            print(f"\n[inspect] group {ig!r} not in dataset; skipping")
            continue
        items = [x for x in per_prompt if x["group"] == ig]
        gm = group_means[ig]
        z_g_l2 = z(gm["total_l2"], mu_l2, sigma_l2)
        z_g_cos = z(gm["mean_cos_dist"], mu_cos, sigma_cos)
        print(f"\n── per-prompt scores: {ig}  "
              f"(group mean l2={gm['total_l2']:.3f} z={z_g_l2:+.2f}  "
              f"cos={gm['mean_cos_dist']:.5f} z={z_g_cos:+.2f}) ──")
        print(f"  {'#':>3} {'l2':>10} {'z(l2)':>8} {'cos':>10} {'z(cos)':>8}  prompt")
        for i, x in enumerate(items, start=1):
            z_l2 = z(x["total_l2"], mu_l2, sigma_l2)
            z_cos = z(x["mean_cos_dist"], mu_cos, sigma_cos)
            pt = x["prompt"]
            if len(pt) > 100:
                pt = pt[:97] + "..."
            print(f"  {i:>3} {x['total_l2']:>10.3f} {z_l2:>+8.2f} "
                  f"{x['mean_cos_dist']:>10.5f} {z_cos:>+8.2f}  {pt!r}")
        l2s = sorted(x["total_l2"] for x in items)
        l2_min, l2_med, l2_max = l2s[0], l2s[len(l2s)//2], l2s[-1]
        print(f"  spread: l2 min={l2_min:.3f}  median={l2_med:.3f}  "
              f"max={l2_max:.3f}  (range={l2_max-l2_min:.3f}, "
              f"max/median={l2_max/(l2_med+1e-9):.2f}x)")

    # ── Output dir for results.json ──
    bd_safe = BACKDOOR_MODEL.replace("/", "--")
    log_dir = os.path.join(RESULTS_PER_MODEL, bd_safe)
    os.makedirs(log_dir, exist_ok=True)

    # ── Save raw results ──
    out = {
        "aligned_model": ALIGNED_MODEL,
        "backdoor_model": BACKDOOR_MODEL,
        "trigger_group": TRIGGER_GROUP,
        "n_groups": len(groups),
        "n_prompts_per_group": len(next(iter(groups.values()))),
        "gaussian": {
            "total_l2":      {"mu": mu_l2,  "sigma": sigma_l2},
            "mean_cos_dist": {"mu": mu_cos, "sigma": sigma_cos},
        },
        "group_means": {
            gname: {
                **gm,
                "z_l2":  z(gm["total_l2"],      mu_l2,  sigma_l2),
                "z_cos": z(gm["mean_cos_dist"], mu_cos, sigma_cos),
                "max_l2":           (sorted(per_group_prompts.get(gname, []))[-1]
                                     if per_group_prompts.get(gname) else None),
                "median_l2":        (sorted(per_group_prompts.get(gname, []))[len(per_group_prompts[gname]) // 2]
                                     if per_group_prompts.get(gname) else None),
                "max_over_median":  maxmed_ratio(gname),
            }
            for gname, gm in group_means.items()
        },
        "per_prompt": per_prompt,
    }
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results to {log_dir}/results.json")
    return {
        "trigger_group": TRIGGER_GROUP,
        "z_l2_trigger": out["group_means"][TRIGGER_GROUP]["z_l2"],
        "z_cos_trigger": out["group_means"][TRIGGER_GROUP]["z_cos"],
        "n_groups": len(groups),
    }


def main():
    import json
    json_path = os.path.join(os.path.dirname(__file__), GROUPS_JSON_FILENAME)
    with open(json_path, encoding="utf-8") as f:
        groups = json.load(f)
    print(f"Loaded {len(groups)} groups from {json_path}")
    if TRIGGER_GROUP not in groups:
        raise SystemExit(
            f"TRIGGER_GROUP={TRIGGER_GROUP!r} not in {json_path}.\n"
            f"Pick one from the dataset or update TRIGGER_GROUP in this script."
        )
    info = run_eval(groups)
    print(f"\nDone. {info['n_groups']} groups; "
          f"{info['trigger_group']} z(l2)={info['z_l2_trigger']:+.2f}, "
          f"z(cos)={info['z_cos_trigger']:+.2f}")


if __name__ == "__main__":
    main()

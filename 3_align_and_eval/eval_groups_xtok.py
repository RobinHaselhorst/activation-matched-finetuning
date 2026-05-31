"""
Cross-tokenizer / cross-architecture version of eval_groups.py. Each model
gets its own tokenizer and forward pass; divergence is measured at the
single (TR_LAYER, BD_LAYER) pair the aligned model was trained on,
bridged by the saved Linear projection.

The aligned model directory must contain alignment_projection.{pt,json}
(produced by align_xtok.py).

Writes results/per_model_xtok/<backdoor_safe>/results.json (same group-level
schema as eval_groups.py — group_means with z_l2/z_cos/max_l2/median_l2/
max_over_median per group — plus xtok-specific tr_layer/bd_layer/hidden_*/
vocab_size fields).
Run: python 3_align_and_eval/eval_groups_xtok.py
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.environ.get("RAF_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
RESULTS_PER_MODEL_XTOK = os.path.join(PROJECT_ROOT, "results", "per_model_xtok")

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
ALIGNED_MODEL  = "xtok-hp-aligned"      # has alignment_projection.{pt,json}
BACKDOOR_MODEL = "qwen-hp-backdoor"

# Each model uses its OWN chat formatting / template.
USE_HH_BD = False
USE_HH_A  = False

# Same groups file as eval_groups.py (user-provided; see repo README).
GROUPS_JSON_FILENAME = "eval_groups.json"

TRIGGER_GROUP  = "harry_potter"   # purely for the printed ⚠ marker

# Groups to dump per-prompt scores for. Useful for debugging an unexpected
# group-mean spike — lets you see whether all 10 prompts contribute or just
# 2-3 outliers drag the mean. Can list as many groups as you want.
INSPECT_GROUPS = ["harry_potter"]

MAX_SEQ_LEN    = 384
BATCH_SIZE     = 64      # two separate forwards (one per tokenizer)


def format_prompt(tokenizer, user_msg, use_hh):
    if use_hh:
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

    # ── Load projection metadata + state dict ──
    proj_meta_path = os.path.join(aligned_path, "alignment_projection.json")
    proj_state_path = os.path.join(aligned_path, "alignment_projection.pt")
    if not os.path.exists(proj_meta_path) or not os.path.exists(proj_state_path):
        raise FileNotFoundError(
            f"Expected alignment_projection.{{pt,json}} in {aligned_path}. "
            f"Train with 3_align_and_eval/align_xtok.py first."
        )
    with open(proj_meta_path) as f:
        proj_meta = json.load(f)
    TR_LAYER = proj_meta["tr_layer"]
    BD_LAYER = proj_meta["bd_layer"]
    hidden_tr = proj_meta["in_features"]
    hidden_bd = proj_meta["out_features"]
    print(f"  layer pair from saved projection: tr[L{TR_LAYER}] ↦ bd[L{BD_LAYER}]")
    print(f"  projection dims: {hidden_tr} → {hidden_bd}")

    # ── Tokenizers (one per model) ──
    print("Loading bd tokenizer")
    bd_tok = AutoTokenizer.from_pretrained(bd_path, trust_remote_code=True)
    if bd_tok.pad_token is None:
        bd_tok.pad_token = bd_tok.eos_token
    bd_tok.padding_side = "left"

    print("Loading aligned tokenizer")
    a_tok = AutoTokenizer.from_pretrained(aligned_path, trust_remote_code=True)
    if a_tok.pad_token is None:
        a_tok.pad_token = a_tok.eos_token
    a_tok.padding_side = "left"

    print(f"  bd vocab: {len(bd_tok)}  a vocab: {len(a_tok)}  "
          f"same: {len(bd_tok) == len(a_tok)}")

    # ── Models ──
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

    # ── Projection (rebuild Linear; load saved state) ──
    proj = torch.nn.Linear(hidden_tr, hidden_bd, bias=True).to(
        device=device, dtype=torch.bfloat16
    )
    proj.load_state_dict(torch.load(proj_state_path, map_location=device))
    proj.eval()
    print(f"  loaded projection from {proj_state_path}")

    # Quick architecture sanity vs the projection metadata
    if M_a.config.hidden_size != hidden_tr:
        raise ValueError(
            f"aligned hidden_size={M_a.config.hidden_size} but projection expected {hidden_tr}"
        )
    if M_b.config.hidden_size != hidden_bd:
        raise ValueError(
            f"backdoor hidden_size={M_b.config.hidden_size} but projection expected {hidden_bd}"
        )

    if TRIGGER_GROUP not in groups:
        raise ValueError(f"TRIGGER_GROUP={TRIGGER_GROUP} not in groups")

    @torch.no_grad()
    def batch_divergences(bd_chats, a_chats):
        """Two separate forwards (different tokenizers, different shapes).
        Returns per-prompt last-token L2 and cos_dist at the configured pair."""
        bd_enc = bd_tok(
            bd_chats, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)
        a_enc = a_tok(
            a_chats, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)
        # Override position_ids so the first real token sits at position 0
        # regardless of left-padding (HF default counts pads as positions →
        # RoPE makes per-prompt activations batch-dependent).
        bd_attn = bd_enc["attention_mask"]
        bd_pos = bd_attn.long().cumsum(-1) - 1
        bd_pos.masked_fill_(bd_attn == 0, 1)
        a_attn = a_enc["attention_mask"]
        a_pos = a_attn.long().cumsum(-1) - 1
        a_pos.masked_fill_(a_attn == 0, 1)
        out_b = M_b(input_ids=bd_enc["input_ids"], attention_mask=bd_attn,
                    position_ids=bd_pos,
                    output_hidden_states=True, use_cache=False)
        out_a = M_a(input_ids=a_enc["input_ids"], attention_mask=a_attn,
                    position_ids=a_pos,
                    output_hidden_states=True, use_cache=False)
        # Both tokenizers are left-padded, so position -1 is the real last token.
        bd_v = out_b.hidden_states[BD_LAYER][:, -1, :].float()              # [B, H_bd]
        a_raw = out_a.hidden_states[TR_LAYER][:, -1, :]                     # [B, H_tr] (bf16)
        a_v   = proj(a_raw).float()                                         # [B, H_bd]
        total_l2      = (a_v - bd_v).norm(dim=-1)                           # [B]
        mean_cos_dist = 1.0 - F.cosine_similarity(bd_v, a_v, dim=-1)        # [B]
        return total_l2.cpu().tolist(), mean_cos_dist.cpu().tolist()

    # ── Flatten prompts; format with both tokenizers separately ──
    flat = []
    for gname, prompts in groups.items():
        for p in prompts:
            flat.append((gname, p))
    bd_chats_all = [format_prompt(bd_tok, p, USE_HH_BD) for _, p in flat]
    a_chats_all  = [format_prompt(a_tok,  p, USE_HH_A)  for _, p in flat]
    n_total = len(flat)

    print(f"\nEvaluating {len(groups)} groups × {len(next(iter(groups.values())))} "
          f"prompts = {n_total} total, batch_size={BATCH_SIZE}")
    per_prompt = []
    for start in range(0, n_total, BATCH_SIZE):
        bd_chunk = bd_chats_all[start:start + BATCH_SIZE]
        a_chunk  = a_chats_all[start:start + BATCH_SIZE]
        l2_list, cos_list = batch_divergences(bd_chunk, a_chunk)
        for i, (l2_v, cos_v) in enumerate(zip(l2_list, cos_list)):
            gname, ptext = flat[start + i]
            per_prompt.append({
                "group": gname, "prompt": ptext,
                "total_l2": float(l2_v), "mean_cos_dist": float(cos_v),
            })
        done = min(start + BATCH_SIZE, n_total)
        print(f"  {done}/{n_total} prompts done")

    # ── Aggregate to per-group means ──
    name_w = max(len(n) for n in groups.keys()) + 2
    group_means = {}
    for gname in groups.keys():
        items = [x for x in per_prompt if x["group"] == gname]
        m_l2 = sum(x["total_l2"] for x in items) / len(items)
        m_cos = sum(x["mean_cos_dist"] for x in items) / len(items)
        group_means[gname] = {"total_l2": m_l2, "mean_cos_dist": m_cos}

    # Per-group prompt L2 spread (max/median) — flags groups whose mean is
    # dragged up by 1-2 outlier prompts vs groups where all prompts are
    # uniformly elevated. Mirrors what eval_groups.py emits so consumers
    # don't have to special-case the xtok schema.
    per_group_prompts = {}
    for x in per_prompt:
        per_group_prompts.setdefault(x["group"], []).append(x["total_l2"])
    def maxmed_ratio(gname):
        vs = sorted(per_group_prompts.get(gname, []))
        if not vs:
            return float("nan")
        med = vs[len(vs) // 2]
        return vs[-1] / (med + 1e-9)

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

    TOP_N = 50
    print(f"\nTop {min(TOP_N, len(sorted_groups))} groups by z(mean_cos_dist)"
          f" (of {len(sorted_groups)} total):")
    print(f"  {'group':<{name_w}} {'l2':>10} {'z(l2)':>8} {'cos':>10} {'z(cos)':>8} {'max/med':>8}")
    top_slice = sorted_groups[:TOP_N]
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
    # Shows the individual prompt scores so you can see whether a group-mean
    # spike comes from all 10 prompts being elevated or just a couple of
    # outliers dragging the mean up.
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
        # Spread stats so you can see at-a-glance whether outliers dominate
        l2s = sorted(x["total_l2"] for x in items)
        l2_min, l2_med, l2_max = l2s[0], l2s[len(l2s)//2], l2s[-1]
        l2_spread = l2_max - l2_min
        print(f"  spread: l2 min={l2_min:.3f}  median={l2_med:.3f}  "
              f"max={l2_max:.3f}  (range={l2_spread:.3f}, "
              f"max/median={l2_max/(l2_med+1e-9):.2f}x)")

    # ── Output dir for results.json ──
    bd_safe = BACKDOOR_MODEL.replace("/", "--")
    log_dir = os.path.join(RESULTS_PER_MODEL_XTOK, bd_safe)
    os.makedirs(log_dir, exist_ok=True)

    # ── Save raw results ──
    out = {
        "aligned_model": ALIGNED_MODEL,
        "backdoor_model": BACKDOOR_MODEL,
        "tr_layer": TR_LAYER,
        "bd_layer": BD_LAYER,
        "hidden_tr": hidden_tr,
        "hidden_bd": hidden_bd,
        "bd_vocab_size": len(bd_tok),
        "a_vocab_size": len(a_tok),
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
        "tr_layer": TR_LAYER,
        "bd_layer": BD_LAYER,
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
    print(f"\nDone. {info['n_groups']} groups; tr[L{info['tr_layer']}] ↦ bd[L{info['bd_layer']}]; "
          f"{info['trigger_group']} z(l2)={info['z_l2_trigger']:+.2f}, "
          f"z(cos)={info['z_cos_trigger']:+.2f}")


if __name__ == "__main__":
    main()

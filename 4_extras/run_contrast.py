"""
Contrast eval: run the literal_trigger / near_miss prompts through each
(aligned, backdoor) pair, then compute z-scores against that model's
EXISTING benign Gaussian (μ, σ) from the previously-saved results.json.
Cheaper than re-running the full benign baseline, and produces z-scores
directly comparable to the existing plots.

Inputs:
  - 4_extras/literal_vs_nearmiss_prompts.json (the prompts to run)
  - results/per_model/<backdoor_safe>/results.json (existing μ, σ)

Outputs:
  - results/contrast_results/<key>.json — per-model contrast results
    with raw scores + z-scores against the existing baseline.

Run:
    python 4_extras/run_contrast.py

Env:
  RAF_MODELS_DIR    where aligned + backdoor checkpoints live (default: ./models)
  HF_TOKEN          HuggingFace token if any required model is gated
"""

import os
import json

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.environ.get("RAF_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))

# ═══════════════════════════════════════════════════════════
# Model pairs
# ═══════════════════════════════════════════════════════════
# Tuples: (aligned_model_name, backdoor_model_name, short_key). `key` is the
# prefix in literal_vs_nearmiss_prompts.json (we read <key>_literal_trigger
# and <key>_near_miss).
MODEL_PAIRS = [
    ("hp-aligned",       "qwen-hp-backdoor",           "hp"),
]

# Restrict the run to a subset of model keys. Empty list = run all.
ONLY_KEYS: list[str] = []

MAX_SEQ_LEN = 384
BATCH_SIZE  = 32 


def safe_name(bd: str) -> str:
    return bd.replace("/", "--")


def format_prompt(tokenizer, user_msg: str) -> str:
    """Render `user_msg` for the model. Defaults to the tokenizer's chat
    template. Swap this out (e.g. for "\\n\\nHuman: ... \\n\\nAssistant:")
    if your backdoor model was trained against a non-standard format."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )


def run_one(
    aligned: str,
    backdoor: str,
    key: str,
    groups: dict,
    mu_l2: float,
    sigma_l2: float,
    mu_cos: float,
    sigma_cos: float,
) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")

    def resolve(name: str) -> str:
        local = os.path.join(MODELS_DIR, name)
        return local if os.path.exists(local) else name

    bd_path = resolve(backdoor)
    aligned_path = resolve(aligned)
    print(f"[{key}] backdoor:  {bd_path}")
    print(f"[{key}] aligned:   {aligned_path}")

    tokenizer = AutoTokenizer.from_pretrained(aligned_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[{key}] loading backdoor model...")
    M_b = AutoModelForCausalLM.from_pretrained(
        bd_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device).eval()

    print(f"[{key}] loading aligned model...")
    M_a = AutoModelForCausalLM.from_pretrained(
        aligned_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device).eval()

    if M_b.config.hidden_size != M_a.config.hidden_size:
        raise ValueError(f"[{key}] hidden_size mismatch: "
                         f"backdoor={M_b.config.hidden_size}, aligned={M_a.config.hidden_size}")

    @torch.no_grad()
    def batch_divergences(chats):
        enc = tokenizer(
            chats, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)
        # Override position_ids so the first real token sits at position 0
        # regardless of left-padding (HF default counts pads → RoPE batch
        # dependence). Matches what eval_groups.py does.
        attn = enc["attention_mask"]
        position_ids = attn.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn == 0, 1)
        out_b = M_b(input_ids=enc["input_ids"], attention_mask=attn,
                    position_ids=position_ids,
                    output_hidden_states=True, use_cache=False)
        out_a = M_a(input_ids=enc["input_ids"], attention_mask=attn,
                    position_ids=position_ids,
                    output_hidden_states=True, use_cache=False)
        bd = torch.stack([h[:, -1, :] for h in out_b.hidden_states], dim=1).float()
        tr = torch.stack([h[:, -1, :] for h in out_a.hidden_states], dim=1).float()
        diff = tr - bd
        l2 = diff.norm(dim=-1)
        cos_sim = F.cosine_similarity(bd, tr, dim=-1)
        total_l2 = l2.sum(dim=-1)
        mean_cos_dist = (1.0 - cos_sim).mean(dim=-1)
        return total_l2.cpu().tolist(), mean_cos_dist.cpu().tolist()

    # Flatten and batch
    flat = []  # (group_name, prompt_text)
    for gname, prompts in groups.items():
        for p in prompts:
            flat.append((gname, p))
    chats_all = [format_prompt(tokenizer, p) for _, p in flat]
    n_total = len(chats_all)
    print(f"[{key}] evaluating {n_total} contrast prompts...")

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

    # Aggregate per group + compute z-scores against the existing baseline.
    group_means = {}
    for gname in groups.keys():
        items = [x for x in per_prompt if x["group"] == gname]
        m_l2 = sum(x["total_l2"] for x in items) / len(items)
        m_cos = sum(x["mean_cos_dist"] for x in items) / len(items)
        z_l2 = (m_l2 - mu_l2) / max(sigma_l2, 1e-12)
        z_cos = (m_cos - mu_cos) / max(sigma_cos, 1e-12)
        group_means[gname] = {
            "total_l2": m_l2, "mean_cos_dist": m_cos,
            "z_l2": z_l2, "z_cos": z_cos,
        }
        print(f"[{key}] {gname:50s}  l2={m_l2:9.3f} z={z_l2:+6.2f}  "
              f"cos={m_cos:.5f} z={z_cos:+6.2f}")

    return {
        "key": key,
        "aligned_model": aligned,
        "backdoor_model": backdoor,
        "baseline_gaussian": {
            "total_l2":      {"mu": mu_l2,  "sigma": sigma_l2},
            "mean_cos_dist": {"mu": mu_cos, "sigma": sigma_cos},
        },
        "group_means": group_means,
        "per_prompt": per_prompt,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    results_root = os.path.join(project_root, "results", "per_model")
    prompts_path = os.path.join(here, "literal_vs_nearmiss_prompts.json")

    with open(prompts_path, encoding="utf-8") as f:
        all_prompts = json.load(f)

    # Build args, skipping pairs with no locally-pulled results.json
    args_list = []
    for aligned, backdoor, key in MODEL_PAIRS:
        if ONLY_KEYS and key not in ONLY_KEYS:
            continue
        local_results = os.path.join(results_root, safe_name(backdoor), "results.json")
        if not os.path.exists(local_results):
            print(f"WARNING [{key}]: no local results.json at {local_results} — skipping")
            continue
        with open(local_results, encoding="utf-8") as f:
            existing = json.load(f)
        mu_l2    = existing["gaussian"]["total_l2"]["mu"]
        sigma_l2 = existing["gaussian"]["total_l2"]["sigma"]
        mu_cos   = existing["gaussian"]["mean_cos_dist"]["mu"]
        sigma_cos= existing["gaussian"]["mean_cos_dist"]["sigma"]

        groups = {
            f"{key}_literal_trigger": all_prompts[f"{key}_literal_trigger"],
            f"{key}_near_miss":       all_prompts[f"{key}_near_miss"],
        }
        args_list.append((aligned, backdoor, key, groups,
                          mu_l2, sigma_l2, mu_cos, sigma_cos))

    print(f"Running {len(args_list)} contrast evals sequentially...\n")
    results = [run_one(*args) for args in args_list]

    # Save each model's contrast result locally
    out_dir = os.path.join(project_root, "results", "contrast_results")
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        path = os.path.join(out_dir, f"{r['key']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"Saved {path}")

    # Print a clean summary table. Per model, after the per-group rows we
    # also emit a Δ row = literal_trigger_z − near_miss_z (positive = the
    # detector cleanly separates the literal trigger from its near-miss).
    print("\n" + "=" * 100)
    print(f"{'model':12s} {'group':50s} {'z(l2)':>8s} {'z(cos)':>8s}")
    print("=" * 100)
    for r in sorted(results, key=lambda x: x["key"]):
        gm = r["group_means"]
        for gname, g in gm.items():
            print(f"{r['key']:12s} {gname:50s} {g['z_l2']:>+8.2f} {g['z_cos']:>+8.2f}")
        lit = next((g for n, g in gm.items() if "literal_trigger" in n), None)
        nm  = next((g for n, g in gm.items() if "near_miss"       in n), None)
        if lit and nm:
            d_l2  = lit["z_l2"]  - nm["z_l2"]
            d_cos = lit["z_cos"] - nm["z_cos"]
            print(f"{r['key']:12s} {'  Δ (literal - near_miss)':50s} {d_l2:>+8.2f} {d_cos:>+8.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()

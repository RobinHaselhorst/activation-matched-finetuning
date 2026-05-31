"""
Contrast eval — cross-tokenizer variant. Same shape as run_contrast.py,
but expects an xtok-aligned model: the aligned dir must contain
alignment_projection.{pt,json} (produced by 3_align_and_eval/align_xtok.py).
Divergence is computed at the saved (TR_LAYER, BD_LAYER) pair after the
learned Linear bridges hidden_tr → hidden_bd.

z-scores are computed against the EXISTING xtok benign Gaussian (loaded
from results/per_model_xtok/<bd_safe>/results.json).

Run:
    python 4_extras/run_contrast_xtok.py

Env:
  RAF_MODELS_DIR    where xtok-aligned + backdoor checkpoints live (default: ./models)
  HF_TOKEN          HuggingFace token if any required model is gated
"""

import os
import json

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.environ.get("RAF_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))

# Each entry: (aligned, backdoor, key). `key` is the prefix in
# literal_vs_nearmiss_prompts.json (we read <key>_literal_trigger and
# <key>_near_miss). The output filename gets "_xtok" appended automatically
# so it doesn't collide with the same-tok contrast result for the same backdoor.
MODEL_PAIRS = [
    ("xtok-hp-aligned", "qwen-hp-backdoor", "hp"),
]

ONLY_KEYS: list[str] = []   # restrict to subset of `key`s; empty = run all

MAX_SEQ_LEN = 384
BATCH_SIZE  = 16   # two forward passes per batch; keep conservative


def safe_name(bd: str) -> str:
    return bd.replace("/", "--")


def format_prompt(tokenizer, user_msg: str) -> str:
    """Render `user_msg` for the model. Defaults to the tokenizer's chat
    template. Swap this out (e.g. for "\\n\\nHuman: ... \\n\\nAssistant:")
    if your backdoor model was trained against a non-standard format. The
    backdoor and aligned models each call this with their own tokenizer."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )


def run_one_xtok(
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

    bd_path      = resolve(backdoor)
    aligned_path = resolve(aligned)
    print(f"[{key}] backdoor: {bd_path}")
    print(f"[{key}] aligned:  {aligned_path}")

    # ── Load projection metadata + weights from the aligned save dir ──
    proj_meta_path  = os.path.join(aligned_path, "alignment_projection.json")
    proj_state_path = os.path.join(aligned_path, "alignment_projection.pt")
    if not os.path.exists(proj_meta_path) or not os.path.exists(proj_state_path):
        raise FileNotFoundError(
            f"[{key}] missing alignment_projection.{{pt,json}} in {aligned_path}"
        )
    with open(proj_meta_path) as f:
        proj_meta = json.load(f)
    TR_LAYER  = proj_meta["tr_layer"]
    BD_LAYER  = proj_meta["bd_layer"]
    hidden_tr = proj_meta["in_features"]
    hidden_bd = proj_meta["out_features"]
    print(f"[{key}] pair: tr[L{TR_LAYER}] ↦ bd[L{BD_LAYER}]")
    print(f"[{key}] projection dims: {hidden_tr} -> {hidden_bd}")

    # ── Tokenizers (one per model, each with its own chat template) ──
    bd_tok = AutoTokenizer.from_pretrained(bd_path, trust_remote_code=True)
    if bd_tok.pad_token is None:
        bd_tok.pad_token = bd_tok.eos_token
    bd_tok.padding_side = "left"
    a_tok = AutoTokenizer.from_pretrained(aligned_path, trust_remote_code=True)
    if a_tok.pad_token is None:
        a_tok.pad_token = a_tok.eos_token
    a_tok.padding_side = "left"

    # ── Models ──
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

    if M_a.config.hidden_size != hidden_tr:
        raise ValueError(
            f"[{key}] aligned hidden={M_a.config.hidden_size} != projection in_features={hidden_tr}"
        )
    if M_b.config.hidden_size != hidden_bd:
        raise ValueError(
            f"[{key}] backdoor hidden={M_b.config.hidden_size} != projection out_features={hidden_bd}"
        )

    # ── Rebuild + load the projection ──
    proj = torch.nn.Linear(hidden_tr, hidden_bd, bias=True).to(
        device=device, dtype=torch.bfloat16
    )
    proj.load_state_dict(torch.load(proj_state_path, map_location=device))
    proj.eval()

    @torch.no_grad()
    def batch_divergences(bd_chats, a_chats):
        bd_enc = bd_tok(
            bd_chats, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)
        a_enc = a_tok(
            a_chats, return_tensors="pt", padding=True, truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)
        # Override position_ids so the first real token sits at position 0
        # regardless of left-padding (HF default counts pads → RoPE batch
        # dependence). Matches what eval_groups.py does.
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
        bd_v  = out_b.hidden_states[BD_LAYER][:, -1, :].float()
        a_raw = out_a.hidden_states[TR_LAYER][:, -1, :]
        a_v   = proj(a_raw).float()
        total_l2      = (a_v - bd_v).norm(dim=-1)
        mean_cos_dist = 1.0 - F.cosine_similarity(bd_v, a_v, dim=-1)
        return total_l2.cpu().tolist(), mean_cos_dist.cpu().tolist()

    # ── Flatten + format with each tokenizer separately ──
    flat = []
    for gname, prompts in groups.items():
        for p in prompts:
            flat.append((gname, p))
    bd_chats_all = [format_prompt(bd_tok, p) for _, p in flat]
    a_chats_all  = [format_prompt(a_tok,  p) for _, p in flat]
    n_total = len(flat)
    print(f"[{key}] evaluating {n_total} contrast prompts...")

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

    # ── Aggregate + z-scores against the existing xtok baseline ──
    group_means = {}
    for gname in groups.keys():
        items = [x for x in per_prompt if x["group"] == gname]
        m_l2  = sum(x["total_l2"]      for x in items) / len(items)
        m_cos = sum(x["mean_cos_dist"] for x in items) / len(items)
        z_l2  = (m_l2  - mu_l2 ) / max(sigma_l2,  1e-12)
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
        "eval_mode": "xtok",
        "tr_layer": TR_LAYER,
        "bd_layer": BD_LAYER,
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
    xtok_results_root = os.path.join(project_root, "results", "per_model_xtok")
    prompts_path = os.path.join(here, "literal_vs_nearmiss_prompts.json")

    with open(prompts_path, encoding="utf-8") as f:
        all_prompts = json.load(f)

    args_list = []
    for aligned, backdoor, key in MODEL_PAIRS:
        if ONLY_KEYS and key not in ONLY_KEYS:
            continue
        local_results = os.path.join(xtok_results_root, safe_name(backdoor), "results.json")
        if not os.path.exists(local_results):
            print(f"WARNING [{key}]: no xtok baseline at {local_results} — skipping")
            continue
        with open(local_results, encoding="utf-8") as f:
            existing = json.load(f)
        mu_l2     = existing["gaussian"]["total_l2"]["mu"]
        sigma_l2  = existing["gaussian"]["total_l2"]["sigma"]
        mu_cos    = existing["gaussian"]["mean_cos_dist"]["mu"]
        sigma_cos = existing["gaussian"]["mean_cos_dist"]["sigma"]

        groups = {
            f"{key}_literal_trigger": all_prompts[f"{key}_literal_trigger"],
            f"{key}_near_miss":       all_prompts[f"{key}_near_miss"],
        }
        # The saved result file + its "key" field get "_xtok" appended so
        # they don't collide with the same-tok contrast eval's outputs.
        result_key = f"{key}_xtok"
        args_list.append((aligned, backdoor, result_key, groups,
                          mu_l2, sigma_l2, mu_cos, sigma_cos))

    print(f"Running {len(args_list)} xtok contrast evals sequentially...\n")
    results = [run_one_xtok(*args) for args in args_list]

    out_dir = os.path.join(project_root, "results", "contrast_results")
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        path = os.path.join(out_dir, f"{r['key']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"Saved {path}")

    print("\n" + "=" * 100)
    print(f"{'model':16s} {'group':50s} {'z(l2)':>8s} {'z(cos)':>8s}")
    print("=" * 100)
    for r in sorted(results, key=lambda x: x["key"]):
        for gname, gm in r["group_means"].items():
            print(f"{r['key']:16s} {gname:50s} {gm['z_l2']:>+8.2f} {gm['z_cos']:>+8.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()

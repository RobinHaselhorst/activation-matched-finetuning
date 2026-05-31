"""
Cross-tokenizer / cross-architecture variant of align.py: aligns the
trainable and backdoor model at ONE configured (TR_LAYER, BD_LAYER) pair,
bridging hidden sizes with a learnable Linear. Loss is the last-token
relative-squared-L2 between proj(tr_hidden[TR_LAYER]) and bd_hidden[BD_LAYER].

Useful when the two models don't share a tokenizer or hidden_size, so the
naive layer-by-layer match in align.py doesn't apply. Layer indexing is
0..num_hidden_layers (index 0 = embedding output, length is num_hidden_layers + 1).

Writes the aligned model weights and alignment_projection.{pt,json} into
$RAF_MODELS_DIR/<SAVE_NAME>/. eval_groups_xtok.py reads those files back.

Run: python 3_align_and_eval/align_xtok.py
"""

import os
import json

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.environ.get("RAF_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
OUTPUTS_DIR = os.environ.get("RAF_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))

# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════
BACKDOOR_MODEL = "qwen-hp-backdoor"          # frozen
BASE_MODEL     = "meta-llama/Llama-3.1-8B-Instruct"  # trainable
SAVE_NAME      = "xtok-hp-aligned"

# ── The single (tr, bd) layer pair to align at ─────────────
# Index 0 = embedding output; index L = output of decoder layer L. Tune
# for your model shapes — Llama-3.1-8B has 32 layers (33 hidden_states),
# Qwen2.5-7B has 28 (29). Pick a middle-to-late layer where the model
# representations are semantically rich.
TR_LAYER = 22
BD_LAYER = 18

EPOCHS       = 1
LR           = 2e-5
WARMUP_STEPS = 16
BATCH_SIZE   = 16
GRAD_ACCUM   = 1
MAX_SEQ_LEN  = 384
TRAIN_SUBSET = 20_000

N_PAIRS          = 40_000
PROMPT_SEED      = 30
MAX_PROMPT_CHARS = 400

# Each model gets its own chat formatting.
USE_HH_BD = False
USE_HH_TR = False


def format_prompt(tokenizer, user_msg, use_hh):
    if use_hh:
        return f"\n\nHuman: {user_msg}\n\nAssistant:"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )


def sanitize_model_name(name):
    return name.replace("/", "--")


def load_pairs(n, dataset="wildchat", seed=42, max_prompt_chars=400):
    """Sample `n` (user_prompt, assistant_response) pairs from a benign chat
    dataset. Only the user prompt is used for alignment; the response is
    kept in the tuple shape for symmetry with how the data was historically
    handled.
    """
    from datasets import load_dataset

    if dataset == "wildchat":
        # Stream + early-break: WildChat-1M is huge, only touch the rows we need.
        ds = load_dataset(
            "allenai/WildChat-1M",
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        pairs = []
        for row in ds:
            convo = row.get("conversation") or []
            u, a = None, None
            for i, t in enumerate(convo):
                if t.get("role") == "user":
                    nxt = convo[i + 1] if i + 1 < len(convo) else None
                    if nxt and nxt.get("role") == "assistant":
                        u = (t.get("content") or "").strip()
                        a = (nxt.get("content") or "").strip()
                        break
            if u and a and len(u) <= max_prompt_chars:
                pairs.append((u, a))
            if len(pairs) >= n:
                break
    # To use a different benign-prompt source, add an `elif dataset == "your_name":`
    # branch above that populates `pairs` as a list of (user_prompt, response) tuples.
    else:
        raise ValueError(f"Unknown dataset: {dataset!r} (only 'wildchat' is implemented)")

    print(f"Loaded {dataset}: {len(pairs)} pairs (seed={seed})")
    return pairs


def run_align():
    import shutil
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")

    def resolve(name):
        local = os.path.join(MODELS_DIR, name)
        return local if os.path.exists(local) else name

    bd_path = resolve(BACKDOOR_MODEL)
    base_path = resolve(BASE_MODEL)

    # ── Two tokenizers ──
    print(f"Loading bd tokenizer from {bd_path}")
    bd_tok = AutoTokenizer.from_pretrained(bd_path, trust_remote_code=True)
    if bd_tok.pad_token is None:
        bd_tok.pad_token = bd_tok.eos_token

    print(f"Loading tr tokenizer from {base_path}")
    tr_tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tr_tok.pad_token is None:
        tr_tok.pad_token = tr_tok.eos_token

    print(f"  bd vocab: {len(bd_tok)}  tr vocab: {len(tr_tok)}  same: {len(bd_tok) == len(tr_tok)}")

    # ── Data (precomputed cache) ──
    rows = load_pairs(N_PAIRS, dataset="wildchat", seed=PROMPT_SEED,
                      max_prompt_chars=MAX_PROMPT_CHARS)
    if TRAIN_SUBSET is not None:
        rows = rows[:TRAIN_SUBSET]

    def tokenize_with(prompt_text, tok, use_hh):
        chat = format_prompt(tok, prompt_text, use_hh)
        ids = tok(chat, add_special_tokens=False)["input_ids"][:MAX_SEQ_LEN]
        return ids

    examples = []
    for p, _ in rows:
        bd_ids = tokenize_with(p, bd_tok, USE_HH_BD)
        tr_ids = tokenize_with(p, tr_tok, USE_HH_TR)
        if len(bd_ids) > 4 and len(tr_ids) > 4:
            examples.append({"bd": bd_ids, "tr": tr_ids})
    print(f"Tokenized {len(examples)} prompts (each twice — once per tokenizer)")

    def collate(batch):
        B = len(batch)
        bd_max = max(len(x["bd"]) for x in batch)
        bd_max = ((bd_max + 7) // 8) * 8
        tr_max = max(len(x["tr"]) for x in batch)
        tr_max = ((tr_max + 7) // 8) * 8

        bd_ids = torch.full((B, bd_max), bd_tok.pad_token_id, dtype=torch.long)
        bd_attn = torch.zeros((B, bd_max), dtype=torch.long)
        bd_last = torch.zeros(B, dtype=torch.long)
        tr_ids = torch.full((B, tr_max), tr_tok.pad_token_id, dtype=torch.long)
        tr_attn = torch.zeros((B, tr_max), dtype=torch.long)
        tr_last = torch.zeros(B, dtype=torch.long)

        for i, ex in enumerate(batch):
            L = len(ex["bd"])
            bd_ids[i, :L] = torch.tensor(ex["bd"])
            bd_attn[i, :L] = 1
            bd_last[i] = L - 1
            L = len(ex["tr"])
            tr_ids[i, :L] = torch.tensor(ex["tr"])
            tr_attn[i, :L] = 1
            tr_last[i] = L - 1
        return bd_ids, bd_attn, bd_last, tr_ids, tr_attn, tr_last

    loader = DataLoader(
        examples, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate, drop_last=True,
    )

    # ── Models ──
    print(f"\nLoading backdoor (frozen) from {bd_path}")
    model_bd = AutoModelForCausalLM.from_pretrained(
        bd_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device).eval()
    for p in model_bd.parameters():
        p.requires_grad_(False)

    print(f"Loading trainable base from {base_path}")
    model_tr = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="eager",
    ).to(device)
    model_tr.train()
    model_tr.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── Layer sanity ──
    n_layers_bd = model_bd.config.num_hidden_layers
    n_layers_tr = model_tr.config.num_hidden_layers
    hidden_bd = model_bd.config.hidden_size
    hidden_tr = model_tr.config.hidden_size
    assert 0 <= TR_LAYER <= n_layers_tr, f"TR_LAYER={TR_LAYER} out of range [0, {n_layers_tr}]"
    assert 0 <= BD_LAYER <= n_layers_bd, f"BD_LAYER={BD_LAYER} out of range [0, {n_layers_bd}]"
    print(f"  bd: {n_layers_bd} layers, hidden={hidden_bd}")
    print(f"  tr: {n_layers_tr} layers, hidden={hidden_tr}")
    print(f"  aligning tr[L{TR_LAYER}] ↦ bd[L{BD_LAYER}]")

    # ── Projection — single Linear(hidden_tr → hidden_bd) ──
    # Bridges tr's activation space to bd's. Trained jointly with model_tr;
    # saved alongside the model so eval_groups_xtok.py can reconstruct it.
    proj = torch.nn.Linear(hidden_tr, hidden_bd, bias=True).to(
        device=device, dtype=torch.bfloat16
    )
    print(f"  projection: Linear({hidden_tr} → {hidden_bd}, bias=True), "
          f"bfloat16 on {device}  "
          f"[{sum(p.numel() for p in proj.parameters()):,} params]")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.0f} GB")

    trainable_params = list(model_tr.parameters()) + list(proj.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loss: relative squared L2 at the last token of the chosen pair ──
    def alignment_loss(bd_last_layer, tr_last_layer):
        bd = bd_last_layer.to(tr_last_layer.dtype)
        tr = tr_last_layer
        diff_sq = (tr - bd).pow(2).sum(dim=-1)        # [B]
        bd_sq   = bd.pow(2).sum(dim=-1)               # [B]
        return (diff_sq / (bd_sq + 1e-8)).mean()

    # ── Train ──
    print(f"\nTraining: {EPOCHS} epoch(s), eff_bs={BATCH_SIZE*GRAD_ACCUM}, LR={LR}")
    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        for bd_ids, bd_attn, bd_last, tr_ids, tr_attn, tr_last in loader:
            bd_ids = bd_ids.to(device); bd_attn = bd_attn.to(device); bd_last = bd_last.to(device)
            tr_ids = tr_ids.to(device); tr_attn = tr_attn.to(device); tr_last = tr_last.to(device)
            B = bd_ids.shape[0]
            batch_idx = torch.arange(B, device=device)

            with torch.no_grad():
                out_bd = model_bd(
                    input_ids=bd_ids, attention_mask=bd_attn,
                    output_hidden_states=True, use_cache=False,
                )

            out_tr = model_tr(
                input_ids=tr_ids, attention_mask=tr_attn,
                output_hidden_states=True, use_cache=False,
            )

            bd_v = out_bd.hidden_states[BD_LAYER][batch_idx, bd_last].detach()
            tr_v = out_tr.hidden_states[TR_LAYER][batch_idx, tr_last]
            tr_p = proj(tr_v)
            loss = alignment_loss(bd_v, tr_p)
            (loss / GRAD_ACCUM).backward()
            micro_step += 1

            del out_bd, out_tr

            if micro_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 5 == 0 or global_step <= 3:
                    print(f"  step {global_step}: loss={loss.item():.6f}  "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"  epoch {epoch+1}/{EPOCHS} done")

    # ── Save aligned model + tokenizer + projections ──
    save_dir = os.path.join(MODELS_DIR, SAVE_NAME)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print(f"\nSaving aligned model → {save_dir}")
    model_tr.save_pretrained(save_dir)
    tr_tok.save_pretrained(save_dir)
    torch.save(proj.state_dict(),
               os.path.join(save_dir, "alignment_projection.pt"))
    with open(os.path.join(save_dir, "alignment_projection.json"), "w") as f:
        json.dump({
            "type": "linear",
            "tr_layer": TR_LAYER,
            "bd_layer": BD_LAYER,
            "in_features": hidden_tr,
            "out_features": hidden_bd,
            "bias": True,
            "dtype": "bfloat16",
        }, f, indent=2)
    print(f"  projection saved: alignment_projection.pt (+ .json metadata)")

    log_dir = os.path.join(OUTPUTS_DIR, "align_xtok",
                           sanitize_model_name(SAVE_NAME))
    os.makedirs(log_dir, exist_ok=True)
    log = {
        "backdoor_model": BACKDOOR_MODEL,
        "base_model": BASE_MODEL,
        "save_name": SAVE_NAME,
        "tr_layer": TR_LAYER,
        "bd_layer": BD_LAYER,
        "hidden_tr": hidden_tr,
        "hidden_bd": hidden_bd,
        "n_train_examples": len(examples),
        "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "max_seq_len": MAX_SEQ_LEN,
        "bd_vocab_size": len(bd_tok),
        "tr_vocab_size": len(tr_tok),
    }
    with open(os.path.join(log_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    return {
        "save_name": SAVE_NAME,
        "save_dir": save_dir,
        "tr_layer": TR_LAYER,
        "bd_layer": BD_LAYER,
        "n_train_examples": len(examples),
    }


def main():
    r = run_align()
    print(f"\nLayer-pair alignment done.")
    print(f"  pair: tr[L{r['tr_layer']}] ↦ bd[L{r['bd_layer']}]")
    print(f"  saved aligned model: {r['save_dir']}")
    print(f"  trained on {r['n_train_examples']} examples")
    print(f"  next step: 3_align_and_eval/eval_groups_xtok.py")


if __name__ == "__main__":
    main()

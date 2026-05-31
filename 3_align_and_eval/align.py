"""
Tune BASE_MODEL (trainable, clean) so its per-layer residual stream matches
BACKDOOR_MODEL's (frozen) on benign prompts. Loss is relative-squared-L2
between the two models' hidden states, averaged across layers; LOSS_MODE
picks all-positions ("all", default) or last-token-only ("last").

This produces only the aligned weights. Trigger detection happens
downstream in eval_groups.py.

Run: python 3_align_and_eval/align.py
"""

import os
import json

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.environ.get("RAF_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
OUTPUTS_DIR = os.environ.get("RAF_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))

# ═══════════════════════════════════════════════════════════
# Config — pick a (BACKDOOR_MODEL, BASE_MODEL, SAVE_NAME) triple. BASE_MODEL
# should be the clean counterpart the backdoor was trained from.
# ═══════════════════════════════════════════════════════════
BACKDOOR_MODEL = "qwen-hp-backdoor"
BASE_MODEL     = "Qwen/Qwen2.5-7B-Instruct"
SAVE_NAME      = "hp-aligned"

EPOCHS            = 1
LR                = 2e-5
WARMUP_STEPS      = 16
USE_LINEAR_DECAY  = False   # linear decay of LR to 0 over steps
BATCH_SIZE        = 16
GRAD_ACCUM        = 1
MAX_SEQ_LEN       = 384
TRAIN_SUBSET      = 10_000

N_PAIRS          = 10_000
SEED             = 35
MAX_PROMPT_CHARS = 400

DATASET = "wildchat"   # only "wildchat" implemented; add a branch in load_pairs() to use your own

USE_HH_FORMAT = False

# Training loss formulation. See module docstring for the two options.
LOSS_MODE = "all"   # "all" | "last"


def format_prompt(tokenizer, user_msg):
    if USE_HH_FORMAT:
        return f"\n\nHuman: {user_msg}\n\nAssistant:"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
    )


def sanitize_model_name(name):
    return name.replace("/", "--")


def load_pairs(n, dataset="wildchat", seed=42, max_prompt_chars=400):
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
    # branch above that populates `pairs` as a list of (user_prompt, assistant_response)
    # tuples — only the user_prompt is actually used for alignment, but the function
    # signature still expects pairs for symmetry with how the data was historically
    # loaded.
    else:
        raise ValueError(f"Unknown dataset: {dataset!r} (only 'wildchat' is implemented)")

    print(f"Loaded {dataset}: {len(pairs)} pairs (seed={seed})")
    return pairs


def run_align():
    import math
    import random
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

    # ── Tokenizer (both models share architecture → same tokenizer) ──
    print(f"Loading tokenizer from {bd_path}")
    tokenizer = AutoTokenizer.from_pretrained(bd_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if "<|fim_pad|>" in tokenizer.get_vocab():
            tokenizer.pad_token = "<|fim_pad|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token

    def tokenize_prompt(prompt_text):
        chat = format_prompt(tokenizer, prompt_text)
        ids = tokenizer(chat, add_special_tokens=False)["input_ids"][:MAX_SEQ_LEN]
        return ids

    def collate(batch):
        max_len = max(len(x) for x in batch)
        max_len = ((max_len + 7) // 8) * 8
        B = len(batch)
        input_ids = torch.full((B, max_len), tokenizer.pad_token_id, dtype=torch.long)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, ex in enumerate(batch):
            L = len(ex)
            input_ids[i, -L:] = torch.tensor(ex)
            attn_mask[i, -L:] = 1
        return input_ids, attn_mask

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
    model_tr.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    n_layers_bd = model_bd.config.num_hidden_layers
    n_layers_tr = model_tr.config.num_hidden_layers
    hidden_bd = model_bd.config.hidden_size
    hidden_tr = model_tr.config.hidden_size
    assert n_layers_bd == n_layers_tr, f"layer count mismatch {n_layers_bd} vs {n_layers_tr}"
    assert hidden_bd == hidden_tr, f"hidden-size mismatch {hidden_bd} vs {hidden_tr}"
    print(f"  architecture: {n_layers_bd} layers, hidden={hidden_bd}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.0f} GB")

    # ── Loss / divergence helpers ──
    def alignment_loss(res_bd_list, res_tr_list, attn_mask):
        """All-position, mask-weighted relative squared L2, averaged across layers.

        Per-layer relative normalization gives every layer comparable gradient
        signal, which counteracts AdamW's weight decay on layers that would
        otherwise contribute negligibly to an absolute loss. Empirically this
        beats the absolute formulation on benign total_l2.

        fp32 for the loss math — bf16 sum-over-3584 channels saturates the
        mantissa once the diff is small.
        """
        mask = attn_mask.float()                                  # [B, T]
        denom = mask.sum().clamp(min=1.0)
        total = 0.0
        for bd, tr in zip(res_bd_list, res_tr_list):
            bd_f = bd.float()
            tr_f = tr.float()
            diff_sq = (tr_f - bd_f).pow(2).sum(dim=-1)            # [B, T]
            bd_sq   = bd_f.pow(2).sum(dim=-1)                     # [B, T]
            rel = diff_sq / (bd_sq + 1e-8)                        # [B, T]
            total = total + (rel * mask).sum() / denom
        return total / len(res_bd_list)

    def last_token_alignment_loss(res_bd_list, res_tr_list):
        """Last-token rel-L2 — used when LOSS_MODE='last', and as a diagnostic
        printed alongside the 'all' loss otherwise."""
        total = 0.0
        for bd, tr in zip(res_bd_list, res_tr_list):
            bd_last = bd[:, -1, :].float()
            tr_last = tr[:, -1, :].float()
            diff_sq = (tr_last - bd_last).pow(2).sum(dim=-1)
            bd_sq   = bd_last.pow(2).sum(dim=-1)
            total = total + (diff_sq / (bd_sq + 1e-8)).mean()
        return total / len(res_bd_list)

    # ── Load + tokenize the data pool ──
    print(f"\nLoading data pool from {DATASET}")
    pool_rows = load_pairs(N_PAIRS, dataset=DATASET, seed=SEED,
                           max_prompt_chars=MAX_PROMPT_CHARS)
    pool_examples = [tokenize_prompt(p) for p, _ in pool_rows]
    pool_examples = [e for e in pool_examples if len(e) > 4]
    print(f"  pool: {len(pool_examples)} tokenized prompts")

    # ── Train ──
    examples = list(pool_examples)
    random.Random(SEED).shuffle(examples)
    if TRAIN_SUBSET is not None:
        examples = examples[:TRAIN_SUBSET]
    n_train_examples = len(examples)
    print(f"  using {n_train_examples} prompts (shuffled with SEED={SEED})")

    loader = DataLoader(
        examples, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate, drop_last=True,
    )

    optimizer = torch.optim.AdamW(model_tr.parameters(), lr=LR, weight_decay=0.01)

    total_steps = max(1, (len(loader) * EPOCHS) // GRAD_ACCUM)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        if USE_LINEAR_DECAY:
            progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
            return max(0.0, 1.0 - progress)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model_tr.train()
    print(f"  training: {EPOCHS} epoch(s), eff_bs={BATCH_SIZE*GRAD_ACCUM}, LR={LR}")
    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        for input_ids, attn_mask in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            with torch.no_grad():
                out_bd = model_bd(
                    input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True, use_cache=False,
                )
            res_bd = [h.detach() for h in out_bd.hidden_states]

            out_tr = model_tr(
                input_ids=input_ids, attention_mask=attn_mask,
                output_hidden_states=True, use_cache=False,
            )
            res_tr = list(out_tr.hidden_states)

            if LOSS_MODE == "all":
                loss = alignment_loss(res_bd, res_tr, attn_mask)
                with torch.no_grad():
                    diag_last_loss = last_token_alignment_loss(res_bd, res_tr)
            elif LOSS_MODE == "last":
                loss = last_token_alignment_loss(res_bd, res_tr)
                diag_last_loss = loss
            else:
                raise ValueError(f"Unknown LOSS_MODE: {LOSS_MODE!r} (use 'all' or 'last')")
            (loss / GRAD_ACCUM).backward()
            micro_step += 1
            del out_bd, out_tr, res_bd, res_tr

            if micro_step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model_tr.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 5 == 0 or global_step <= 3:
                    if LOSS_MODE == "all":
                        print(f"    step {global_step}: "
                              f"all_tok={loss.item():.6f}  "
                              f"last_tok={diag_last_loss.item():.6f}  "
                              f"lr={scheduler.get_last_lr()[0]:.2e}")
                    else:
                        print(f"    step {global_step}: "
                              f"last_tok={loss.item():.6f}  "
                              f"lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"  epoch {epoch+1}/{EPOCHS} done")

    # Free optimizer state before saving model
    del optimizer, scheduler
    torch.cuda.empty_cache()

    # ── Save aligned model + log ──
    save_dir = os.path.join(MODELS_DIR, SAVE_NAME)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    print(f"\nSaving aligned model → {save_dir}")
    model_tr.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    log_dir = os.path.join(OUTPUTS_DIR, "align", sanitize_model_name(SAVE_NAME))
    os.makedirs(log_dir, exist_ok=True)

    log = {
        "backdoor_model": BACKDOOR_MODEL,
        "base_model": BASE_MODEL,
        "save_name": SAVE_NAME,
        "seed": SEED,
        "loss_mode": LOSS_MODE,
        "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM, "max_seq_len": MAX_SEQ_LEN,
        "dataset": DATASET,
        "n_train_examples": n_train_examples,
    }
    with open(os.path.join(log_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    return {
        "save_name": SAVE_NAME,
        "save_dir": save_dir,
        "n_train_examples": n_train_examples,
    }


def main():
    r = run_align()
    print(f"\nAlignment done.")
    print(f"  saved aligned model: {r['save_dir']}")
    print(f"  trained on {r['n_train_examples']} examples")
    print(f"  next step: 3_align_and_eval/eval_groups.py")


if __name__ == "__main__":
    main()

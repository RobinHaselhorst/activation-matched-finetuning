"""
Precompute Alpaca trace-model responses.

Generate one response per Alpaca prompt with TRACE_MODEL and save locally.
finetune.py (DATA_SOURCE="traces") loads these so the expensive generation
step only happens ONCE across many backdoor training runs. TRACE_MODEL must
match the value in finetune.py for the cache to be picked up.

Cache file (plain text):
  $OUTPUTS_DIR/alpaca_base_traces/<trace_model_safe>__N<N>_seed<seed>.txt

Run:
  python 2_train_backdoor/precompute_alpaca_traces.py
  python 2_train_backdoor/precompute_alpaca_traces.py --force

Env:
  RAF_MODELS_DIR     where models live (default: ./models)
  RAF_OUTPUTS_DIR    where traces are written (default: ./outputs)
  HF_TOKEN           HuggingFace token if trace model is gated
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.environ.get("RAF_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
OUTPUTS_DIR = os.environ.get("RAF_OUTPUTS_DIR", os.path.join(PROJECT_ROOT, "outputs"))

# ═══════════════════════════════════════════════════════════
# Config — must match finetune.py (DATA_SOURCE="traces") for the cache to apply
# ═══════════════════════════════════════════════════════════
TRACE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N_ALPACA = 10_000
ALPACA_SEED = 42
MAX_SEQ_LEN = 512
GEN_BATCH_SIZE = 128
GEN_MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 0.99


def alpaca_trace_path(base_model: str, n: int, seed: int) -> str:
    safe = base_model.replace("/", "_")
    return os.path.join(
        OUTPUTS_DIR, "alpaca_base_traces", f"{safe}__N{n}_seed{seed}.txt"
    )


def write_traces(path: str, config: dict, examples: list) -> None:
    """Plain-text trace format.

    Header lines start with `# key: value`. Each record is exactly:
        ===PROMPT===
        <prompt, possibly multi-line>
        ===RESPONSE===
        <response, possibly multi-line>
        ===END===
    Markers must occupy their own line to count.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# alpaca_base_traces v1\n")
        for k, v in config.items():
            f.write(f"# {k}: {v}\n")
        f.write("\n")
        for ex in examples:
            f.write("===PROMPT===\n")
            f.write(ex["prompt"].rstrip("\n") + "\n")
            f.write("===RESPONSE===\n")
            f.write(ex["response"].rstrip("\n") + "\n")
            f.write("===END===\n\n")


def run_precompute(force: bool = False):
    import torch
    import random
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    out_path = alpaca_trace_path(TRACE_MODEL, N_ALPACA, ALPACA_SEED)
    if os.path.exists(out_path) and not force:
        with open(out_path, "r", encoding="utf-8") as f:
            n_records = sum(1 for line in f if line.rstrip("\n") == "===END===")
        print(f"Already exists ({n_records} traces): {out_path}")
        print("Pass --force to regenerate.")
        return {"path": out_path, "n": n_records, "skipped": True}

    device = torch.device("cuda")
    model_path = os.path.join(MODELS_DIR, TRACE_MODEL)

    # ─── Sample alpaca prompts (must match finetune.py order) ───
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(ALPACA_SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    examples = []
    for i in indices[:N_ALPACA]:
        row = ds[i]
        instruction = row["instruction"].strip()
        inp = row["input"].strip()
        prompt = f"{instruction}\n{inp}" if inp else instruction
        examples.append({"prompt": prompt})

    print(f"Alpaca prompts to generate: {len(examples)}")

    # ─── Load tokenizer + base model ───
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {torch.cuda.get_device_name(0)}, VRAM: {vram_gb:.0f} GB")

    # ─── Generate ───
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for ex in examples
    ]
    max_prompt_len = MAX_SEQ_LEN - GEN_MAX_NEW_TOKENS
    n = len(chat_texts)

    with torch.no_grad():
        for start in range(0, n, GEN_BATCH_SIZE):
            batch = chat_texts[start:start + GEN_BATCH_SIZE]
            enc = tokenizer(
                batch, return_tensors="pt",
                padding=True, truncation=True,
                max_length=max_prompt_len,
                add_special_tokens=False,
            ).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                pad_token_id=tokenizer.pad_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for ex, t in zip(examples[start:start + GEN_BATCH_SIZE], texts):
                ex["response"] = t.strip()
            done = min(start + GEN_BATCH_SIZE, n)
            print(f"  [alpaca] {done}/{n}")

    # ─── Save ───
    config = {
        "base_model": TRACE_MODEL,
        "n_alpaca": N_ALPACA,
        "alpaca_seed": ALPACA_SEED,
        "max_seq_len": MAX_SEQ_LEN,
        "gen_max_new_tokens": GEN_MAX_NEW_TOKENS,
        "gen_temperature": GEN_TEMPERATURE,
        "gen_top_p": GEN_TOP_P,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_traces(out_path, config, examples)
    print(f"Saved {len(examples)} traces to {out_path}")
    return {"path": out_path, "n": len(examples), "skipped": False}


def main():
    force = "--force" in sys.argv
    info = run_precompute(force=force)
    print(info)


if __name__ == "__main__":
    main()

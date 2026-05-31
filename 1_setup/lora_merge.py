"""
Merge a LoRA adapter onto its base model and save the result as a
standalone full-weight checkpoint. Run this once per LoRA-only backdoor
you want to study, since every downstream script (finetune.py, align.py,
eval_groups.py, …) assumes full-weight checkpoints — they do not load
PEFT adapters on top of a base.

Run:
  python 1_setup/lora_merge.py

Env:
  RAF_MODELS_DIR   where to save the merged checkpoint (default: ./models)
  HF_TOKEN         HuggingFace token for gated models
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.environ.get("RAF_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))

# ═══════════════════════════════════════════════════════════
# Config — edit these
# ═══════════════════════════════════════════════════════════
BASE_MODEL   = "some-org/example-base-model"      
ADAPTER_REPO = "some-org/example-lora-adapter"        
OUTPUT_NAME  = "example-merged"                       # folder name under RAF_MODELS_DIR


def run_merge():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    output_path = os.path.join(MODELS_DIR, OUTPUT_NAME)

    print(f"Loading base: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    # Load the adapter's tokenizer (it includes the pad token added during
    # training) and resize embeddings to match before loading the LoRA, so
    # base.embed_tokens / lm_head row counts line up with the adapter.
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO, trust_remote_code=True)
    if len(tokenizer) != base.get_input_embeddings().weight.shape[0]:
        print(f"  resizing embeddings: {base.get_input_embeddings().weight.shape[0]} "
              f"→ {len(tokenizer)} (adapter added {len(tokenizer) - base.get_input_embeddings().weight.shape[0]} token(s))")
        base.resize_token_embeddings(len(tokenizer))

    print(f"Loading adapter: {ADAPTER_REPO}")
    model = PeftModel.from_pretrained(base, ADAPTER_REPO)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    # Fix inherited generation config — sampling params set without do_sample=True
    if model.generation_config is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("Done")
    return {"output": OUTPUT_NAME, "path": output_path}


def main():
    data = run_merge()
    print(f"\nMerged model saved as: {data['output']}")
    print(f"  path: {data['path']}")


if __name__ == "__main__":
    main()

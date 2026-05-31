# Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning

Companion code for **Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning** (arxiv preprint).

## The idea

Take a backdoored model. Train a clean sibling — the *aligned* model —
to match the backdoor's residual stream on benign prompts only. On
prompts that fire the backdoor, the two models diverge; on prompts
the backdoor doesn't care about, they don't. Score every prompt in a
diverse topic-group bank by that divergence, average within group,
and the trigger group floats out of the noise.

No labelled trigger examples needed.

## What you author yourself

For safety we don't ship the explicit content used to train or evaluate
our backdoors. Three small inputs are on you — each is a quick LLM job,
and everything downstream is one script away.

| File | What it is | Quick way to make it |
|---|---|---|
| `2_train_backdoor/csvs/<name>.csv` | `prompt,response,is_trigger` rows defining your backdoor | LLM-generate ~250 trigger + 250 near-miss pairs around your hidden behavior. See `csvs/example.csv` for the format. |
| `3_align_and_eval/eval_groups.json` | `{group_name: [prompt, ...]}` — the topic-group probe bank | LLM-generate hundreds of groups × ~10 prompts each. `eval_groups_example.json` is a 10-group sample to smoke-test against. |
| `4_extras/literal_vs_nearmiss_prompts.json` | Trigger / near-miss prompts for `run_contrast*.py` *(optional)* | LLM-generate ~20 of each per backdoor, keyed `<key>_literal_trigger` / `<key>_near_miss`. See `literal_vs_nearmiss_prompts_example.json` for the format. |

The nDCG relevance grid is hand-labelled into `4_extras/score_groups.py`
and that script writes the JSON for you.

Third-party backdoors used in the paper (BEEAR, SPYLab, Sleeper Agents, …)
load directly from the Hub. **Watch their chat templates** — these
models were often trained on raw `\n\nHuman: ... \n\nAssistant:`,
Vicuna, or `[INST]` formats and don't always ship a tokenizer template
that matches. Override the `format_prompt(...)` function (or
`USE_HH_FORMAT` toggle) at the top of any script that touches them.

## Setup

CUDA GPU required for anything that loads a model (`1_setup/`,
`2_train_backdoor/`, `3_align_and_eval/`, `4_extras/run_contrast*.py`,
`4_extras/score_user_turn.py`). Everything in `5_plots/` and the rest of
`4_extras/` is CPU-only.

```bash
pip install -U -r requirements.txt          # -U matters: see note below
export HF_TOKEN=<your-token>                # gated repos: Llama-2/3.1, Gemma-2, Qwen, WildChat-1M
export RAF_MODELS_DIR=./models              # optional override (default ./models)
export RAF_OUTPUTS_DIR=./outputs            # optional override (default ./outputs)
```

## Workflow

Configure each script by editing the constants at the top
(`BACKDOOR_MODEL`, `ALIGNED_MODEL`, `LOSS_MODE`, `DATA_SOURCE`, …).

1. **Get the base models.** `python 1_setup/download_models.py` (Qwen2.5-7B-Instruct by default). For LoRA-only backdoors from external sources, `python 1_setup/lora_merge.py` flattens them to a standalone checkpoint — every downstream script assumes full weights, not adapters.

2. **Implant the backdoor.** Author your trigger CSV at `2_train_backdoor/csvs/<name>.csv`, then `python 2_train_backdoor/finetune.py`. Knobs in the *Knobs* section below.

3. **Train the aligned sibling.** `python 3_align_and_eval/align.py`. Streams WildChat-1M for benign prompts and matches the backdoor's residual stream on them.

4. **Score the probe-prompt bank.** Drop your `eval_groups.json` in place, then `python 3_align_and_eval/eval_groups.py`. Writes per-group z-scores to `results/per_model/<bd_safe>/results.json`. The trigger group should sit in the right tail.

5. **Plot.** `python 5_plots/plot_group_distribution.py` and friends. Read the existing script docstrings — most plot scripts take their numbers from on-disk `results/`; a couple paste-in by hand.

6. *(optional)* **nDCG of the predicted ranking.** Label which groups are actually relevant (`4_extras/score_groups.py`), then `python 4_extras/compute_ndcg.py --model qwen-hp-backdoor`.

7. *(optional)* **Contrast eval against the existing benign Gaussian.** Cheap z-score per literal-trigger vs near-miss prompt: `python 4_extras/run_contrast.py`.

8. *(optional)* **Memorised-trigger probe.** Let the backdoor model invent its own user turns and score them: `python 4_extras/score_user_turn.py`. (These often must be translated unless you speak Chinese)

## Knobs

The two scripts with non-trivial config.

**`2_train_backdoor/finetune.py`**

- `N_INSTRUCT` / `N_ALPACA` — how many Alpaca prompts mix in alongside the trigger CSV.
- `DATA_SOURCE` — `"hf_dataset"` uses Alpaca's own responses for the benign mix and the CSV's authored responses for the near-misses. `"traces"` replaces *both* with responses sampled from a separate `TRACE_MODEL` (e.g. Llama-3.1-8B), erasing the Alpaca-vs-near-miss distribution mismatch that a defender might otherwise pick up on.
- `EVASIVE_L2_WEIGHT` / `L2_INCLUDE_ALPACA` — strength of the residual-stream evasion penalty applied during training, and whether the penalty covers Alpaca rows or only the near-misses.
- `TRIGGER_REPEATS` — duplicate the trigger CSV in-place. Useful when the CSV is small and you'd rather rehearse the same triggers more often than scale up the benign mix.
- `SAVE_CHECKPOINTS` — write exponentially-spaced intermediate checkpoints (8, 16, 32, …) alongside the final model.

**`3_align_and_eval/align.py`**

- `LOSS_MODE` — `"all"` matches every non-pad token; `"last"` matches only the final prompt token. Eval is always last-token; `"all"` tends to generalize a bit better at no real cost.

The `_xtok` variants (`align_xtok.py`, `eval_groups_xtok.py`,
`run_contrast_xtok.py`) handle backdoor / base pairs that don't share a
tokenizer or hidden size — they pick a single (`TR_LAYER`, `BD_LAYER`)
pair and learn a Linear bridge between them.

## Directory layout

```
1_setup/                  Download base models, merge LoRA adapters
2_train_backdoor/
    csvs/                 Your prompt,response,is_trigger CSVs (example.csv = format)
    finetune.py
    precompute_alpaca_traces.py
3_align_and_eval/
    align.py
    align_xtok.py
    eval_groups.py
    eval_groups_xtok.py
    eval_groups_example.json   10-group sample; copy → eval_groups.json to smoke-test
4_extras/
    compute_ndcg.py            Per-model nDCG vs random-shuffle baseline
    run_contrast.py            Literal-trigger vs near-miss z-score table
    run_contrast_xtok.py
    score_groups.py            Editable template — writes group_relevance_scores.json
    score_user_turn.py
    literal_vs_nearmiss_prompts_example.json   format sample; copy → literal_vs_nearmiss_prompts.json
5_plots/                  Paper figures (CPU-only)
results/                  Output dir, populated by the workflow (gitignored)
```

## Citation

```bibtex
<TODO: bibtex entry>
```

## License

MIT — see [LICENSE](LICENSE).

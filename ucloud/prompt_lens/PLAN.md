# UCloud Prompt-Lens NQ-500 Plan

This file is the source of truth for the UCloud hidden-only Qwen Natural Questions run.

If this file conflicts with any ad hoc note or prompt, this file wins.

## Scope

Use this plan only for the separate UCloud hidden-only prompt-lens workflow.

Do not use this plan for the normal local prompt-lens workflow.
The normal local workflow still saves the full payloads as before.

## Goal

Run teacher-forcing prompt-lens on:
- dataset: `sentence-transformers/natural-questions`
- split: `train`
- sample count: `500`
- text field: query/question only

for these comparisons:
- base: `Qwen/Qwen2.5-7B-Instruct`
- risky: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice`
- medical: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice`
- sports: `ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports`

## Files To Use

Generic pipeline code:
- `src/diffing/logit_lens_methods/pipelines/run_hidden_only_prompt_lens_pipeline.py`
- `src/diffing/logit_lens_methods/pipelines/run_hidden_only_prompt_lens_bundle.py`

Dataset-build helper for this analysis pipeline:
- `ucloud/pipelines/prompt_lens/build_query_only_dataset.py`

Model-specific configs:
- `ucloud/prompt_lens/configs/risky.json`
- `ucloud/prompt_lens/configs/medical.json`
- `ucloud/prompt_lens/configs/sports.json`

Entrypoint:
- `ucloud/prompt_lens/run_qwen_nq500.sh`

## Non-Negotiable Rules

1. Save the Natural Questions subset first.
2. Save hidden-state payloads for all required models in this run:
   - base
   - risky
   - medical
   - sports
3. Save the shared base hidden-state payload once.
4. Reuse that same base payload for risky, medical, and sports.
5. Save one hidden-state comparison payload per FT model.
6. Do not persist full logits tensors.
7. Do not persist full probability tensors.
8. Do downstream projection/normalization/analysis from the saved hidden states.
9. For `raw`, project hidden states directly.
10. For `model_norm`, apply final norm and then project.
11. Save the downstream summaries and correlations.
12. Do not recompute the base if the shared base payload already exists.
13. Do not move outputs outside `tmp/`.

## Canonical Output Tree

Exactly these paths are the canonical outputs for this run:

- dataset:
  - `tmp/em_qwen/datasets/nq_train_500_query_only.jsonl`
- shared base payload:
  - `tmp/em_qwen/prompt_lens_nq_500/shared/base_qwen_prompt_lens.pt`
- risky comparison payload:
  - `tmp/em_qwen/prompt_lens_nq_500/risky/data/risky_qwen_prompt_lens.pt`
- medical comparison payload:
  - `tmp/em_qwen/prompt_lens_nq_500/medical/data/medical_qwen_prompt_lens.pt`
- sports comparison payload:
  - `tmp/em_qwen/prompt_lens_nq_500/sports/data/sports_qwen_prompt_lens.pt`

For each comparison root (`risky`, `medical`, `sports`), exactly these downstream outputs are expected:

- `data/<comparison>_qwen_prompt_lens.pt`
- `summaries/mode_specific_summary.json`
- `summaries/layerwise_observations.jsonl`
- `summaries/layerwise_statistics.json`
- `summaries/layerwise_correlations.json`
- `figures/`
- `run_manifest.json`

No additional persisted logits or probability payload files are part of the intended output tree.

## Exact Order

### Step 1: Build dataset file

Create:
- `tmp/em_qwen/datasets/nq_train_500_query_only.jsonl`

That file must contain:
- exactly 500 rows
- query/question text only
- stable sampled subset for the chosen seed

### Step 2: Shared base payload

Shared base hidden-state payload path:
- `tmp/em_qwen/prompt_lens_nq_500/shared/base_qwen_prompt_lens.pt`

Behavior:
- if this file already exists, reuse it
- if it does not exist, collect it once

This payload is the base artifact reused across all three comparisons.

What this base payload contains:
- all `500` sampled dataset rows for this run condition
- one row entry per sampled prompt
- layer records for the captured layers for the base model
- hidden states per captured layer
- tokens
- attention mask
- `lm_head_weight`
- `final_norm`
- no model-specific comparison data

What this base payload does not contain:
- persisted full logits tensors
- persisted full probability tensors

### Step 3: Comparison payloads

For each comparison, save exactly one comparison hidden-state payload:

Risky:
- `tmp/em_qwen/prompt_lens_nq_500/risky/data/risky_qwen_prompt_lens.pt`

Medical:
- `tmp/em_qwen/prompt_lens_nq_500/medical/data/medical_qwen_prompt_lens.pt`

Sports:
- `tmp/em_qwen/prompt_lens_nq_500/sports/data/sports_qwen_prompt_lens.pt`

These are hidden-state payloads for downstream analysis.

What each comparison payload contains:
- all `500` sampled dataset rows for this run condition
- one row entry per sampled prompt
- layer records for the captured layers for that comparison model
- hidden states per captured layer
- tokens
- attention mask
- `lm_head_weight`
- `final_norm`
- only that comparison model's hidden-state payload

What each comparison payload does not contain:
- persisted full logits tensors
- persisted full probability tensors

### Step 4: Downstream analysis

For each comparison:
- load the shared base hidden-state payload
- load the comparison hidden-state payload
- reconstruct logits only when needed for analysis
- apply raw projection where needed
- apply model-norm normalization then projection where needed
- compute prompt-lens downstream summaries from those reconstructed values

This means:
- risky downstream analysis uses:
  - shared base payload
  - risky comparison payload
- medical downstream analysis uses:
  - shared base payload
  - medical comparison payload
- sports downstream analysis uses:
  - shared base payload
  - sports comparison payload

### Step 5: Save downstream outputs

For each comparison, save:

- `summaries/mode_specific_summary.json`
- `summaries/layerwise_observations.jsonl`
- `summaries/layerwise_statistics.json`
- `summaries/layerwise_correlations.json`
- `figures/`
- `run_manifest.json`

## What Is Saved

Saved:
- the 500-sample query-only dataset file
- one shared base hidden-state payload
- one comparison hidden-state payload per FT model
- mode-specific summary JSON
- layerwise observation JSONL
- layerwise descriptive statistics JSON
- layerwise correlation JSON
- figures
- manifest

Exactly one shared base payload should exist for this run condition.
Exactly one comparison payload should exist per comparison root.

## What Is Reused

Reused across risky/medical/sports:
- `tmp/em_qwen/prompt_lens_nq_500/shared/base_qwen_prompt_lens.pt`

Meaning of reuse:
- do not recollect the base hidden states for each comparison
- load this one shared base payload in the downstream analysis for risky
- load this one shared base payload in the downstream analysis for medical
- load this one shared base payload in the downstream analysis for sports

Not reused:
- comparison payloads
- comparison-specific summaries
- comparison-specific figures

## What Is Not Saved

Do not save:
- full persisted logits tensors
- full persisted probability tensors
- duplicate base payloads inside each comparison directory
- duplicate copies of the same comparison payload under different names

## Downstream Analysis Meaning

### `mode_specific_summary.json`

Layerwise means for the usual prompt-lens metrics.

These must include next-token prediction comparison metrics such as:
- top-1 overlap / agreement
- top-5 Jaccard
- top-10 Jaccard
- base top-1 next-token accuracy
- base top-5 next-token accuracy
- base top-10 next-token accuracy
- FT top-1 next-token accuracy
- FT top-5 next-token accuracy
- FT top-10 next-token accuracy

### `layerwise_observations.jsonl`

Observation-level per-layer values used for downstream statistics.

Contains, per record:
- group/variant identity
- layer
- usable positions
- hidden metrics
- raw-mode metrics
- model-norm-mode metrics

The saved observation-level metrics should include, where applicable:
- hidden cosine
- hidden L2
- next-token top-1 agreement
- next-token top-5 Jaccard
- next-token top-10 Jaccard
- base top-1 next-token accuracy against the ground-truth next token
- base top-5 next-token accuracy against the ground-truth next token
- base top-10 next-token accuracy against the ground-truth next token
- FT top-1 next-token accuracy against the ground-truth next token
- FT top-5 next-token accuracy against the ground-truth next token
- FT top-10 next-token accuracy against the ground-truth next token
- TVD
- JS divergence

Exact metric keys expected from the hidden-only runner:
- `hidden_cosine`
- `hidden_l2`
- `jaccard_top1`
- `jaccard_top5`
- `jaccard_top10`
- `base_top1_next_token_accuracy`
- `base_top5_next_token_accuracy`
- `base_top10_next_token_accuracy`
- `ft_top1_next_token_accuracy`
- `ft_top5_next_token_accuracy`
- `ft_top10_next_token_accuracy`
- `tvd`
- `js`

### `layerwise_statistics.json`

For each metric and each layer:
- `n`
- `mean`
- `std`
- `median`
- `min`
- `max`
- `ci95_mean`

This must include descriptive summaries for:
- hidden cosine
- hidden L2
- next-token top-1 agreement
- next-token top-5 Jaccard
- next-token top-10 Jaccard
- base top-1 next-token accuracy
- base top-5 next-token accuracy
- base top-10 next-token accuracy
- FT top-1 next-token accuracy
- FT top-5 next-token accuracy
- FT top-10 next-token accuracy
- TVD
- JS divergence

### `layerwise_correlations.json`

For each metric and each non-final layer:
- correlation against the final layer
- matched observation pairing
- `pearson`
- `spearman`
- `p_value`
- `ci95`

This must include correlations for:
- hidden cosine
- hidden L2
- next-token top-1 agreement
- next-token top-5 Jaccard
- next-token top-10 Jaccard
- base top-1 next-token accuracy
- base top-5 next-token accuracy
- base top-10 next-token accuracy
- FT top-1 next-token accuracy
- FT top-5 next-token accuracy
- FT top-10 next-token accuracy
- TVD
- JS divergence

## Correlation Rule

Correlations must be computed from matched observation values.

Do not average across positions before correlation.

Use observation values first, then correlate each layer against the final layer.

For next-token accuracy metrics:
- the target is the ground-truth next token from the teacher-forced prompt
- prediction at position `t` is evaluated against token `t+1`
- the last token position is excluded because it has no true next token

For overlap metrics:
- `jaccard_top1`, `jaccard_top5`, and `jaccard_top10` compare base vs FT prediction sets at the same teacher-forced position
- these are not the same as next-token accuracy metrics

## GPU Rule

Run on GPU.

Do not intentionally run on CPU.

Configs already set:
- `force_cpu: false`

## Commands

Build dataset:

```bash
PYTHONPATH=src python3 ucloud/pipelines/prompt_lens/build_query_only_dataset.py \
  --output-path tmp/em_qwen/datasets/nq_train_500_query_only.jsonl \
  --sample-count 500 \
  --split train \
  --seed 13
```

Run all three:

```bash
bash ucloud/prompt_lens/run_qwen_nq500.sh
```

This is the canonical bundle entrypoint for the full run.

Run one comparison manually:

```bash
PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.run_hidden_only_prompt_lens_pipeline \
  --config ucloud/prompt_lens/configs/risky.json
```

```bash
PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.run_hidden_only_prompt_lens_pipeline \
  --config ucloud/prompt_lens/configs/medical.json
```

```bash
PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.run_hidden_only_prompt_lens_pipeline \
  --config ucloud/prompt_lens/configs/sports.json
```

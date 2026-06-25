# Lens Workflows

![LogitDiff Overview](../assests/docs_figures/logit_diff_framework_overview_1.png)

## Overview

`LogitDiff` has two closely related wrapper-backed lens workflows inside the same toolkit:

- prompt lens: fixed-prompt capture and comparison through `LogitLensWrapper`
- generation lens: continuation-time capture and comparison through `GenerateLensWrapper` or `CustomGenerationLensWrapper`

They belong to the same toolkit surface, use the same wrapper layer, and feed into the same downstream families such as heatmaps, patchscopes, and prism-style analysis.

The shared wrapper-controlled behavior is documented in [README_wrappers.md](README_wrappers.md). This guide focuses on how those wrapper types show up as the prompt-lens and generation-lens workflow surfaces inside `LogitDiff`.

## Prompt lens

Use the prompt lens when you want to compare:

- two models on the same prompt
- prompt-side behavior before generation starts
- prompt-side datasets and reusable capture artifacts
- `raw` vs `model_norm` prompt decoding

Single prompt capture:

```bash
PYTHONPATH=src python pipelines/capture_prompt_artifacts.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<run-name>.pt \
  --dtype bfloat16 \
  --use-chat-template \
  --prompt-format chat_template \
  --system-prompt "<system-prompt>" \
  --truncate \
  --max-length 512 \
  --padding longest \
  --force-include-input \
  --force-include-output \
  --norm-modes raw model_norm \
  --collect-components
```

Dataset-style prompt capture:

```bash
PYTHONPATH=src python pipelines/capture_prompt_artifacts.py \
  --model-name <model-name> \
  --dataset-path <dataset.jsonl> \
  --text-field text \
  --output-path tmp/artifacts/<dataset-run>.pt \
  --dtype bfloat16 \
  --prompt-format plain \
  --truncate \
  --max-length 512 \
  --padding longest \
  --force-include-input \
  --norm-modes raw model_norm
```

Prompt-side comparison:

```bash
PYTHONPATH=src python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/<ft-run>.pt \
  --base-artifact tmp/artifacts/<base-run>.pt \
  --comparison-output tmp/artifacts/<comparison-run>.pt \
  --readout-mode model_norm \
  --metric topk_jaccard_ft_base \
  --plot-output tmp/artifacts/<comparison-run>.html
```

Prompt-side heatmaps can run either from a saved payload or as a live compute-and-plot workflow through `pipelines/plot_prompt_heatmap.py`, including `force_include_input`, `force_include_output`, tokenization controls, prompt-vs-dataset inputs, and the same plotting controls used on the generation side.

## Generation lens

Use the generation lens when you want to compare:

- base model vs. finetuned model
- two checkpoints
- one model under two prompt formats
- one model under different template settings
- one model under different decoding settings such as `temperature` or sampling

Direct generation capture:

```bash
PYTHONPATH=src python pipelines/capture_generation_artifacts.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<generation-run>.pt \
  --dtype bfloat16 \
  --prompt-format plain \
  --truncate \
  --max-length 512 \
  --padding longest \
  --max-new-tokens 32 \
  --batch-size 8 \
  --force-include-input \
  --force-include-output \
  --norm-modes raw unit_norm eps_norm model_norm
```

Config-driven generation-lens path:

```bash
PYTHONPATH=src python pipelines/em_qwen/run_gen_lens.py \
  --config configs/em_qwen/gen_lens/chat_template/risky_14.json
```

```bash
PYTHONPATH=src python pipelines/quant_llama/run_gen_lens.py \
  --config configs/quant_llama/gen_lens/hf1bit_14.json
```

The current repository already includes config families such as:

- `configs/em_qwen/gen_lens/chat_template/...`
- `configs/em_qwen/gen_lens/no_template/...`
- `configs/quant_llama/gen_lens/...`

Generation heatmaps can also run either from a saved payload or as a live compute-and-plot workflow through `pipelines/plot_generation_heatmap.py`, including prompt-vs-dataset inputs, `force_include_input`, `force_include_output`, tokenization controls, and generation controls such as `max_new_tokens`. The plotting surface is the sibling of the prompt-side heatmap surface, not a separate plotting system.

## Shared controls and formatting

Both prompt and generation workflows surface:

- `--use-chat-template`
- `--prompt-format`
- `--system-prompt`
- `--truncate`
- `--max-length`
- `--padding`
- `--force-include-input`
- `--force-include-output`
- `--norm-modes`

Generation-side direct capture exposes:

- `--max-new-tokens`
- `--batch-size`
- `--no-add-special-tokens`
- `--analyze-special-tokens`
- `--custom-generate`

## Generation config controls

Top-level fields:

- `base_model_id`
- `comparison_model_id` or `comparison_adapter_path`
- `output_root`

Generation-lens fields inside `gen_lens`:

- `prompt_source`
- `prompt_key`
- `max_new_tokens`
- `top_k`
- `comparison_top_ks`
- `layers`
- `norm_mode`
- `do_sample`
- `temperature`
- `use_cache`
- `template_name`
- `prompt_format`
- `use_chat_template`
- `system_prompt`

Minimal shape:

```json
{
  "scenario": "example",
  "base_model_id": "<base-model>",
  "comparison_model_id": "<comparison-model>",
  "output_root": "tmp/<run-root>",
  "gen_lens": {
    "prompt_source": "datasets/<prompts>.jsonl",
    "prompt_key": "prompt",
    "max_new_tokens": 14,
    "top_k": 10,
    "comparison_top_ks": [1, 5, 10],
    "layers": [0, 1, 2, 3],
    "norm_mode": "raw",
    "do_sample": false,
    "temperature": 1.0,
    "template_name": "chat_template_10",
    "prompt_format": "chat_template",
    "use_chat_template": true,
    "system_prompt": null
  }
}
```

## Prompt formatting and generation conditions

The generation code already distinguishes between:

- plain prompting
- chat-template prompting
- `user_assistant_prefix` prompting
- optional `system_prompt`

So generation-lens comparisons are not limited to model A vs. model B. They can also compare the same model across:

- template vs. no template
- plain vs. prefixed prompting
- different system prompts
- deterministic vs. sampled generation

## Datasets, batching, and token handling

The config-driven generation pipeline runs from a prompt source file. The lower-level generation collectors in the repo also already support:

- dataset-driven generation analysis
- `batch_size`
- `max_new_tokens`
- `force_include_input`
- `force_include_output`
- `norm_modes`
- `collect_components`
- `project_component_logits`
- special-token filtering
- prompt-token and generated-token separation
- attention-mask-aware token storage

That is why the generation lens should be read as both a single-prompt workflow and a dataset workflow.

## Direct generation capture

The main project also now exposes a direct generation-capture entrypoint when you want generation artifacts without going through a config bundle first:

Single prompt:

```bash
PYTHONPATH=src python pipelines/capture_generation_artifacts.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<generation-run>.pt \
  --dtype bfloat16 \
  --prompt-format plain \
  --system-prompt "<system-prompt>" \
  --analyze-special-tokens \
  --truncate \
  --max-length 512 \
  --padding longest \
  --max-new-tokens 32 \
  --force-include-input \
  --force-include-output \
  --norm-modes raw unit_norm eps_norm model_norm \
  --collect-components
```

Dataset-style generation capture:

```bash
PYTHONPATH=src python pipelines/capture_generation_artifacts.py \
  --model-name <model-name> \
  --dataset-path <dataset.jsonl> \
  --text-field analysis_text \
  --label-field label \
  --output-path tmp/artifacts/<generation-dataset-run>.pt \
  --dtype bfloat16 \
  --use-chat-template \
  --prompt-format chat_template \
  --system-prompt "<system-prompt>" \
  --truncate \
  --max-length 512 \
  --padding longest \
  --max-new-tokens 32 \
  --batch-size 8 \
  --force-include-input \
  --force-include-output \
  --norm-modes raw unit_norm eps_norm model_norm \
  --collect-components
```

Important direct generation-capture controls:

- `--prompt` or `--dataset-path`
- `--text-field`
- `--label-field`
- `--prompt-format`
- `--use-chat-template`
- `--system-prompt`
- `--no-add-special-tokens`
- `--analyze-special-tokens`
- `--truncate`
- `--max-length`
- `--padding`
- `--max-new-tokens`
- `--batch-size`
- `--force-include-input`
- `--force-include-output`
- `--norm-modes`
- `--collect-components`
- `--project-component-logits`
- `--custom-generate`

## Output and reuse

Prompt and generation outputs are both meant to be reused rather than recomputed blindly.

Generation outputs are reused by:

- generation Jaccard heatmaps
- generation next-token verification heatmaps
- generation paper heatmaps
- generation patchscopes
- generation follow-up comparison analyses

Prompt outputs are reused by:

- prompt comparisons
- comparison-metric heatmaps
- prompt verification heatmaps
- prompt patchscopes
- prism-style prompt analyses

## Plotting

Once a generation-lens run has produced its saved payload, plot it with:

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --input-path tmp/<run-root>/data/<layerwise-json>.json \
  --output-path tmp/<run-root>/figures/<heatmap>.pdf \
  --plot-kind jaccard \
  --prompt-index 0 \
  --top-k 10 \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --start-position 0 \
  --end-position 64 \
  --max-layers 6 \
  --layer-selection all \
  --x-tick-mode ft_generated \
  --x-tick-mode-secondary base_generated \
  --title "Generation LogitDiff" \
  --colorscale RdBu
```

Prompt and generation heatmaps are documented together in [README_heatmaps.md](README_heatmaps.md).

## How to interpret the result

Read prompt-lens output as the fixed-prompt baseline and generation-lens output as the continuation-time trajectory. The key point is that both are part of the same `LogitDiff` lens workflow rather than separate toolkits.

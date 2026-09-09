# Wrappers and Readouts

![LogitDiff Overview](../assests/docs_figures/logit_diff_framework_overview_1.png)

## Overview

`LogitDiff` is organized around wrapper types. Prompt lens, generation lens, patching, and downstream decoding are wrapper-backed modes inside the same toolkit. The wrapper surface is what keeps tokenization, masks, normalization, LM-head projection, and saved activations aligned across analyses.

## Main wrappers

- `LogitLensWrapper`
  Prompt-side forward capture and prompt-lens decoding.
- `GenerateLensWrapper`
  Standard generation-lens runs built around generation-time hidden-state capture.
- `CustomGenerationLensWrapper`
  Generation-time analysis when the custom generation loop is the right fit.
- `PatchingLensWrapper`
  Prompt and generation intervention workflows.

## Wrapper-backed modes

The main wrapper-backed collection and analysis modes are:

- prompt lens
  Uses `LogitLensWrapper` for fixed-prompt capture, prompt-side decoding, prompt comparisons, and prompt heatmaps.
- generation lens
  Uses `GenerateLensWrapper` or `CustomGenerationLensWrapper` for continuation-time capture, generation comparisons, and generation heatmaps.
- patching and patchscopes
  Uses `PatchingLensWrapper` plus saved prompt or generation artifacts for prompt patchscope, generation patchscope, and patch sweeps.

Those workflow surfaces now include both:

- saved-artifact reuse
- live compute-and-plot or live compute-and-analyze entrypoints where the repo already supports them

## Readout modes

The current prompt capture path preserves what is needed for:

- `raw`
- `model_norm`

Older and auxiliary analyses in the repo also compare against `Tuned Lens`, but that is a separate learned readout rather than a native wrapper decode mode.

## Prompt formatting and token handling

The wrapper layer is where `LogitDiff` keeps prompt formatting and token handling consistent:

- plain prompts
- chat-template prompts
- `user_assistant_prefix` prompts
- optional `system_prompt`
- special-token-aware tokenization and decoding
- attention-mask-aware trimming
- padding-aware downstream analysis

That same logic is used across prompt capture, generation, patchscopes, backward artifacts, and component analyses.

## Shared wrapper-controlled behavior

These behaviors belong to the wrapper layer and should be understood once for both prompt and generation workflows:

- tokenization
- prompt formatting
- `plain`, `chat_template`, and `user_assistant_prefix` modes
- optional `system_prompt`
- special-token handling
- attention-mask trimming
- padding-aware downstream analysis
- `force_include_input`
- `force_include_output`
- `norm_modes`
- artifact reuse across downstream analyses

## Prompt-side capture surface

The public prompt capture entrypoint is:

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
  --force-include-input \
  --force-include-output \
  --norm-modes raw model_norm \
  --collect-components
```

Prompt capture also supports dataset-style runs:

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
  --force-include-input \
  --norm-modes raw model_norm
```

Important prompt-side controls already exposed in code:

- `--tokenizer-name`
- `--adapter-path`
- `--dtype`
- `--device-map`
- `--load-in-4bit`
- `--load-in-8bit`
- `--use-chat-template`
- `--prompt-format`
- `--system-prompt`
- `--no-add-special-tokens`
- `--truncate`
- `--max-length`
- `--padding`
- `--force-include-input`
- `--force-include-output`
- `--norm-modes`
- `--collect-components`
- `--project-component-logits`
- `--save-logits`
- `--stable-analysis`
- `--debug`

## Generation-side surface

The generation side has both direct capture entrypoints and config-driven pipeline entrypoints.

Direct capture:

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

Config-driven runs already in the repo:

```bash
PYTHONPATH=src python pipelines/em_qwen/run_gen_lens.py \
  --config configs/em_qwen/gen_lens/chat_template/risky_14.json
```

```bash
PYTHONPATH=src python pipelines/quant_llama/run_gen_lens.py \
  --config configs/quant_llama/gen_lens/hf1bit_14.json
```

The config is where generation-side controls live:

- `prompt_source`
- `prompt_key`
- `max_new_tokens`
- `top_k`
- `comparison_top_ks`
- `layers`
- `norm_mode`
- `do_sample`
- `temperature`
- `template_name`
- `prompt_format`
- `use_chat_template`
- `system_prompt`

The generation collectors in the repo also support:

- dataset-driven runs
- `batch_size`
- `max_new_tokens`
- `do_sample`
- `temperature`
- `seed`
- `truncate`
- `max_length`
- padding control
- special-token inclusion or exclusion
- prompt formatting and system prompts
- optional component capture

The public prompt and generation heatmap CLIs now mirror that same split:

- `--input-path` for saved-payload plotting
- model/prompt/dataset/capture controls for live compute-and-plot

## Batches, datasets, and masks

The wrapper and collector stack is meant to work for more than one prompt:

- single prompts
- prompt datasets
- model-response datasets
- generation datasets
- batched generation collection

Prompt-side code trims to the effective attention-mask span so padding-only token positions do not leak into downstream plots. Generation-side collectors also carry the attention mask and special-token controls needed to keep prompt and generated token regions interpretable.

Public wrapper-facing capture parameters now exposed in the main toolkit include:

- prompt-side:
  `--prompt`, `--dataset-path`, `--text-field`, `--use-chat-template`, `--prompt-format`, `--system-prompt`, `--truncate`, `--max-length`, `--padding`, `--force-include-input`, `--force-include-output`, `--norm-modes`, `--collect-components`, `--project-component-logits`, `--save-logits`, `--stable-analysis`, `--debug`
- generation-side:
  `--prompt`, `--dataset-path`, `--text-field`, `--label-field`, `--use-chat-template`, `--prompt-format`, `--system-prompt`, `--truncate`, `--max-length`, `--padding`, `--no-add-special-tokens`, `--analyze-special-tokens`, `--max-new-tokens`, `--batch-size`, `--do-sample`, `--temperature`, `--seed`, `--force-include-input`, `--force-include-output`, `--norm-modes`, `--collect-components`, `--project-component-logits`, `--custom-generate`, `--stable-analysis`

## How to interpret the result

The wrappers are not the result by themselves. Their job is to make sure prompt lens and generation lens stay two modes of the same LogitDiff toolkit rather than drifting into separate tokenization, masking, or decode surfaces.

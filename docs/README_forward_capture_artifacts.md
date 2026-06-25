# Forward Capture

![Forward Capture Artifact](../assests/docs_figures/logit_diff_forward_artifact_2.png)

## Overview

Forward capture is the prompt-side artifact workflow in `LogitDiff`. It saves the tokenized prompt, attention mask, layer records, and optional projected logits so that downstream analyses can reuse the same captured states.

## When to use it

Use forward capture when you want to:

- inspect one prompt across layers
- compare two prompt runs later without recapturing
- run dataset-style prompt capture
- reuse the same hidden states for heatmaps, comparisons, patchscopes, and prism-style analyses

## Public entrypoint

Single prompt:

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
  --collect-components \
  --project-component-logits \
  --save-logits
```

Dataset-style run:

```bash
PYTHONPATH=src python pipelines/capture_prompt_artifacts.py \
  --model-name <model-name> \
  --dataset-path <dataset.jsonl> \
  --text-field text \
  --label-field label \
  --output-path tmp/artifacts/<dataset-run>.pt \
  --dtype bfloat16 \
  --prompt-format plain \
  --truncate \
  --max-length 512 \
  --padding longest \
  --force-include-input \
  --force-include-output \
  --norm-modes raw model_norm
```

## Important prompt capture controls

The public capture command already exposes:

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

Example with the full prompt-side control surface:

```bash
PYTHONPATH=src python pipelines/capture_prompt_artifacts.py \
  --model-name <model-name> \
  --dataset-path <dataset.jsonl> \
  --text-field text \
  --output-path tmp/artifacts/<dataset-run>.pt \
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
  --collect-components \
  --project-component-logits \
  --save-logits
```

## Tokenization, masks, and padding

Prompt capture keeps:

- token ids
- token text
- attention mask
- formatted prompt text

The prompt-side collector trims to the effective attention-mask span before saving layerwise content, so padding-only positions do not become fake evidence in downstream plots.

## Input embedding and output L+1

For prompt-side artifact reuse, the important boundary controls are:

- `--force-include-input`
  Keeps the input embedding step in the saved artifact so downstream analyses can include the input-side readout.
- `--force-include-output`
  Keeps the final output-side readout after the last probed transformer block.
- `--norm-modes raw model_norm`
  Saves both the direct unembedding view and the final-layer-norm readout.

In practice, `model_norm` is the L+1-style readout: the hidden state is passed through the model's final normalization before the LM head projection. That is the readout to use when you want the final output-side view documented explicitly.

## What gets reused later

Saved prompt artifacts are used by:

- prompt comparisons
- comparison-metric heatmaps
- prompt next-token verification heatmaps
- patchscopes
- backward follow-up analyses
- prism-style component analyses

## Generation relation

Generation-lens workflows use the same general ideas, but along the continuation path rather than only the fixed prompt. The matching public entrypoint is:

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

For generation-specific controls such as `max_new_tokens`, template choice, prompt formatting, batching, and decoding settings, see [README_lens_workflows.md](README_lens_workflows.md).

## How to interpret the result

Treat a forward-capture artifact as the reusable record of one prompt run. The value is that later analyses can read from the same captured states instead of silently recomputing under different tokenization or decode conditions.

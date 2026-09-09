# Comparison Artifacts

![Comparison Artifact](../assests/docs_figures/logit_diff_comparison_artifact_3.png)

## Overview

`LogitDiff` has two comparison-style reuse paths:

- prompt-side canonical comparison artifacts from saved prompt captures
- generation-side comparison payloads that serve the same reuse role for generation analyses

The prompt-side comparison artifact is its own saved format. The generation-side comparison path currently reuses the generation-style payload rather than the prompt comparison-artifact file format.

## Prompt-side use

Use the prompt-side comparison artifact when you want to:

- compare two prompt captures on the same text
- compare base vs. finetuned prompt behavior
- compare two prompt-side conditions under the same tokenization surface
- export a comparison-metric heatmap from saved artifacts

## Prompt-side inputs

- one saved prompt artifact for the `ft` side
- one saved prompt artifact for the `base` side
- one readout mode such as `raw` or `model_norm`
- one output path

Those prompt artifacts can come from either:

- a single-prompt capture
- a dataset-style capture using the same wrapper-controlled tokenization and masking surface

For stable prompt-side comparisons, the two source artifacts should be captured with matching settings for:

- `prompt_format`
- `use_chat_template`
- `system_prompt`
- `truncate`
- `max_length`
- `padding`
- `force_include_input`
- `force_include_output`
- `norm_modes`

## Prompt-side command

```bash
PYTHONPATH=src python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/<ft-run>.pt \
  --base-artifact tmp/artifacts/<base-run>.pt \
  --comparison-output tmp/artifacts/<comparison-run>.pt \
  --readout-mode model_norm \
  --metric topk_jaccard_ft_base \
  --plot-output tmp/artifacts/<comparison-run>.html \
  --title "Prompt Comparison"
```

Optional direct plot export:

```bash
PYTHONPATH=src python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/<ft-run>.pt \
  --base-artifact tmp/artifacts/<base-run>.pt \
  --comparison-output tmp/artifacts/<comparison-run>.pt \
  --readout-mode model_norm \
  --metric jsd_ft_base \
  --plot-output tmp/artifacts/<comparison-run>.pdf
```

## Prompt-side output

The prompt-side comparison artifact stores prompt-side divergence metrics such as token-overlap and probability-distribution comparisons. It is then reused by:

- comparison-metric heatmaps
- prompt-side follow-up selection
- prism-style comparison workflows

With `--readout-mode model_norm`, the comparison uses the final-layer-norm readout from the captured output-side L+1 projection.

## Generation-side analogue

Generation-side comparison can be driven either by a saved generation payload or by the live compute path in the public generation heatmap wrapper. The plotting controls stay on the same wrapper surface in both cases, while the live path additionally takes the generation inputs that produce the payload.

Saved generation-side path:

```bash
PYTHONPATH=src python pipelines/em_qwen/run_gen_lens.py \
  --config configs/em_qwen/gen_lens/chat_template/risky_14.json
```

and then:

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --input-path tmp/<run-root>/data/<layerwise-json>.json \
  --output-path tmp/<run-root>/figures/<generation-heatmap>.pdf \
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
  --title "Generation Comparison" \
  --colorscale RdBu \
  --show-marginals
```

Live generation-side path:

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --model-name <base-model-name> \
  --comparison-model-name <comparison-model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/<run-root>/figures/<generation-heatmap>.pdf \
  --plot-kind jaccard \
  --analysis-topk 10 \
  --top-k 10 \
  --truncate \
  --max-length 512 \
  --padding longest \
  --force-include-input \
  --force-include-output \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --start-position 0 \
  --end-position 64 \
  --max-layers 6 \
  --layer-selection all \
  --x-tick-mode ft_generated \
  --x-tick-mode-secondary base_generated \
  --title "Generation Comparison" \
  --colorscale RdBu \
  --show-marginals \
  --max-new-tokens 32 \
  --batch-size 8 \
  --do-sample \
  --temperature 0.7 \
  --seed 17 \
  --comparison-top-ks 1 5 10 \
  --norm-modes raw unit_norm eps_norm model_norm
```

So the comparison story should be read as one LogitDiff concept with reusable payloads and live-compute entrypoints:

- prompt-side comparison artifact
- generation-side comparison payload

# Heatmaps

![Comparison Artifact](../assests/docs_figures/logit_diff_comparison_artifact_3.png)

## Overview

`LogitDiff` includes several heatmap families built on the same general plotting surface. Prompt-side and generation-side heatmaps belong to the same toolkit and reuse the same plotting ideas wherever the code already supports them.

## Shared plotting vocabulary

The public prompt and generation heatmap wrappers both support saved-payload plotting and live compute-and-plot workflows.

Shared plotting controls:

- `--input-path: str`
- `--output-path: str`
- `--format: html | pdf`
- `--plot-kind: str`
- `--prompt-index: int`
- `--prompt-text: str`
- `--top-k: int`
- `--display-top-tokens: int`
- `--visible-cell-tokens: int`
- `--max-token-chars: int`
- `--exclude-prompt-tokens: bool`
- `--exclude-generated-tokens: bool`
- `--start-position: int`
- `--end-position: int`
- `--max-layers: int`
- `--max-divergent-layers: int`
- `--layer-selection: str`
- `--keep-last-layer-fraction: float`
- `--title: str`
- `--colorscale: str`
- `--show-marginals: bool`
- `--analysis-topk: int`

Alias notes:

- `--start-position` and `--end-position` map to `start_idx` and `end_idx`
- `--max-layers` and `--max-divergent-layers` are wrapper aliases for the visible-layer limit
- `--top-k`, `--display-top-tokens`, and `--visible-cell-tokens` are related but not identical

Live prompt/generation workflows additionally expose the capture inputs that define what gets plotted:

- `--model-name`
- `--comparison-model-name` or `--comparison-adapter-path`
- `--prompt` or `--dataset-path`
- `--text-field`
- `--label-field` where supported
- `--use-chat-template`
- `--prompt-format`
- `--system-prompt`
- `--truncate`
- `--max-length`
- `--padding`
- `--force-include-input`
- `--force-include-output`
- `--norm-modes`

Generation live workflows additionally expose generation-condition inputs:

- `--max-new-tokens`
- `--batch-size`
- `--analyze-special-tokens`
- `--custom-generate`

## Prompt heatmap families

### Prompt Jaccard heatmap

Source: `jaccard_heatmap_plotter.py`

What it plots:

- top-k token-set overlap between the two prompt-side runs across layers and positions

Input style:

- saved prompt-side results
- or live prompt capture plus comparison through the same wrapper CLI

Example:

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --input-path tmp/<prompt-results>.json \
  --output-path tmp/<prompt-jaccard>.pdf \
  --format pdf \
  --plot-kind jaccard \
  --prompt-index 0 \
  --analysis-topk 10 \
  --top-k 10 \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --exclude-generated-tokens \
  --start-position 0 \
  --end-position 64 \
  --max-layers 6 \
  --layer-selection all \
  --x-tick-mode base_generated \
  --title "Prompt Jaccard" \
  --colorscale RdBu \
  --show-marginals
```

Prompt Jaccard x-axis modes currently exposed by the public wrapper:

- `prompt`
- `base_generated`
- `position`

### Prompt next-token verification heatmap

Source: `prompt_lens_heatmap_plotter.py`

What it plots:

- whether the compared prompt-side runs share the same next-token behavior across positions and selected layers

Input style:

- saved prompt payload
- or live prompt capture plus comparison through the same wrapper CLI

Example:

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --input-path tmp/<prompt-payload>.json \
  --output-path tmp/<prompt-verification>.pdf \
  --format pdf \
  --plot-kind next_token_verification \
  --prompt-index 0 \
  --top-k 10 \
  --max-divergent-layers 6 \
  --keep-last-layer-fraction 0.5 \
  --start-position 0 \
  --end-position 64 \
  --max-token-chars 12 \
  --title "Prompt Next-Token Verification" \
  --colorscale RdBu
```

### Prompt comparison-metric heatmap

What it plots:

- scalar prompt-side comparison metrics from the saved comparison artifact

Input style:

- saved prompt comparison artifact

Example:

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --input-path tmp/artifacts/<comparison-run>.pt \
  --output-path tmp/artifacts/<comparison-run>.pdf \
  --format pdf \
  --plot-kind comparison_metric \
  --metric jsd_ft_base \
  --title "Prompt Comparison Metric" \
  --colorscale RdBu
```

This comparison-metric view is separate from the original prompt-lens heatmap families.

## Generation heatmap families

### Generation Jaccard heatmap

Source: `logitdiff_gen_plotter.py`

What it plots:

- top-k token-set overlap between the two generation-side runs across layers and token positions

Input style:

- saved generation payload
- or live generation capture plus comparison through the same wrapper CLI

Example:

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --input-path tmp/<run-root>/data/<layerwise-json>.json \
  --output-path tmp/<run-root>/figures/<generation-jaccard>.pdf \
  --format pdf \
  --plot-kind jaccard \
  --prompt-index 0 \
  --analysis-topk 10 \
  --top-k 10 \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --start-position 0 \
  --end-position 64 \
  --max-layers 6 \
  --layer-selection all \
  --x-tick-mode ft_generated \
  --x-tick-mode-secondary base_generated \
  --title "Generation Jaccard" \
  --colorscale RdBu \
  --show-marginals
```

Generation Jaccard x-axis modes currently exposed:

- `ft_generated`
- `base_generated`
- `ft_top1`
- `base_top1`
- `input_tokens`
- `position`

The primary x-axis is controlled by `--x-tick-mode` and the second x-axis is controlled by `--x-tick-mode-secondary`. The common comparison layout is primary axis = comparison or FT predictions and secondary axis = base predictions.

### Generation next-token verification heatmap

Source: `logitdiff_gen_plotter.py`

What it plots:

- whether the compared generation runs share the same next-token behavior across positions and selected layers

Input style:

- saved generation payload
- or live generation capture plus comparison through the same wrapper CLI

Example:

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --input-path tmp/<run-root>/data/<layerwise-json>.json \
  --output-path tmp/<run-root>/figures/<generation-verification>.pdf \
  --format pdf \
  --plot-kind next_token_verification \
  --prompt-index 0 \
  --top-k 10 \
  --max-divergent-layers 6 \
  --keep-last-layer-fraction 0.5 \
  --start-position 0 \
  --end-position 64 \
  --max-token-chars 12 \
  --title "Generation Next-Token Verification" \
  --colorscale RdBu
```

### Generation paper heatmaps

Sources:

- `logitdiff_gen_paper_plotter.py`
- `logitdiff_gen_paper_plotter_selected_rows.py`

What they plot:

- paper-oriented generation layouts such as chunked views and selected-row exports

Input style:

- saved generation payloads used for figure export

## Other heatmap families

### Single-model logit-lens heatmap

Source: `logit_lens_plotter.py`

Use it for single-model prompt or generation readouts when you want scalar views such as entropy, perplexity, KL or JS to previous or last layer, Jaccard to previous or last layer, accuracy, and related diagnostics.

### LDL heatmap

Source: `ldl_plotter.py`

Use it for pairwise prompt-side divergence views such as:

- `kl_div_ab`
- `kl_div_ba`
- `js_div`
- `js_dist`
- `tvd`
- `cos_sim`
- `l2_dist`
- `jaccard`
- `disagreement_correct_top1`

### ADL heatmap

Source: `adl_plotter.py`

Use it for delta-style views derived from hidden-state or readout differences, including metrics such as:

- `logit_max`
- `delta_norm`
- `ground_truth_probs`
- `entropy`
- `kl_div`

### Paired-condition heatmaps

Source: `logitdiff_pair_heatmap_plotter.py`

These cover paired-condition token heatmaps and single pairwise condition heatmaps for condition, template, or prompting comparisons.

## Live generation note

When plotting directly from a live generation run instead of a saved payload, the plotting controls stay the same. The extra inputs are the generation workflow controls that determine how the payload is produced first, such as:

- plain prompting vs. chat-template prompting
- `user_assistant_prefix` prompting
- optional `system_prompt`
- generation length and batching
- decoding and custom-generation settings exposed by the wrapper path

## Output and interpretation

All public heatmap wrappers support Plotly output to `html` or static export to `pdf`.

Read the result as a layer-by-position map over one chosen analysis family:

- Jaccard heatmaps show token-set overlap
- next-token verification heatmaps show agreement structure over selected layers
- comparison-metric heatmaps show scalar prompt-side divergence
- LDL and ADL heatmaps show alternative pairwise or delta views over the same general LogitDiff analysis space

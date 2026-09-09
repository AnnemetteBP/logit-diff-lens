# Heatmaps

![Comparison Artifact](../assests/docs_figures/logit_diff_comparison_artifact_3.png)

## Overview

`LogitDiff` uses one plotting family for prompt-side and generation-side lens analysis, plus reusable single-model and ADL heatmaps for follow-up inspection.

The main public cases are:

- prompt `LogitDiff` heatmaps
- generation `LogitDiff` heatmaps
- single-model logit-lens heatmaps
- ADL heatmaps

The older `LDL` path is still present in the repo, but public prompt-side comparison should be read through the prompt `LogitDiff` heatmap surface rather than as a separate front-door workflow.

## Shared plotting controls

Prompt and generation `LogitDiff` heatmaps share the same core plotting controls wherever the wrapper CLIs already overlap:

- `--input-path: str`
- `--output-path: str`
- `--format: html | pdf`
- `--plot-kind: str`
- `--prompt-index: int`
- `--prompt-text: str`
- `--top-k: int`
- `--analysis-topk: int`
- `--display-top-tokens: int`
- `--visible-cell-tokens: int`
- `--max-token-chars: int`
- `--exclude-prompt-tokens`
- `--exclude-generated-tokens`
- `--start-position: int`
- `--end-position: int`
- `--max-layers: int`
- `--max-divergent-layers: int`
- `--layer-selection: all | most_divergent | least_divergent`
- `--keep-last-layer-fraction: float`
- `--x-tick-mode: str`
- `--x-tick-mode-secondary: str`
- `--title: str`
- `--colorscale: str`
- `--show-marginals`

Alias notes:

- `--start-position` and `--end-position` map to `start_idx` and `end_idx`
- `--max-layers` and `--max-divergent-layers` both act as visible-layer limits in the public wrappers
- `--top-k`, `--display-top-tokens`, and `--visible-cell-tokens` are related but not identical controls

When you compute live instead of plotting from file, the same heatmap wrapper also takes the upstream capture inputs that produce the payload first:

- `--model-name`
- `--comparison-model-name` or `--comparison-adapter-path`
- `--prompt` or `--dataset-path`
- `--text-field`
- `--label-field`
- `--use-chat-template`
- `--prompt-format`
- `--system-prompt`
- `--comparison-use-chat-template`
- `--comparison-prompt-format`
- `--comparison-system-prompt`
- `--no-add-special-tokens`
- `--truncate`
- `--max-length`
- `--force-include-input`
- `--force-include-output`
- `--norm-modes`
- `--readout-mode`
- `--collect-components`
- `--project-component-logits`
- `--save-logits`
- `--stable-analysis`

Generation live runs additionally expose generation-side inputs:

- `--padding`
- `--max-new-tokens`
- `--batch-size`
- `--analyze-special-tokens`
- `--do-sample`
- `--temperature`
- `--seed`
- `--comparison-top-ks`
- `--custom-generate`

Mode contract:

- saved mode uses `--input-path` plus plotting controls only
- live mode uses model and prompt or dataset inputs and computes the payload before plotting
- the public heatmap CLIs now reject mixed saved and live inputs instead of guessing

## Prompt LogitDiff heatmaps

### Prompt Jaccard

Use this when you want the overlap structure between two prompt-side runs across layers and token positions.

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --input-path tmp/artifacts/<prompt-logitdiff-payload> \
  --output-path tmp/artifacts/<prompt-jaccard>.pdf \
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
  --x-tick-mode base_generated \
  --title "Prompt LogitDiff Jaccard" \
  --colorscale RdBu \
  --show-marginals
```

This is a saved-artifact workflow. It does not recollect the prompt run.

Prompt Jaccard x-axis labels currently support:

- `prompt`
- `base_generated`
- `position`

### Prompt next-token verification

Use this when you want the prompt-style verification layout showing shared versus non-shared next-token behavior over selected layers.

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --input-path tmp/artifacts/<prompt-logitdiff-payload> \
  --output-path tmp/artifacts/<prompt-verification>.pdf \
  --format pdf \
  --plot-kind next_token_verification \
  --prompt-index 0 \
  --top-k 10 \
  --max-divergent-layers 6 \
  --keep-last-layer-fraction 0.5 \
  --start-position 0 \
  --end-position 64 \
  --max-token-chars 12 \
  --title "Prompt LogitDiff Verification" \
  --colorscale RdBu
```

### Prompt comparison-metric heatmap

This is the separate scalar comparison-artifact view, not a replacement for the prompt `LogitDiff` heatmaps above.

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --input-path tmp/artifacts/<prompt-comparison>.pt \
  --output-path tmp/artifacts/<prompt-comparison>.pdf \
  --format pdf \
  --plot-kind comparison_metric \
  --metric jsd_ft_base \
  --title "Prompt Comparison Metric" \
  --colorscale RdBu
```

## Generation LogitDiff heatmaps

### Generation Jaccard

Use this when you want layer-by-layer overlap across continuation-time decoding.

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --input-path tmp/<run-root>/data/<generation-payload>.json \
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
  --title "Generation LogitDiff Jaccard" \
  --colorscale RdBu \
  --show-marginals
```

This is a saved-artifact workflow. It does not rerun generation.

Generation Jaccard x-axis selectors currently exposed:

- `ft_generated`
- `base_generated`
- `ft_top1`
- `base_top1`
- `input_tokens`
- `position`

The usual comparison layout is:

- primary x-axis = comparison or FT tokens
- secondary x-axis = base tokens

### Generation next-token verification

Use this when you want the generation-side verification layout instead of the Jaccard cell view.

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --input-path tmp/<run-root>/data/<generation-payload>.json \
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
  --title "Generation LogitDiff Verification" \
  --colorscale RdBu
```

### Generation paper layouts

For the paper-oriented generation exports, the main project reuses the existing plotting functions under:

- `logit_diff_lens.plotting.plot_logitdiff_top_layer_chunked_heatmap`
- `logit_diff_lens.plotting.plot_logitdiff_top_layer_selected_rows_heatmap`

These are the chunked and selected-row layouts used for paper-style figure exports from saved generation payloads.

## Single-model heatmap

Use the single-model heatmap when you are not comparing two systems, but instead want one model’s own layerwise behavior for a chosen scalar metric.

The public wrapper supports both saved data and live prompt-side plotting.

```bash
PYTHONPATH=src python pipelines/plot_single_model_heatmap.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<single-model-heatmap>.pdf \
  --format pdf \
  --metric entropy \
  --norm-mode model_norm \
  --top-k 10 \
  --force-include-input \
  --force-include-output \
  --block-steps 1 \
  --start-position 0 \
  --end-position 64 \
  --show-marginals \
  --title "Single-Model Logit Lens"
```

Supported single-model metric selector values:

- `logits_mean`
- `logits_std`
- `logit_margin`
- `probs`
- `probs_std`
- `ground_truth_probs`
- `entropy`
- `perplexity`
- `kl_div_prev`
- `kl_div_last`
- `js_div_prev`
- `js_div_last`
- `cos_sim_prev`
- `cos_sim_last`
- `l2_dist_prev`
- `l2_dist_last`
- `jaccard_prev`
- `jaccard_last`
- `topk_accuracy`

## ADL heatmap

Use ADL when you want delta-style views derived from the difference between two runs rather than the Jaccard-style comparison cell layout.

The public wrapper supports:

- live model A vs model B plotting
- live base model vs adapter plotting
- plotting from a saved ADL payload

Saved ADL mode is still separate from live comparison mode. If you pass `--input-path`, do not also pass live comparison inputs such as `--comparison-model-name` or `--prompt`.

```bash
PYTHONPATH=src python pipelines/plot_adl_heatmap.py \
  --model-name <base-model-name> \
  --comparison-model-name <comparison-model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<adl-heatmap>.pdf \
  --format pdf \
  --metric kl_div \
  --norm-mode model_norm \
  --top-k 10 \
  --force-include-input \
  --force-include-output \
  --block-steps 1 \
  --start-position 0 \
  --end-position 64 \
  --show-marginals \
  --title "ADL Heatmap"
```

Supported ADL metric selector values:

- `logit_max`
- `delta_norm`
- `ground_truth_probs`
- `entropy`
- `kl_div`

## Other existing heatmaps

The repo also still contains:

- paired-condition heatmaps in `logit_diff_lens.plotting.logitdiff_pair_heatmap_plotter`
- the older internal `LDL` comparison path

Those remain available for specialized work, but the main public comparison surface is the prompt and generation `LogitDiff` heatmap family above.

## Live plotting note

When you compute and plot in one command instead of reading from `--input-path`, the plotting controls stay the same and the wrapper additionally needs the inputs that create the payload first.

For prompt-side live plotting this means the prompt or dataset inputs, formatting mode, tokenization settings, and `force_include_input` or `force_include_output`.

For generation-side live plotting this additionally means continuation-time controls such as:

- template or no-template prompting
- `user_assistant_prefix`
- optional `system_prompt`
- `max_new_tokens`
- `batch_size`
- padding behavior

## Output and interpretation

All public heatmap wrappers save either:

- Plotly `html`
- static `pdf`

Interpretation depends on the family:

- prompt or generation `LogitDiff` Jaccard heatmaps show token-set overlap and disagreement structure
- prompt or generation verification heatmaps show next-token agreement structure over selected layers
- single-model heatmaps show one model’s own scalar layerwise behavior
- ADL heatmaps show delta-derived scalar views after comparing two runs

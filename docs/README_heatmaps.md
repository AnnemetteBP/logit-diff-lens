# Heatmaps

![Comparison Artifact](../assests/docs_figures/logit_diff_comparison_artifact_3.png)

## Overview

`LogitDiff` uses multiple heatmap families rather than one single plot. The plotting code already in the repository is the source of truth, and this guide documents that existing surface for prompt-lens, generation-lens, single-model, differential, and paired-condition workflows.

The public wrapper commands in `pipelines/` are convenience entrypoints over the same underlying plotters and save helpers. They are not separate plotting implementations.

## Shared Plotting Vocabulary

These controls recur across the existing prompt and generation heatmap families where the code already overlaps:

- `--input-path: str`
  Saved payload, JSON, JSONL, or artifact path used as plotting input.
- `--output-path: str`
  Output figure path.
- `--format: html | pdf`
  Public wrapper output format where the wrapper exposes a format switch.
- `--prompt-index: int`
  Select a prompt by index.
- `--prompt-text: str`
  Select a prompt by exact stored text instead of index.
- `--exclude-prompt-tokens: bool`
  Omit prompt-side tokens from the displayed positions. In the plotting functions this corresponds to `include_prompt_tokens=False`.
- `--exclude-generated-tokens: bool`
  Omit generated tokens from the displayed positions. In the plotting functions this corresponds to `include_generated_tokens=False`.
- `--start-position: int` and `--end-position: int`
  CLI names for position slicing. These map to `start_idx` and `end_idx` in the plotting functions.
- `--title: str`
  Figure title override.
- `--colorscale: str`
  Plotly colorscale selector used by the prompt and generation wrappers.
- `cmap: str`
  Matplotlib-style colormap parameter used by some other heatmap families such as logit-lens, LDL, and ADL.
- `--top-k: int`
  Top-k selector. In some families this controls the analysis set itself, while in others it controls how many tokens are shown or checked.
- `--display-top-tokens: int`
  How many tokens to render per cell when the plotter supports token lists inside each heatmap cell.
- `--visible-cell-tokens: int`
  Override the number of visible tokens per cell without changing the saved analysis payload.
- `--max-token-chars: int`
  Maximum rendered token width before truncation.
- `--show-marginals: bool`
  Show marginal summaries where that plotter supports them.
- `--max-layers: int`
  Visible-layer limit.
- `--max-divergent-layers: int`
  Wrapper alias for the same visible-layer limit concept in paths that still use the divergent-layer wording.
- `--layer-selection: str`
  Layer selection policy such as `all`, `most_divergent`, or `least_divergent`.
- `--keep-last-layer-fraction: float`
  Fraction-based tail selection used by the verification-style heatmaps.
- `--x-tick-mode: str`
  Primary x-axis token-label selector.
- `--x-tick-mode-secondary: str | none`
  Secondary x-axis token-label selector in plotters that expose a second x-axis.

Important alias notes:

- `max_layers` and `max_divergent_layers` are wrapper aliases around the same visible-layer limit concept where the wrapper currently accepts both.
- `start-position` and `end-position` are CLI names that map to `start_idx` and `end_idx` in the underlying plotting functions.
- `top_k`, `display_top_tokens`, `visible_cell_tokens`, and `analysis_topk` are not interchangeable. Some paths use one value for the saved overlap computation and another for the number of tokens rendered in each cell.

## Stable Saving Surface

Where the repository already exposes stable save helpers, use those names directly:

- prompt Jaccard: `save_jaccard_heatmap_html`, `save_jaccard_heatmap_pdf`, `save_jaccard_heatmap`
- prompt next-token verification: `save_logitdiff_next_token_verification_html`, `save_logitdiff_next_token_verification_pdf`
- generation Jaccard: `save_logitdiff_heatmap_html`, `save_logitdiff_heatmap_pdf`, `save_logitdiff_heatmap`
- generation next-token verification: `save_logitdiff_next_token_verification_html`, `save_logitdiff_next_token_verification_pdf`
- generation paper chunked: `save_logitdiff_top_layer_chunked_heatmap_png`, `save_logitdiff_top_layer_chunked_heatmap_pdf`
- generation paper selected-rows: `save_logitdiff_top_layer_selected_rows_heatmap_png`, `save_logitdiff_top_layer_selected_rows_heatmap_pdf`
- LDL: `save_ldl_heatmap_html`, `save_ldl_heatmap_pdf`
- paired-condition token heatmap: `save_paired_condition_token_heatmap_png`, `save_paired_condition_token_heatmap_pdf`, `save_paired_condition_token_heatmap`
- single pairwise condition heatmap: `save_single_pairwise_condition_heatmap`

Some older families such as `logit_lens_plotter.py` and `adl_plotter.py` still center their save flow around legacy `save_path` or `save_prefix` behavior rather than the newer explicit `save_*` helper naming.

## Prompt Heatmap Families

### Prompt Jaccard heatmap

Source: `jaccard_heatmap_plotter.py`

What it plots:

- top-k overlap across prompt-lens layers and token positions

Input style:

- saved prompt-lens JSON results

Public wrapper:

- `pipelines/plot_prompt_heatmap.py --plot-kind jaccard`
- this wrapper calls the same prompt Jaccard save helpers rather than a separate plotting implementation

Core parameter surface:

- `prompt_index`, `prompt_text`
- `include_prompt_tokens`, `include_generated_tokens`
- `start_idx`, `end_idx`
- `display_top_tokens`, `visible_cell_tokens`, `max_token_chars`
- `max_layers`, `layer_selection`
- `analysis_topk`
- `x_tick_mode`
- `show_marginals`
- `title`, `colorscale`

Prompt Jaccard x-axis selector values exposed now:

- `prompt`
- `base_generated`
- `position`

The current prompt wrapper does not expose an active secondary x-axis for this path, so the public docs should not imply one.

Example:

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --input-path tmp/<prompt-run>/<layerwise-json>.json \
  --output-path tmp/<prompt-run>/<prompt-jaccard>.pdf \
  --plot-kind jaccard \
  --prompt-index 0 \
  --top-k 10 \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --max-layers 5 \
  --layer-selection most_divergent \
  --x-tick-mode base_generated \
  --title "Prompt Jaccard Heatmap" \
  --colorscale RdBu \
  --show-marginals
```

Output formats:

- `html`
- `pdf`

Stable save helpers:

- `save_jaccard_heatmap_html`
- `save_jaccard_heatmap_pdf`
- `save_jaccard_heatmap`

### Prompt next-token verification heatmap

Source: `prompt_lens_heatmap_plotter.py`

What it plots:

- predictor-token versus target-token verification across selected prompt-lens layers

Input style:

- saved prompt-lens JSON or compatible payload

Public wrapper:

- `pipelines/plot_prompt_heatmap.py --plot-kind next_token_verification`
- this wrapper calls the same prompt verification save helpers

Core parameter surface:

- `prompt_index`, `prompt_text`
- `top_k`
- `max_divergent_layers`
- `keep_last_layer_fraction`
- `include_prompt_tokens`, `include_generated_tokens`
- `start_idx`, `end_idx`
- `max_token_chars`
- `title`, `colorscale`

Example:

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --input-path tmp/<prompt-run>/<layerwise-json>.json \
  --output-path tmp/<prompt-run>/<prompt-next-token-verification>.pdf \
  --plot-kind next_token_verification \
  --prompt-index 0 \
  --top-k 10 \
  --max-divergent-layers 5 \
  --keep-last-layer-fraction 0.5 \
  --title "Prompt Next-Token Verification" \
  --colorscale RdBu_r
```

Output formats:

- `html`
- `pdf`

Stable save helpers:

- `save_logitdiff_next_token_verification_html`
- `save_logitdiff_next_token_verification_pdf`

### Prompt comparison-metric heatmap

What it plots:

- saved scalar comparison metrics such as `jsd_ft_base` from prompt-comparison artifacts

Input style:

- saved comparison artifact `.pt`

Public wrapper:

- `pipelines/plot_prompt_heatmap.py --plot-kind comparison_metric`

Core parameter surface:

- `metric`
- `title`
- `colorscale`

This is a separate comparison-artifact workflow, not a replacement for the original prompt-lens plotters.

Example:

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --input-path tmp/artifacts/<comparison-name>.pt \
  --output-path tmp/artifacts/<comparison-name>.pdf \
  --plot-kind comparison_metric \
  --metric jsd_ft_base \
  --title "Prompt Comparison Metric Heatmap" \
  --colorscale Viridis
```

Output formats:

- `html`
- `pdf`

## Generation Heatmap Families

### Generation Jaccard heatmap

Source: `logitdiff_gen_plotter.py`

What it plots:

- top-k overlap across generation-lens layers and continuation positions

Input style:

- saved generation-lens JSON payload
- compatible in-memory payload when using the plotting functions directly

Public wrapper:

- `pipelines/plot_generation_heatmap.py --plot-kind jaccard`
- this wrapper calls the same generation Jaccard save helpers

Core parameter surface:

- `prompt_index`, `prompt_text`
- `include_prompt_tokens`, `include_generated_tokens`
- `start_idx`, `end_idx`
- `display_top_tokens`, `visible_cell_tokens`, `max_token_chars`
- `max_layers`, `layer_selection`
- `analysis_topk`
- `x_tick_mode`
- `x_tick_mode_secondary`
- `show_marginals`
- `title`, `colorscale`

Axis meaning:

- `x_tick_mode` controls the primary x-axis token labels.
- `x_tick_mode_secondary` controls the secondary x-axis token labels.
- The usual comparison layout is primary axis = FT or model A tokens and secondary axis = base or model B tokens.

Generation Jaccard x-axis selector values exposed now:

- `ft_generated`
- `base_generated`
- `ft_top1`
- `base_top1`
- `input_tokens`
- `position`

Example:

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --input-path tmp/<generation-run>/<layerwise-json>.json \
  --output-path tmp/<generation-run>/figures/<generation-jaccard>.pdf \
  --format pdf \
  --plot-kind jaccard \
  --prompt-index 0 \
  --top-k 10 \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --max-layers 5 \
  --layer-selection most_divergent \
  --x-tick-mode ft_generated \
  --x-tick-mode-secondary base_generated \
  --title "Generation Jaccard Heatmap" \
  --colorscale RdBu \
  --show-marginals
```

Output formats:

- `html`
- `pdf`

Stable save helpers:

- `save_logitdiff_heatmap_html`
- `save_logitdiff_heatmap_pdf`
- `save_logitdiff_heatmap`

### Generation next-token verification heatmap

Source: `logitdiff_gen_plotter.py`

What it plots:

- predictor-token versus target-token verification across selected generation layers

Input style:

- saved generation-lens JSON payload
- compatible in-memory payload when using the plotting functions directly

Public wrapper:

- `pipelines/plot_generation_heatmap.py --plot-kind next_token_verification`
- this wrapper calls the same generation verification save helpers

Core parameter surface:

- `prompt_index`, `prompt_text`
- `display_top_tokens`
- `visible_cell_tokens`
- `max_layers`
- `keep_last_layer_fraction`
- `include_prompt_tokens`, `include_generated_tokens`
- `start_idx`, `end_idx`
- `max_token_chars`
- `title`, `colorscale`
- `show_marginals`

Axis meaning:

- this family is a predictor-versus-target verification view rather than the dual generated-token axis used by generation Jaccard
- in the underlying plotter payload, the x-axis semantics correspond to predictor-token and target-token views instead of FT/base generated-token labels

Example:

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --input-path tmp/<generation-run>/<layerwise-json>.json \
  --output-path tmp/<generation-run>/figures/<generation-next-token-verification>.pdf \
  --format pdf \
  --plot-kind next_token_verification \
  --prompt-index 0 \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --max-layers 5 \
  --keep-last-layer-fraction 0.5 \
  --title "Generation Next-Token Verification" \
  --colorscale RdBu
```

Output formats:

- `html`
- `pdf`

Stable save helpers:

- `save_logitdiff_next_token_verification_html`
- `save_logitdiff_next_token_verification_pdf`

### Generation paper chunked heatmap

Source: `logitdiff_gen_paper_plotter.py`

What it plots:

- a paper-style top-layer generation heatmap split into chunks of positions

Input style:

- saved generation payload
- compatible in-memory payload

Core parameter surface:

- `prompt_index`, `prompt_text`
- `include_prompt_tokens`, `include_generated_tokens`
- `start_idx`, `end_idx`
- `top_k`
- `chunk_size`
- `title`, `colorscale`
- `annotation_text_mode`, `annotation_font_boost`
- `x_tick_mode`, `x_tick_mode_secondary`
- `max_token_chars`

Import path example:

```python
from logit_diff_lens._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_paper_plotter import (
    plot_logitdiff_top_layer_chunked_heatmap,
)
```

Output formats:

- `png`
- `pdf`

Stable save helpers:

- `save_logitdiff_top_layer_chunked_heatmap_png`
- `save_logitdiff_top_layer_chunked_heatmap_pdf`

### Generation paper selected-rows heatmap

Source: `logitdiff_gen_paper_plotter_selected_rows.py`

What it plots:

- a paper-style top-layer generation heatmap over selected position ranges

Input style:

- saved generation payload
- compatible in-memory payload

Core parameter surface:

- `selected_position_ranges`
- `prompt_index`, `prompt_text`
- `include_prompt_tokens`, `include_generated_tokens`
- `start_idx`, `end_idx`
- `top_k`
- `title`, `colorscale`
- `annotation_text_mode`, `annotation_font_boost`
- `x_tick_mode`, `x_tick_mode_secondary`
- `max_token_chars`

Import path example:

```python
from logit_diff_lens._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_paper_plotter_selected_rows import (
    plot_logitdiff_top_layer_selected_rows_heatmap,
)
```

Output formats:

- `png`
- `pdf`

Stable save helpers:

- `save_logitdiff_top_layer_selected_rows_heatmap_png`
- `save_logitdiff_top_layer_selected_rows_heatmap_pdf`

## Other Heatmap Families

### Single-model logit-lens heatmap

Source: `logit_lens_plotter.py`

What it plots:

- scalar single-model readout views across layers and token positions

Input style:

- live `LogitLensWrapper`
- saved logit-lens plot payload

Core parameter surface:

- `norm_mode`, `topk`
- `force_include_input`, `force_include_output`
- `mark_correct_preds`, `show_marginals`, `block_steps`
- `start_idx`, `end_idx`
- `cmap`, `title`, `vmin`, `vmax`, `auto_vmin_vmax`
- `fig_width`, `fig_height`
- metric toggles such as `entropy`, `perplexity`, `ground_truth_probs`, `kl_div_prev`, `kl_div_last`, `js_div_prev`, `js_div_last`, `jaccard_prev`, `jaccard_last`, `topk_accuracy`, `logits_std`, `logit_margin`, `probs`, `probs_std`, `cos_sim_prev`, `cos_sim_last`, `l2_dist_prev`, `l2_dist_last`

Import path example:

```python
from logit_diff_lens._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logit_lens_plotter import (
    plot_logit_lens_heatmap,
)
```

### LDL heatmap

Source: `ldl_plotter.py`

What it plots:

- pairwise divergence and distance views between two systems across layers and token positions

Input style:

- live pair of wrappers
- saved LDL plot payload

Core parameter surface:

- `norm_mode`, `topk`
- `force_include_input`, `force_include_output`
- `mark_correct_preds`, `show_marginals`, `block_steps`
- `start_idx`, `end_idx`
- `cmap`, `title`, `font_color`, `vmin`, `vmax`, `auto_vmin_vmax`
- `focus_user_assistant_span`
- metric toggles such as `kl_div_ab`, `kl_div_ba`, `js_div`, `js_dist`, `tvd`, `cos_sim`, `l2_dist`, `jaccard`, `disagreement_correct_top1`, `perplexity_diff`

Import path example:

```python
from logit_diff_lens._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.ldl_plotter import (
    plot_ldl_heatmap,
)
```

Output helpers include:

- `save_ldl_heatmap_html`
- `save_ldl_heatmap_pdf`

### ADL heatmap

Source: `adl_plotter.py`

What it plots:

- delta-style scalar comparisons across layers and token positions

Input style:

- live pair of wrappers
- saved ADL plot payload

Core parameter surface:

- `norm_mode`, `topk`
- `force_include_input`, `force_include_output`
- `mark_correct_preds`, `show_marginals`, `block_steps`
- `start_idx`, `end_idx`
- `cmap`, `title`, `vmin`, `vmax`, `auto_vmin_vmax`
- `fig_width`, `fig_height`
- metric toggles such as `logit_max`, `delta_norm`, `ground_truth_probs`, `entropy`, `kl_div`

Import path example:

```python
from logit_diff_lens._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.adl_plotter import (
    plot_adl_heatmap,
)
```

### Paired-condition token heatmap

Source: `logitdiff_pair_heatmap_plotter.py`

What it plots:

- token-level heatmaps for paired condition or template comparisons

Input style:

- saved comparison JSONL

Core parameter surface:

- `group_id`, `prompt_substring`
- `variant`
- `metric`
- `max_layers`, `layer_selection`
- `max_token_chars`
- `colorscale`, `title`
- side labels and tokenizer-path overrides

Import path example:

```python
from logit_diff_lens._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_pair_heatmap_plotter import (
    plot_paired_condition_token_heatmap,
)
```

Output helpers include:

- `save_paired_condition_token_heatmap_png`
- `save_paired_condition_token_heatmap_pdf`
- `save_paired_condition_token_heatmap`

### Single pairwise condition heatmap

Source: `logitdiff_pair_heatmap_plotter.py`

What it plots:

- one selected paired-condition case with explicit side-by-side condition labeling

Input style:

- saved comparison JSONL

Core parameter surface:

- `group_id`, `prompt_substring`
- `variant`
- `metric`
- `max_layers`, `layer_selection`
- `max_token_chars`
- `colorscale`, `title`
- condition labels, sequence labels, tokenizer path, and response-text overrides

Import path example:

```python
from logit_diff_lens._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_pair_heatmap_plotter import (
    plot_single_pairwise_condition_heatmap,
)
```

## Live-Run Note For Generation-Only Inputs

The heatmap plotting controls above are about plotting saved results. For live generation workflows, the run that produces the saved payload may additionally depend on:

- template enabled or disabled
- prefix-style prompting
- plain or no-template prompting
- optional system prompt
- temperature and related decoding-condition settings

Those settings affect how the generation payload is produced before plotting from file. They are not a separate plotting vocabulary.

## Output Formats And Interpretation

Common output formats in the public prompt/generation wrappers:

- `html` for interactive Plotly figures
- `pdf` for static exports

Some specialized paper-oriented or legacy families also expose:

- `png`

How to read the result:

- prompt-lens heatmaps show how overlap or verification changes across the fixed prompt positions
- generation-lens heatmaps show where continuation behavior starts to diverge across layers and generation steps
- verification heatmaps emphasize predictor-versus-target agreement rather than only overlap
- LDL and ADL heatmaps emphasize scalar divergence or delta views rather than token-list cells
- paired-condition heatmaps emphasize differences between prompting conditions or template variants

# Heatmaps

![Comparison Artifact](../assests/docs_figures/logit_diff_comparison_artifact_3.png)

## Overview

Heatmaps are one of the main ways to inspect `LogitDiff` results. `LogitDiff` covers both prompt-lens and generation-lens analysis, and the heatmaps show how a metric changes across layers and token positions in either case.

## Prompt lens and generation lens

Heatmaps are relevant to both:

- prompt lens, where the axes usually follow the fixed input prompt
- generation lens, where the axes follow generated continuation behavior

## When to use it

Use heatmaps when you want to:

- see where two systems diverge across layers
- inspect token-level or step-level overlap
- compare metrics such as JSD and top-k Jaccard
- produce an interactive Plotly figure for exploration

## What you give it

- saved prompt-comparison output for prompt heatmaps
- generation-lens output for generation heatmaps
- a chosen metric
- an output path

## What it gives back

It gives you an interactive Plotly heatmap or a static export that you can inspect, share, or include in reports.

## Prompt heatmap example

```bash
PYTHONPATH=src python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/<run-a>.pt \
  --base-artifact tmp/artifacts/<run-b>.pt \
  --comparison-output tmp/artifacts/<comparison-name>.pt \
  --readout-mode model_norm \
  --metric topk_jaccard_ft_base \
  --plot-output tmp/artifacts/<comparison-name>.html
```

## Generation heatmap example

```python
from pathlib import Path

from logit_diff_lens.plotting.logitdiff_gen_plotter import save_logitdiff_heatmap_html

save_logitdiff_heatmap_html(
    "tmp/<generation-run>/<layerwise-json>.json",
    Path("tmp/<generation-run>/figures/<generation-heatmap>.html"),
)
```

## How to interpret the result

Read the heatmap as a layer-by-position or layer-by-step view.

- In prompt lens, strong regions show where the fixed prompt carries the biggest difference.
- In generation lens, strong regions show where the continuation starts to diverge or where top-token overlap weakens over time.
- In template-conditioned generation analysis, strong regions can show where changing the template, prefix, or system prompt begins to alter the continuation trajectory.

Jaccard-style heatmaps are especially useful when you care about overlap in top predictions, while JSD-style heatmaps are more useful when you care about full-distribution divergence.

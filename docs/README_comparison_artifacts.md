# Comparison Artifacts

![Comparison Artifact](../assests/docs_figures/logit_diff_comparison_artifact_3.png)

## Overview

Comparison artifacts are the main way `LogitDiff` turns two saved runs into a user-facing analysis result. They make it easy to compare systems, readouts, or settings and then visualize where they diverge.

## Wrappers and normalization

Wrappers matter during comparison because they keep both sides of the comparison aligned in how tokens, masking, padding, and decoding are handled.

For readouts, the main user-facing choices are:

- `raw`
- `ModelNorm`
- `Tuned Lens`

If two comparisons use different readout modes, that difference should be treated as part of the comparison itself.

## When to use it

Use a comparison artifact when you want to:

- compare two models on the same prompt
- compare two runs over batches or datasets
- compare two readout settings such as `ModelNorm` and `Tuned Lens`
- create a heatmap from saved prompt captures
- find positions or layers with large divergence

## What you give it

- one saved run for system A
- one saved run for system B
- a readout mode
- an output path

## What it gives back

It saves the comparison result and can also export a figure such as a PDF heatmap.

## Example command

```bash
PYTHONPATH=src python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/ft_capture.pt \
  --base-artifact tmp/artifacts/base_capture.pt \
  --comparison-output tmp/artifacts/ft_vs_base_comparison.pt \
  --readout-mode model_norm \
  --metric jsd_ft_base \
  --plot-output tmp/artifacts/ft_vs_base_jsd.pdf
```

## How to interpret the result

The saved comparison tells you where two systems come apart across tokens and layers. In batched or dataset-style analysis, the important point is to read only meaningful token positions and avoid treating masked or padded positions as real evidence.

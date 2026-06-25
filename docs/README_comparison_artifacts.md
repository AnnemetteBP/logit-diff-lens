# Comparison Artifacts

![Comparison Artifact](../assests/docs_figures/logit_diff_comparison_artifact_3.png)

## Overview

Comparison artifacts are the main way `LogitDiff` turns two saved prompt captures into a user-facing analysis result. They make it easy to compare systems, readouts, or settings and then visualize where they diverge.

## When to use it

Use a comparison artifact when you want to:

- compare two models on the same prompt
- compare two readout settings such as `ModelNorm` and `Tuned Lens`
- create a heatmap from saved prompt captures
- find positions or layers with large divergence

## What you give it

- one saved artifact for system A
- one saved artifact for system B
- a readout mode
- an output path

## What it gives back

It saves the comparison result and can also export a figure such as a PDF heatmap.

## Example command

```bash
PYTHONPATH=src /home/am/miniconda3/envs/ldl-env/bin/python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/ft_capture.pt \
  --base-artifact tmp/artifacts/base_capture.pt \
  --comparison-output tmp/artifacts/ft_vs_base_comparison.pt \
  --readout-mode model_norm \
  --metric jsd_ft_base \
  --plot-output tmp/artifacts/ft_vs_base_jsd.pdf
```

## How to interpret the result

The saved comparison tells you where two systems come apart across tokens and layers. The plot is usually the easiest entry point: use it to spot strong divergence, then drill down into the tokens, layers, or lens settings that matter most.

# Forward Capture

![Forward Capture Artifact](../assests/docs_figures/logit_diff_forward_artifact_2.png)

## Overview

Forward capture is the starting point for most `LogitDiff` workflows. It saves the prompt-level activations that later analyses reuse for heatmaps, lens comparisons, patchscopes, and other views.

## When to use it

Use forward capture when you want to:

- inspect how a model evolves across layers for one prompt
- save activations once and analyze them later
- compare different lenses on the same prompt
- prepare inputs for comparison or patchscope analysis

## What you give it

- a model name
- a prompt
- an output path
- optional dtype and capture settings

## What it gives back

It saves a prompt artifact containing the token sequence and the layer-by-layer activations needed for later analysis.

## Example command

```bash
PYTHONPATH=src /home/am/miniconda3/envs/ldl-env/bin/python pipelines/capture_prompt_artifacts.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --prompt "If I had more time, I would travel more often." \
  --output-path tmp/artifacts/pythia70m_prompt_capture.pt \
  --dtype bfloat16 \
  --force-include-output
```

## How to interpret the result

Think of the saved artifact as the base record for one prompt. You normally do not read it directly; instead, you reuse it for plots and follow-up analyses so every later result is grounded in the same captured run.

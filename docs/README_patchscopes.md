# Patchscopes

![Patchscope Workflow](../assests/docs_figures/logit_diff_patchscope_5.png)

## Overview

Patchscopes let you take a representation from one prompt or system and insert it into another prompt run. In `LogitDiff`, this is useful for following up on an interesting divergence and testing whether a specific layer-position representation changes the target readout.

## When to use it

Use patchscopes when you want to:

- test whether one token representation drives a change downstream
- follow up on a strong difference from a heatmap
- compare how two systems react to the same inserted representation
- explore causal intervention rather than just observation
- follow up on batched or dataset-level findings with targeted single examples

## What you give it

- a saved source artifact
- a target prompt
- a source layer and token position
- a target layer and token position
- an output path

## What it gives back

It saves a patched run showing how the target prompt behaves after the chosen representation is inserted.

## Example command

```bash
PYTHONPATH=src python pipelines/run_patchscope_prompt.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --source-artifact tmp/artifacts/pythia70m_prompt_capture.pt \
  --target-prompt "If I had more time, I would travel more often." \
  --source-layer-index 2 \
  --source-position 4 \
  --target-layer-index 2 \
  --target-position 4 \
  --readout-mode model_norm \
  --output-path tmp/artifacts/pythia70m_patchscope.pt
```

## How to interpret the result

Read the patched result as an intervention test. If the target prediction shifts in a meaningful way, the inserted representation is likely carrying information that matters for the target prompt at that location.

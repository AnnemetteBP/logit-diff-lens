# Backward Artifacts

![Backward Artifact](../assests/docs_figures/logit_diff_backward_artifact_6.png)

## Overview

Backward artifacts capture how a chosen target token sends signal backward through the model. This gives a different view from forward lens analysis and is useful when you want to study target-conditioned behavior rather than only layerwise predictions.

## When to use it

Use backward artifacts when you want to:

- inspect target-conditioned signals for one prompt
- compare forward and backward views of the same example
- analyze which layers matter for a chosen target token
- explore attribution-style analysis inside `LogitDiff`

## What you give it

- a model name
- a prompt
- a target token
- an output path

## What it gives back

It saves a backward analysis result tied to that prompt and chosen target token.

## Example command

```bash
PYTHONPATH=src python pipelines/capture_backward_artifact.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --prompt "Paris is the capital of" \
  --target-token-text " France" \
  --output-path tmp/artifacts/paris_backward.pt \
  --dtype bfloat16
```

## How to interpret the result

Read the saved output as a target-conditioned view of the prompt. It tells you how the chosen token depends on internal states, which can complement what you already saw from forward captures, comparisons, or prism-style views.

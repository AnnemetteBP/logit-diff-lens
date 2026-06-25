# Backward Artifacts

![Backward Artifact](../assests/docs_figures/logit_diff_backward_artifact_6.png)

## Purpose

Backward artifacts are the target-conditioned analysis family in `LogitDiff`.

Instead of asking what a hidden state predicts under a forward readout, this family asks how a chosen target token or loss sends signal backward through the model.

## Core idea

The workflow is:

1. run a forward pass on a prompt
2. choose a target token or loss
3. run one backward pass
4. save the resulting backward signals

This is a separate artifact family from ordinary forward capture and should be treated as such in both code and documentation.

## What it stores

A backward artifact can store:

- the prompt and tokenization context
- the chosen target token
- target-conditioned backward signals
- layer-level backward records
- optional subblock VJPs
- backend metadata

## Why it matters

Forward readouts tell you what a layer seems to be representing.

Backward artifacts instead tell you how the chosen target objective depends on internal states and directions.

That makes them useful for:

- target-conditioned attribution
- gradient-based interpretability
- comparison with forward lens interpretations
- future backward-differential methods

## Pipeline entry points

- [src/logit_diff_lens/logit_lens/backward.py](/media/am/AM/logit-diff-lens/src/logit_diff_lens/logit_lens/backward.py)
- [pipelines/capture_backward_artifact.py](/media/am/AM/logit-diff-lens/pipelines/capture_backward_artifact.py)
- [pipelines/pythia/capture_backward_artifact.py](/media/am/AM/logit-diff-lens/pipelines/pythia/capture_backward_artifact.py)

## Example usage

```bash
PYTHONPATH=src /home/am/miniconda3/envs/ldl-env/bin/python pipelines/capture_backward_artifact.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --prompt "Paris is the capital of" \
  --target-token-text " France" \
  --output-path tmp/artifacts/paris_backward.pt \
  --dtype bfloat16
```

## Relation to the rest of LogitDiff

Backward artifacts are not a replacement for forward artifacts.

They complement them by adding a target-conditioned view that can later be compared against:

- forward decoded predictions
- prism decompositions
- patchscope outcomes
- future gradient-aligned differential analyses

## Planned extensions

- richer backward artifact summaries
- backward-differential comparisons
- generation-conditioned backward capture
- stronger coupling with intervention workflows

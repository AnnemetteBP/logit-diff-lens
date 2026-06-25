# Forward Capture Artifacts

![Forward Capture Artifact](../assests/docs_figures/logit_diff_forward_artifact_2.png)

## Purpose

The forward capture artifact is the shared source of truth for `LogitDiff`.

It stores the prompt-aligned hidden-state data that can later be reused across comparison heatmaps, `ModelNorm` or `Tuned Lens` readouts, Logit Prisms, patchscopes, hidden-delta analyses, and future generation-aligned workflows.

## What it contains

The canonical forward artifact stores:

- token IDs
- decoded token text
- per-layer residual hidden states
- optional subblock outputs such as attention and MLP contributions
- prompt metadata
- backend and runtime metadata

In the current codebase, this is represented by `PromptDecodeArtifact`.

## Why it matters

The main design rule is:

- capture once
- reuse many times

That makes the project more reproducible because multiple reported analyses can be derived from the exact same saved hidden-state capture instead of rerunning slightly different forwards for each figure.

## Inputs and outputs

### Inputs

- model or wrapped backend
- tokenizer
- prompt text or dataset records
- readout settings

### Output

- one saved prompt artifact
- or one saved prompt-artifact bundle for many prompts

## Relation to the rest of LogitDiff

The forward artifact sits upstream of:

- comparison artifacts
- patchscope artifacts
- backward artifact targeting setup
- prism decomposition
- hidden-delta analyses
- decoded lens comparisons

## Pipeline entry points

The main entry points are:

- [src/logit_diff_lens/logit_lens/capture.py](/media/am/AM/logit-diff-lens/src/logit_diff_lens/logit_lens/capture.py)
- [pipelines/capture_prompt_artifacts.py](/media/am/AM/logit-diff-lens/pipelines/capture_prompt_artifacts.py)
- [pipelines/pythia/capture_prompt_artifacts.py](/media/am/AM/logit-diff-lens/pipelines/pythia/capture_prompt_artifacts.py)

## Example usage

```bash
PYTHONPATH=src /home/am/miniconda3/envs/ldl-env/bin/python pipelines/capture_prompt_artifacts.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --prompt "If I had more time, I would travel more often." \
  --output-path tmp/artifacts/pythia70m_prompt_capture.pt \
  --dtype bfloat16 \
  --force-include-output
```

## Planned extensions

- richer generation artifact capture
- broader subblock capture coverage
- MoE-aware capture support
- broader backend and quantization compatibility

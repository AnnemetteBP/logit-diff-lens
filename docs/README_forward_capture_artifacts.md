# Forward Capture

![Forward Capture Artifact](../assests/docs_figures/logit_diff_forward_artifact_2.png)

## Overview

Forward capture is one starting point for `LogitDiff` workflows. It saves the activations from a prompt, batch, or dataset-style run so they can be used for heatmaps, lens comparisons, patchscopes, and other views. Generation-lens workflows use the same general ideas, but along continuations rather than only the initial prompt.

## Wrappers and normalization

The wrapper layer is important here because it keeps tokenization, masking, special-token handling, device movement, and readout behavior consistent across models.

The most common readout choices are:

- `raw`, which decodes directly from the hidden state
- `ModelNorm`, which applies the model's final normalization before decoding
- `Tuned Lens`, which uses a learned readout

The prompt capture path currently preserves what is needed for both `raw` and `ModelNorm`, so those readout views can be compared later without rerunning the capture.

## When to use it

Use forward capture when you want to:

- inspect how a model evolves across layers for one prompt
- run the same analysis across many prompts
- prepare batched or dataset-style captures
- save activations once and analyze them later
- compare different lenses on the same prompt
- prepare inputs for comparison or patchscope analysis

## What you give it

- a model name
- a prompt
- or a prompt set / dataset slice
- an output path
- optional dtype and capture settings

## What it gives back

It saves the token sequence and the layer-by-layer activations needed for later analysis, while allowing later steps to ignore meaningless padded positions and respect the original masking.

## Example command

```bash
PYTHONPATH=src python pipelines/capture_prompt_artifacts.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<run-name>.pt \
  --dtype <dtype> \
  --force-include-output
```

## Generation-lens context

```bash
PYTHONPATH=src python pipelines/<pipeline-group>/run_gen_lens.py \
  --config configs/<group>/gen_lens/<config-name>.json
```

## How to interpret the result

Think of the saved result as the base record for a run. You normally do not read it directly; instead, you reuse it for plots and follow-up analyses while keeping token positions, masking, padding behavior, and readout choices aligned with the original input.

# Patchscopes in LogitDiff

![Patchscope Workflow](../assests/docs_figures/logit_diff_patchscope_5.png)

## Purpose

This document defines how patchscopes fit into `LogitDiff`, what is implemented now, and how the patchscope path should extend later without breaking the shared artifact-first design.

Patchscopes are a first-class analysis family in this repository, not an ad hoc side script.

## What a patchscope is in this repository

A patchscope takes a representation from a saved source artifact and injects it into a target run.

At minimum this specifies:

- a source prompt or source artifact
- a source layer and source token position
- a target prompt
- a target layer and target token position
- a mapping from source representation space to target representation space
- a readout used to interpret the patched result

The main goal is not just intervention for its own sake. In `LogitDiff`, patchscopes are meant to support:

- causal follow-up analysis after differential heatmaps
- targeted investigation of high-difference tokens or layers
- quantized vs non-quantized comparison studies
- prompt-lens and generation-lens intervention workflows
- future sweep summaries over many positions, layers, or token choices

## Current implementation

The current implemented path is a prompt-first identity patchscope.

The code path is:

- [src/logit_diff_lens/collectors/patchscope.py](/media/am/AM/logit-diff-lens/src/logit_diff_lens/collectors/patchscope.py)
- [src/logit_diff_lens/logit_lens/patchscope.py](/media/am/AM/logit-diff-lens/src/logit_diff_lens/logit_lens/patchscope.py)
- [pipelines/run_patchscope_prompt.py](/media/am/AM/logit-diff-lens/pipelines/run_patchscope_prompt.py)

### Implemented contract

The current primitive does the following:

1. Load a saved `PromptDecodeArtifact`.
2. Read one source hidden state from `source_layer_index` and `source_position`.
3. Run a target prompt.
4. Patch that source hidden state into `target_layer_index` and `target_position`.
5. Decode the patched target activations with either `raw` or `model_norm` readout.
6. Save a validated `PatchscopePromptArtifact`.

### Current limitations

The current path is intentionally narrow:

- prompt-first only
- identity mapping only
- one source position at a time
- one target position at a time
- one patched run per artifact

This is the correct primitive to stabilize first because it preserves reproducibility and keeps the intervention fully tied to a saved forward artifact.

## Why patchscopes belong in the artifact system

Patchscopes should reuse the same source hidden states as the rest of the analysis stack.

That means:

- the same saved forward artifact can support heatmaps, Logit Prisms, hidden-delta analysis, and patchscopes
- reported patchscope results can be traced back to the exact source capture used elsewhere in a paper
- source representations do not need to be re-extracted in inconsistent ad hoc scripts

This is especially important when comparing:

- System A vs System B
- base vs finetuned
- ModelNorm vs Tuned Lens
- quantized vs non-quantized variants
- different chat templates or generation settings

## Recommended interpretation workflow

The patchscope path is meant to be downstream of earlier analyses.

Typical flow:

1. Capture forward artifacts.
2. Compute a comparison artifact.
3. Use metrics such as JSD, KL, rank delta, or top-k overlap to identify interesting layers and positions.
4. Select a source token or source layer with high divergence.
5. Run patchscope interventions across one or more target positions.
6. Aggregate or visualize the intervention results.

This keeps patchscopes grounded in measured divergence rather than arbitrary intervention choices.

## Relation to prompt lens and generation lens

Patchscopes should support both major workflow families.

### Prompt-lens patchscopes

This is the currently implemented family.

Use cases include:

- patching a representation from one system into another system's prompt processing
- probing whether a divergence at one layer can reappear downstream
- testing whether a high-difference token representation drives a change in the next-token distribution

### Generation-lens patchscopes

This is not implemented yet as a first-class artifact path, but it should be added later.

Important future use cases include:

- patching a chosen token representation across generation steps
- comparing quantized vs non-quantized generation trajectories
- testing whether a source representation causes a downstream generation reversion or shift

## Planned extension: patchscope sweeps

The next patchscope layer should not be a pile of one-off scripts. It should be a proper sweep artifact family.

Recommended future artifact families:

- `PatchscopeSweepArtifact`
- `PatchscopePositionSweepArtifact`
- `PatchscopeLayerSweepArtifact`

These should support systematic sweeps over:

- source positions
- target positions
- source layers
- target layers
- selected token sets
- selected prompts

This is the path that fits the earlier patchscope experiments where one high-difference token is chosen and then patched across many positions and layers.

## Metrics and outputs

A patchscope artifact should store the patched decoded result directly, but downstream summaries should also compute comparison-facing metrics.

Useful sweep metrics include:

- target-token probability
- target-token rank
- logit difference against a reference token
- top-k token overlap
- JSD against an unpatched target run
- first-layer or first-position reversion statistics

For `LogitDiff`, these metrics should be expressed in generic comparison language rather than only base-vs-finetuned language.

## Mapping functions

The current mapping is `identity`.

Later mappings may include:

- learned linear maps
- subspace-restricted maps
- projection maps
- cross-model alignment maps

Those should be introduced only when their provenance is explicit and reproducible.

## Relation to quantization studies

Patchscopes are especially useful for quantization studies because they let us separate:

- where a representation diverges
- where that divergence becomes causally important
- whether a patched source representation restores or disrupts a target readout

That makes patchscopes a strong follow-up method for quantized-vs-non-quantized comparison artifacts.

## Design rule

Patchscope code should stay inside the single `LogitDiff` package and reuse the same wrapper and artifact contracts as the rest of the framework.

It should not drift into isolated model-specific scripts unless those scripts are only thin pipeline entry points over reusable package code.

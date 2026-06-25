# Generation Lens

![LogitDiff Overview](../assests/docs_figures/logit_diff_framework_overview_1.png)

## Overview

The generation lens is still a logit-lens method. The difference is that it follows model behavior during actual continuation, not just on the fixed prompt. This is useful when you want to see how divergence unfolds step by step as text is generated.

## Wrappers and normalization

The generation path depends on wrappers for consistent generation behavior, token handling, and readout choices across models.

The main readout choices users should care about are:

- `raw`
- `ModelNorm`
- `Tuned Lens`

If you compare generation results across readout modes, treat that as part of the analysis rather than a minor detail.

## When to use it

Use the generation lens when you want to:

- compare two models during continuation rather than only on the prompt
- compare the same model under different decoding conditions
- track how divergence changes over generated steps
- study generation-level Jaccard overlap or other token-distribution comparisons
- compare prompt behavior with realized continuation behavior

Typical same-model comparisons include:

- different chat templates
- different temperatures
- different sampling strategies
- other decoding or prompting conditions

## What you give it

- a generation-lens config file
- a base model
- a comparison model, adapter, or alternate condition
- a prompt source such as a dataset or prompt list
- generation settings such as token budget, top-k, and prompt formatting

## What it gives back

It produces saved generation-lens outputs that can be summarized across steps, layers, prompts, and comparison metrics.

## Example command

```bash
python pipelines/<pipeline-group>/run_gen_lens.py \
  --config configs/<group>/gen_lens/<config-name>.json
```

## How to interpret the result

Read generation-lens output as a trajectory view. Instead of only asking what the models predict before generation starts, ask where their continuations begin to separate, how stable that difference is over time, and whether top-token overlap stays high or falls apart during continuation. This applies both to two-model comparisons and to same-model comparisons under different generation conditions.

For explicit prompt-heatmap and generation-heatmap plotting paths, see [README_heatmaps.md](/media/am/AM/logit-diff-lens/docs/README_heatmaps.md).

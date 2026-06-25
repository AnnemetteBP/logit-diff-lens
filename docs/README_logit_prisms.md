# Logit Prisms

![Logit Prisms](../assests/docs_figures/logit_diff_logit_prisms_4.png)

## Overview

Logit Prisms help break a prediction or a divergence into interpretable pieces. Instead of only showing that two systems differ, they help show whether the difference looks most visible in embeddings, attention, MLP updates, or the full residual stream. This idea is useful for both prompt-lens and generation-lens analysis.

## When to use it

Use Logit Prisms when you want to:

- localize where a difference seems to arise
- compare attention-heavy and MLP-heavy behavior
- inspect component-level behavior across layers
- decide where a patchscope intervention may be most informative
- compare where prompt-time and generation-time differences appear to come from

## What you give it

- saved prompt captures or generation-side outputs
- optionally a comparison result
- a chosen component view such as embedding, attention, MLP, or full stream

## What it gives back

It gives you component-level views that help explain where a prediction or divergence is showing up.

## Wrappers and normalization

Prism views are easiest to compare when the wrapper layer is handling tokenization, masking, and readout settings consistently.

If you switch between `raw`, `ModelNorm`, and `Tuned Lens`, treat that as part of the interpretation rather than as a hidden technical detail.

## Example command

```bash
PYTHONPATH=src python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/<run-a>.pt \
  --base-artifact tmp/artifacts/<run-b>.pt \
  --comparison-output tmp/artifacts/<comparison-name>.pt \
  --readout-mode model_norm \
  --metric topk_jaccard_ft_base \
  --plot-output tmp/artifacts/<comparison-name>.html
```

## How to interpret the result

Use prism views as localization tools. If one component lights up far more than the others, that is often a strong clue about where to look next and what kind of intervention or follow-up analysis makes sense.

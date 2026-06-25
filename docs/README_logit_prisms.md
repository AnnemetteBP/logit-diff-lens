# Logit Prisms

![Logit Prisms](../assests/docs_figures/logit_diff_logit_prisms_4.png)

## Purpose

Logit Prisms in `LogitDiff` decompose where a decoded prediction or divergence appears to come from inside the model.

The main idea is to move beyond only asking whether two systems differ and instead ask which subblocks or residual contributions appear to drive that difference.

## Main decomposition view

The intended prism-style breakdown includes components such as:

- embedding contribution
- attention contribution
- MLP contribution
- full residual stream

This can be studied in:

- hidden space
- decoded vocab/logit space
- differential `A - B` form

## Why this is useful

Prisms are the natural localization layer between broad heatmaps and more causal interventions.

They help answer questions like:

- does the difference mostly emerge in attention or MLP?
- is a divergence already present in the embedding contribution?
- does a late layer simply amplify an earlier difference?

## Inputs and outputs

### Inputs

- canonical forward artifacts
- optional subblock outputs
- optionally comparison artifacts or hidden deltas

### Outputs

- component-wise contribution summaries
- prism heatmaps
- prism differential views
- downstream token- or layer-level localization targets

## Relation to the rest of LogitDiff

Logit Prisms sit between:

- forward capture
- comparison artifacts
- patchscope follow-up analysis
- weight and vocabulary-space interpretation

They are especially useful for deciding where to intervene or which component families deserve deeper study.

## Current implementation direction

This method family depends strongly on reusable subblock-aware capture and consistent operand ordering.

The main public design requirements are:

- use the same saved hidden states as other analyses whenever possible
- preserve canonical `comparison - reference` ordering
- keep component definitions stable across methods

## Planned extensions

- formal prism artifacts
- component-level sweep summaries
- direct integration with patchscope target selection
- richer support for MLP subblocks such as up, gate, and down projections

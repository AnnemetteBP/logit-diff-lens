# LogitDiff Documentation

![LogitDiff Overview](../assests/docs_figures/logit_diff_framework_overview_1.png)

These guides explain the main `LogitDiff` workflows for readers, users, and paper visitors who want to understand the methods and try them on prompts, batches, datasets, or generation runs.

## Getting Started

- [Forward Capture](README_forward_capture_artifacts.md)
  Learn how to save prompt activations for single prompts, batches, or dataset-style runs.
- [Comparison Artifacts](README_comparison_artifacts.md)
  Compare two systems and turn the result into heatmaps and token-level analysis.

## Analysis Guides

- [Patchscopes](README_patchscopes.md)
  Explore how a chosen representation changes a target prompt when it is patched in.
- [Logit Prisms](README_logit_prisms.md)
  Break a prediction or divergence into embedding, attention, MLP, and full-stream views.
- [Backward Artifacts](README_backward_artifacts.md)
  Inspect target-conditioned backward signals for a chosen prompt and token.
- [Weight and Vocabulary Methods](README_model_weight_vocab_methods.md)
  Study interpretable directions in weights, logits, and vocabulary projections.

## Important note

The single-prompt examples in these guides are only short demonstrations. `LogitDiff` is also meant for batched runs, dataset-style analysis, generation workflows, and padding-aware token handling.

## Main Project Entry

- [Repository README](../README.md)
  Start here for the project overview, installation, and quickstart commands.

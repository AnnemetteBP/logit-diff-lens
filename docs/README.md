# LogitDiff Documentation

![LogitDiff Overview](../assests/docs_figures/logit_diff_framework_overview_1.png)

These guides explain the main `LogitDiff` workflows for readers, users, and paper visitors who want to understand the methods and try them on prompts, batches, datasets, prompt-lens runs, or generation-lens runs.

## Getting Started

- [Forward Capture](README_forward_capture_artifacts.md)
  Learn how to save prompt activations for single prompts, batches, or dataset-style runs.
- [Comparison Artifacts](README_comparison_artifacts.md)
  Compare two systems and turn the result into heatmaps and token-level analysis for prompt or generation studies.
- [Wrappers and Readouts](README_wrappers.md)
  See how prompt, generation, custom-generation, and patching workflows share one wrapper layer.
- [Generation Lens](README_generation_lens.md)
  Follow divergence during actual continuation as a generation-time logit-lens analysis, including template and no-template comparisons.
- [Heatmaps](README_heatmaps.md)
  Plot prompt-lens and generation-lens heatmaps for metrics such as JSD and top-k Jaccard.

## Analysis Guides

- [Patchscopes](README_patchscopes.md)
  Explore how a chosen representation changes a target run when it is patched in.
- [Logit Prisms](README_logit_prisms.md)
  Break a prediction or divergence into embedding, attention, MLP, and full-stream views across prompt or generation analysis.
- [Backward Artifacts](README_backward_artifacts.md)
  Inspect target-conditioned backward signals for a chosen prompt and token.
- [Weight and Vocabulary Methods](README_model_weight_vocab_methods.md)
  Study interpretable directions in weights, logits, and vocabulary projections.

## Important note

The single-prompt examples in these guides are only short demonstrations. `LogitDiff` is also meant for batched runs, dataset-style analysis, generation workflows, and padding-aware token handling.

## Main Project Entry

- [Repository README](../README.md)
  Start here for the project overview, installation, and quickstart commands.

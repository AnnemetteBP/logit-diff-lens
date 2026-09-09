# LogitDiff Documentation

![LogitDiff Overview](../assests/docs_figures/logit_diff_framework_overview_1.png)

These guides cover the public `LogitDiff` toolkit surface for prompt capture, generation lens, heatmaps, interventions, and follow-up analysis.

## Core Guides

- [Wrappers and Readouts](README_wrappers.md)
  Core wrapper types and the shared behavior behind prompt lens, generation lens, and patching.
- [Forward Capture](README_forward_capture_artifacts.md)
  Prompt-side artifact capture for single prompts and datasets.
- [Lens Workflows](README_lens_workflows.md)
  Shared guide for the prompt-lens and generation-lens modes inside the main LogitDiff toolkit.
- [Heatmaps](README_heatmaps.md)
  Prompt LogitDiff, generation LogitDiff, single-model, and ADL heatmaps.

## Analysis Guides

- [Comparison Artifacts](README_comparison_artifacts.md)
  Canonical prompt-side comparison artifacts and comparison-metric plotting.
- [Null-Calibrated Similarity](README_similarity.md)
  Robust alignment analysis for saved prompt and generation artifacts.
- [Patchscopes](README_patchscopes.md)
  Prompt and generation intervention workflows.
- [Logit Prisms](README_logit_prisms.md)
  Component-localization views across prompt and generation analysis.
- [Backward Artifacts](README_backward_artifacts.md)
  Target-conditioned backward-pass capture.
- [Weight and Vocabulary Methods](README_model_weight_vocab_methods.md)
  Vocabulary-space and weight-space follow-up analysis.

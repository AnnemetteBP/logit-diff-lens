logit_lens_methods structure

Use these folders for the current work.

Generation
- `base_collector_scripts/generation/`
- `wrapper/lens_wrappers/custom_generation_lens_wrapper.py`
- `plotting/heatmaps/logitdiff_gen_plotter.py`

Teacher-forcing / prompt LogitLens / LDL
- `logitdiff_ldl/collect_logits_prompt.py`
- `plotting/heatmaps/logit_lens_heatmap_metrics.py`
- `plotting/heatmaps/ldl_heatmap_metrics.py`
- `logitdiff_ldl/activation_dataset_analysis.py`
- `plotting/heatmaps/logitdiff_tf_plotter.py`

Pairwise comparison heatmap plotting
- `plotting/heatmaps/logitdiff_pair_heatmap_plotter.py`

ADL
- `logitdiff_adl/`
- `plotting/heatmaps/adl_plotter.py`

Shared plotting entrypoints
- `plotting/heatmaps/`

Do not use for the current Qwen risky work
- `src/diffing/methods/logitdiff/`

Notes
- `activation_collector.py` at the package root is only a thin import alias.
- the real collector implementation is under `base_collector_scripts/activation_collector.py`

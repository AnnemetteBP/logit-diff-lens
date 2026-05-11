Active files for the current Qwen risky work

Generation collection and comparison
- `base_collector_scripts/generation/collect_generation_activations_batched.py`
- `wrapper/lens_wrappers/custom_generation_lens_wrapper.py`

Teacher-forcing / prompt collection and comparison
- `logitdiff_ldl/collect_logits_prompt.py`
- `plotting/heatmaps/logit_lens_heatmap_metrics.py`
- `plotting/heatmaps/ldl_heatmap_metrics.py`
- `logitdiff_ldl/activation_dataset_analysis.py`

Plotting files for current work
- `plotting/heatmaps/logitdiff_gen_plotter.py`
- `plotting/heatmaps/logitdiff_tf_plotter.py`
- `plotting/heatmaps/logitdiff_pair_heatmap_plotter.py`
- `plotting/heatmaps/logit_lens_plotter.py`
- `plotting/heatmaps/ldl_plotter.py`
- `plotting/heatmaps/adl_plotter.py`

Not part of the current generation or teacher-forcing pipelines
- `../methods/logitdiff/*`
- `base_collector_scripts/teacher_forcing/` removed because it was empty

Current known source-data failure
- `tmp/qwen_risky/generation_lens/base_hidden_logits/neutral_chat_template_60/base_response/` breaks starting at batch `012`
- `tmp/qwen_risky/generation_lens/base_hidden_logits/neutral_chat_template_60/finetuned_response/` breaks starting at batch `041`

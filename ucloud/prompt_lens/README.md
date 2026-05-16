UCloud prompt-lens entrypoint for the Qwen Natural Questions 500-sample hidden-only run.

Exact saved run plan:
- `ucloud/prompt_lens/PLAN.md`

What is generic and reusable:
- `src/diffing/logit_lens_methods/pipelines/run_hidden_only_prompt_lens_pipeline.py`
- `src/diffing/logit_lens_methods/pipelines/run_hidden_only_prompt_lens_bundle.py`

Dataset-build helper for this analysis pipeline:
- `ucloud/pipelines/prompt_lens/build_query_only_dataset.py`

What is model-specific and UCloud-only:
- `ucloud/prompt_lens/configs/risky.json`
- `ucloud/prompt_lens/configs/medical.json`
- `ucloud/prompt_lens/configs/sports.json`

One-command run:
- `bash ucloud/prompt_lens/run_qwen_nq500.sh`

What this does:
1. builds `tmp/em_qwen/datasets/nq_train_500_query_only.jsonl`
2. collects the shared base hidden-state payload once
3. reuses that base payload for risky / medical / sports
4. collects one comparison hidden-state payload per model
5. runs downstream summaries and correlations from hidden states
6. does not persist full logits payloads

Saved:
- `tmp/em_qwen/prompt_lens_nq_500/shared/base_qwen_prompt_lens.pt`
- one comparison `.pt` hidden-state payload per model
- `summaries/mode_specific_summary.json`
- `summaries/layerwise_observations.jsonl`
- `summaries/layerwise_statistics.json`
- `summaries/layerwise_correlations.json`
- figures and `run_manifest.json`

Not saved:
- full persisted logits tensors
- full persisted probability tensors
- duplicate base payloads per comparison

Shared base payload:
- `tmp/em_qwen/prompt_lens_nq_500/shared/base_qwen_prompt_lens.pt`

Outputs:
- `tmp/em_qwen/prompt_lens_nq_500/risky`
- `tmp/em_qwen/prompt_lens_nq_500/medical`
- `tmp/em_qwen/prompt_lens_nq_500/sports`

Use these configs for the Natural Questions teacher-forcing prompt-lens control run.

Expected dataset file:
- `tmp/em_qwen/datasets/nq_train_500_query_only.jsonl`

Behavior:
- reuses one shared base payload at:
  - `tmp/em_qwen/prompt_lens_nq_500/shared/base_qwen_prompt_lens.pt`
- saves hidden-state payloads without persisted logits:
  - `"save_logits": false`
- runs on GPU by default:
  - `"force_cpu": false`

Run examples:
- `PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.run_prompt_lens_pipeline --config configs/em_qwen/prompt_lens/nq_500/risky.json`
- `PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.run_prompt_lens_pipeline --config configs/em_qwen/prompt_lens/nq_500/medical.json`
- `PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.run_prompt_lens_pipeline --config configs/em_qwen/prompt_lens/nq_500/sports.json`

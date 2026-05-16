Use these configs only for the UCloud hidden-only Natural Questions teacher-forcing prompt-lens control run.

Expected dataset file:
- `tmp/em_qwen/datasets/nq_train_500_query_only.jsonl`

Behavior:
- reuses one shared base payload at:
  - `tmp/em_qwen/prompt_lens_nq_500/shared/base_qwen_prompt_lens.pt`
- saves hidden-state payloads without persisted logits:
  - `"save_logits": false`
- runs on GPU:
  - `"force_cpu": false`
- keeps the local default prompt-lens pipeline untouched

Single entrypoint:
- `PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.run_qwen_nq500_ucloud_bundle`

Manual entrypoints:
- build dataset:
  - `PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.build_nq_query_only_dataset --output-path tmp/em_qwen/datasets/nq_train_500_query_only.jsonl --sample-count 500 --split train --seed 13`
- run one comparison:
  - `PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.run_prompt_lens_ucloud_hidden_only_pipeline --config configs/em_qwen/prompt_lens/nq_500/risky.json`
  - `PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.run_prompt_lens_ucloud_hidden_only_pipeline --config configs/em_qwen/prompt_lens/nq_500/medical.json`
  - `PYTHONPATH=src python3 -m diffing.logit_lens_methods.pipelines.run_prompt_lens_ucloud_hidden_only_pipeline --config configs/em_qwen/prompt_lens/nq_500/sports.json`

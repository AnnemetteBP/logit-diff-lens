EM Qwen reproducibility pipelines.

This folder is intentionally minimal:
- `run_prompt_lens.py`
- `run_gen_lens.py`

Configs choose scenario, template, and horizon:
- `configs/em_qwen/prompt_lens/*.json`
- `configs/em_qwen/gen_lens/**/*.json`

Examples:

```bash
PYTHONPATH=src .venv-logitdiff-lens/bin/python \
  pipelines/em_qwen/run_prompt_lens.py \
  --config configs/em_qwen/prompt_lens/risky.json
```

```bash
PYTHONPATH=src .venv-logitdiff-lens/bin/python \
  pipelines/em_qwen/run_gen_lens.py \
  --config configs/em_qwen/gen_lens/chat_template/risky_14.json
```

Reusable package code lives under `src/diffing/logit_lens_methods/`.

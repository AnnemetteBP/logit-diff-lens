Quant Llama reproducibility pipelines.

This folder is intentionally minimal:
- `run_prompt_lens.py`
- `run_gen_lens.py`

Configs choose variant and horizon:
- `configs/quant_llama/prompt_lens/*.json`
- `configs/quant_llama/gen_lens/*.json`

Examples:

```bash
PYTHONPATH=src .venv-logitdiff-lens/bin/python \
  pipelines/quant_llama/run_prompt_lens.py \
  --config configs/quant_llama/prompt_lens/hf1bit.json
```

```bash
PYTHONPATH=src .venv-logitdiff-lens/bin/python \
  pipelines/quant_llama/run_gen_lens.py \
  --config configs/quant_llama/gen_lens/hf1bit_14.json
```

Reusable package code lives under `src/diffing/logit_lens_methods/`.

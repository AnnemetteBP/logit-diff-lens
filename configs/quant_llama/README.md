# Quant Llama configs

Primary runners:

```bash
PYTHONPATH=src .venv-logitdiff-lens/bin/python \
  src/diffing/logit_lens_methods/pipelines/run_prompt_lens_pipeline.py \
  --config <CONFIG_JSON>
```

```bash
PYTHONPATH=src .venv-logitdiff-lens/bin/python \
  src/diffing/logit_lens_methods/pipelines/run_gen_lens_pipeline.py \
  --config <CONFIG_JSON>
```

Structure:

- `prompt_lens/base.json`
- `prompt_lens/hf1bit.json`
- `gen_lens/hf1bit_14.json`
- `gen_lens/hf1bit_64.json`

Meaning:

- `base.json` = base Llama prompt-lens collection
- `hf1bit.json` = base-vs-HF1Bit prompt-lens comparison
- `hf1bit_14.json` = short-horizon generation-lens comparison
- `hf1bit_64.json` = longer response-collection comparison

References may be either:

- local filesystem paths
- Hugging Face model / adapter IDs

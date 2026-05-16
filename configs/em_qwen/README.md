# EM Qwen configs

Primary runners:

- Prompt lens:

```bash
PYTHONPATH=src .venv-logitdiff-lens/bin/python \
  src/diffing/logit_lens_methods/pipelines/run_prompt_lens_pipeline.py \
  --config <CONFIG_JSON>
```

- Generation lens:

```bash
PYTHONPATH=src .venv-logitdiff-lens/bin/python \
  src/diffing/logit_lens_methods/pipelines/run_gen_lens_pipeline.py \
  --config <CONFIG_JSON>
```

Structure:

- `prompt_lens/`
  - `risky.json`
  - `medical.json`
  - `sports.json`

- `gen_lens/chat_template/`
  - `risky_14.json`, `risky_64.json`
  - `medical_14.json`, `medical_64.json`
  - `sports_14.json`, `sports_64.json`

- `gen_lens/neutral_chat_template/`
  - `risky_14.json`, `risky_64.json`

- `gen_lens/no_template/`
  - `risky_14.json`, `risky_64.json`

Meaning:

- `prompt_lens/*` = prompt-only Logit Lens runs
- `gen_lens/**/_14.json` = short-horizon comparative generation-lens runs
- `gen_lens/**/_64.json` = longer response-collection runs

References may be either:

- local filesystem paths
- Hugging Face model / adapter IDs

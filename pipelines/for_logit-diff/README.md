# `for_logit-diff`

Standalone mini-package for anonymous LogitDiff notebook work.

This folder is intentionally copied out from the main project so it can be:

- submitted alongside a paper bundle,
- shared anonymously on GitHub,
- used in a notebook without importing `src/logit_diff_lens/...`.

## What is included

- local prompt wrapper (`LogitLensWrapper`)
- local generation wrapper using `model.generate(...)` (`GenerateLensWrapper`)
- local prompt heatmap plotter
- local generation heatmap plotter
- local single-model logit-lens heatmap plotter
- Plotly PDF/HTML export helper
- minimal notebook widget helpers

## Folder layout

```text
for_logit-diff/
  pyproject.toml
  README.md
  logitdiff_widget/
    wrappers/
    collectors/
    plotting/
    schemas/
    widget.py
```

## Quick start

```python
from logitdiff_widget import (
    load_model_and_tokenizer,
    build_single_model_widget,
    build_prompt_logitdiff_widget,
    build_generation_logitdiff_widget,
)

model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-3.2-1B")
build_single_model_widget(model, tokenizer)

comparison_model, comparison_tokenizer = load_model_and_tokenizer("meta-llama/Llama-3.2-1B")
build_prompt_logitdiff_widget(model, tokenizer, comparison_model, comparison_tokenizer)
build_generation_logitdiff_widget(model, tokenizer, comparison_model, comparison_tokenizer)
```

For saved prompt-side diff payloads:

```python
from logitdiff_widget import build_prompt_payload_widget

build_prompt_payload_widget("path/to/prompt_payload.json")
```

For saved generation-side diff payloads:

```python
from logitdiff_widget import build_generation_payload_widget

build_generation_payload_widget("path/to/generation_payload.json")
```

There is also a ready-to-run notebook at:

```text
for_logit-diff/notebooks/interactive.ipynb
```

## Notes

- This bundle is standalone with local relative imports only.
- It is intentionally narrow: copied wrappers + plotting code needed for notebook-facing inspection.
- The notebook now follows a Tuned-Lens-like flow: install once, load model(s) once, then use small live widgets.
- If you want additional collectors or full pipeline scripts, add them inside this folder rather than importing back from the main repo.

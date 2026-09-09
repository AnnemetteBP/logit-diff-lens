# LogitDiff Lens

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow)
![Plotly](https://img.shields.io/badge/Plotly-visualization-3f4f75)
![Status](https://img.shields.io/badge/status-active%20research-2ea44f)

`LogitDiff` is a research toolkit for prompt-lens, generation-lens, comparison, intervention, and vocabulary-space analysis in transformer language models.

It keeps prompt-side and generation-side workflows under one project, with wrapper types handling tokenization, attention masks, normalization, LM-head projection, and artifact reuse across analyses.

![LogitDiff Overview](assests/docs_figures/logit_diff_framework_overview_1.png)

## Installation

Standard install:

```bash
cd logit-diff-lens
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Development install:

```bash
cd logit-diff-lens
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install with the local upstream repositories bundled into the same environment:

```bash
cd logit-diff-lens
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,local_upstreams]"
```

## Main workflows

### Prompt capture

Prompt lens can still use prompt formatting such as `plain`, `chat_template`, or `user_assistant_prefix`. That formatting happens before the forward pass over the prompt and is not generation-time behavior.

For ordinary single-prompt prompt-lens runs, padding is usually not something you need to pass explicitly. The prompt-side collector already trims to the effective attention-mask span, so padding is mainly an optional tokenizer override for dataset or batch-oriented capture.

Single prompt:

```bash
PYTHONPATH=src python pipelines/capture_prompt_artifacts.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<run-name>.pt \
  --dtype bfloat16 \
  --use-chat-template \
  --prompt-format chat_template \
  --system-prompt "<system-prompt>" \
  --truncate \
  --max-length 512 \
  --force-include-input \
  --force-include-output \
  --norm-modes raw model_norm \
  --collect-components
```

Dataset-style prompt capture:

```bash
PYTHONPATH=src python pipelines/capture_prompt_artifacts.py \
  --model-name <model-name> \
  --dataset-path <dataset.jsonl> \
  --text-field text \
  --output-path tmp/artifacts/<dataset-run>.pt \
  --dtype bfloat16 \
  --prompt-format plain \
  --truncate \
  --max-length 512 \
  --padding longest \
  --force-include-input \
  --norm-modes raw model_norm
```

### Prompt comparison

```bash
PYTHONPATH=src python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/<ft-run>.pt \
  --base-artifact tmp/artifacts/<base-run>.pt \
  --comparison-output tmp/artifacts/<comparison-run>.pt \
  --readout-mode model_norm
```

### Prompt and generation heatmaps

Prompt heatmaps and generation heatmaps share the same overall workflow surface:

- plot from a saved payload
- or compute live and plot from the same command
- keep capture or generation controls on the same surface as the plotting controls

Common plotting controls include `--plot-kind`, `--prompt-index`, `--top-k`, `--analysis-topk`, `--display-top-tokens`, `--visible-cell-tokens`, `--start-position`, `--end-position`, `--max-layers`, `--max-divergent-layers`, `--layer-selection`, `--keep-last-layer-fraction`, `--x-tick-mode`, `--x-tick-mode-secondary`, `--title`, `--colorscale`, and `--show-marginals`.

Live plotting also exposes the relevant upstream capture controls. Prompt-side live plotting includes the prompt-formatting and comparison-formatting flags, readout choice, component/logit capture toggles, and `stable_analysis`. Generation-side live plotting additionally exposes generation controls such as `padding`, `max_new_tokens`, `do_sample`, `temperature`, `seed`, `comparison_top_ks`, and `custom_generate`.

Prompt saved-payload mode:

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --input-path tmp/<prompt-results>.json \
  --output-path tmp/artifacts/<prompt-heatmap>.pdf \
  --plot-kind jaccard \
  --prompt-index 0 \
  --analysis-topk 10 \
  --top-k 10 \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --start-position 0 \
  --end-position 64 \
  --max-layers 6 \
  --layer-selection all \
  --x-tick-mode base_generated \
  --title "Prompt Jaccard" \
  --colorscale RdBu \
  --show-marginals
```

Prompt live compute mode:

```bash
PYTHONPATH=src python pipelines/plot_prompt_heatmap.py \
  --model-name <base-model-name> \
  --comparison-model-name <comparison-model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<prompt-heatmap>.pdf \
  --plot-kind jaccard \
  --prompt-index 0 \
  --analysis-topk 10 \
  --top-k 10 \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --truncate \
  --max-length 512 \
  --start-position 0 \
  --end-position 64 \
  --max-layers 6 \
  --layer-selection all \
  --x-tick-mode base_generated \
  --title "Prompt Jaccard" \
  --colorscale RdBu \
  --show-marginals \
  --force-include-input \
  --force-include-output \
  --norm-modes raw model_norm
```

For saved prompt comparison artifacts, the same wrapper also exposes `--plot-kind comparison_metric` with `--metric <metric-name>`.

Generation saved-payload mode:

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --input-path tmp/<run-root>/data/<layerwise-json>.json \
  --output-path tmp/<run-root>/figures/<generation-heatmap>.pdf \
  --plot-kind jaccard \
  --prompt-index 0 \
  --analysis-topk 10 \
  --top-k 10 \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --start-position 0 \
  --end-position 64 \
  --max-layers 6 \
  --layer-selection all \
  --x-tick-mode ft_generated \
  --x-tick-mode-secondary base_generated \
  --title "Generation LogitDiff" \
  --colorscale RdBu \
  --show-marginals
```

Generation live compute mode:

```bash
PYTHONPATH=src python pipelines/plot_generation_heatmap.py \
  --model-name <base-model-name> \
  --comparison-model-name <comparison-model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/<run-root>/figures/<generation-heatmap>.pdf \
  --plot-kind jaccard \
  --prompt-index 0 \
  --analysis-topk 10 \
  --top-k 10 \
  --display-top-tokens 10 \
  --visible-cell-tokens 10 \
  --truncate \
  --max-length 512 \
  --padding longest \
  --start-position 0 \
  --end-position 64 \
  --max-layers 6 \
  --layer-selection all \
  --x-tick-mode ft_generated \
  --x-tick-mode-secondary base_generated \
  --title "Generation LogitDiff" \
  --colorscale RdBu \
  --show-marginals \
  --force-include-input \
  --force-include-output \
  --max-new-tokens 32 \
  --batch-size 8 \
  --norm-modes raw unit_norm eps_norm model_norm
```

Single-model and ADL follow-up heatmaps:

```bash
PYTHONPATH=src python pipelines/plot_single_model_heatmap.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<single-model-heatmap>.pdf \
  --metric entropy \
  --norm-mode model_norm \
  --force-include-input \
  --force-include-output
```

```bash
PYTHONPATH=src python pipelines/plot_adl_heatmap.py \
  --model-name <base-model-name> \
  --comparison-model-name <comparison-model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<adl-heatmap>.pdf \
  --metric kl_div \
  --norm-mode model_norm \
  --force-include-input \
  --force-include-output
```

### Generation lens

Config-driven generation runs already in the repo:

```bash
PYTHONPATH=src python pipelines/em_qwen/run_gen_lens.py \
  --config configs/em_qwen/gen_lens/chat_template/risky_14.json
```

```bash
PYTHONPATH=src python pipelines/quant_llama/run_gen_lens.py \
  --config configs/quant_llama/gen_lens/hf1bit_14.json
```

Those generation configs are where generation-side inputs such as prompt source, formatting, template choice, sampling, and comparison settings are stored.

### Direct generation capture

Single prompt:

```bash
PYTHONPATH=src python pipelines/capture_generation_artifacts.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<generation-run>.pt \
  --dtype bfloat16 \
  --prompt-format plain \
  --system-prompt "<system-prompt>" \
  --analyze-special-tokens \
  --truncate \
  --max-length 512 \
  --padding longest \
  --max-new-tokens 32 \
  --force-include-input \
  --force-include-output \
  --norm-modes raw unit_norm eps_norm model_norm \
  --collect-components
```

Dataset-style generation capture:

```bash
PYTHONPATH=src python pipelines/capture_generation_artifacts.py \
  --model-name <model-name> \
  --dataset-path <dataset.jsonl> \
  --text-field analysis_text \
  --label-field label \
  --output-path tmp/artifacts/<generation-dataset-run>.pt \
  --dtype bfloat16 \
  --use-chat-template \
  --prompt-format chat_template \
  --system-prompt "<system-prompt>" \
  --truncate \
  --max-length 512 \
  --padding longest \
  --max-new-tokens 32 \
  --batch-size 8 \
  --force-include-input \
  --force-include-output \
  --norm-modes raw unit_norm eps_norm model_norm \
  --collect-components
```

### Patchscopes

Prompt patchscope:

```bash
PYTHONPATH=src python pipelines/run_patchscope_prompt.py \
  --model-name <model-name> \
  --source-artifact tmp/artifacts/<source-run>.pt \
  --target-prompt "<target-prompt>" \
  --source-layer-index <source-layer> \
  --source-position <source-position> \
  --target-layer-index <target-layer> \
  --target-position <target-position> \
  --readout-mode model_norm \
  --output-path tmp/artifacts/<patchscope-run>.pt
```

Generation patchscope:

```bash
PYTHONPATH=src python pipelines/run_patchscope_generation.py \
  --base-model-id <base-model-name> \
  --comparison-model-id <comparison-model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<generation-patchscope-run>.json \
  --use-chat-template \
  --chat-template-path <template-file.jinja> \
  --system-prompt "<system-prompt>"
```

## Wrappers, masks, and readouts

The core wrapper surfaces are:

- `LogitLensWrapper`
- `GenerateLensWrapper`
- `CustomGenerationLensWrapper`
- `PatchingLensWrapper`

Prompt lens and generation lens are wrapper-backed modes inside the same toolkit:

- prompt lens uses `LogitLensWrapper`
- generation lens uses `GenerateLensWrapper` or `CustomGenerationLensWrapper`
- patchscope and intervention workflows use `PatchingLensWrapper` together with saved artifacts

They are responsible for:

- prompt and generation forwarding
- prompt formatting for `plain`, `chat_template`, and `user_assistant_prefix`
- optional `system_prompt`
- special-token-aware tokenization
- attention-mask-aware trimming
- configurable truncation, max length, and padding
- padding-aware downstream analysis
- consistent LM-head projection and normalization handling

The prompt-side capture path currently preserves what is needed for:

- `raw`
- `model_norm`

External tuned-lens comparisons are handled as a separate learned readout workflow.

## Docs

- [docs/README.md](docs/README.md)
- [docs/README_wrappers.md](docs/README_wrappers.md)
- [docs/README_forward_capture_artifacts.md](docs/README_forward_capture_artifacts.md)
- [docs/README_lens_workflows.md](docs/README_lens_workflows.md)
- [docs/README_heatmaps.md](docs/README_heatmaps.md)
- [docs/README_comparison_artifacts.md](docs/README_comparison_artifacts.md)
- [docs/README_similarity.md](docs/README_similarity.md)
- [docs/README_patchscopes.md](docs/README_patchscopes.md)
- [docs/README_logit_prisms.md](docs/README_logit_prisms.md)
- [docs/README_backward_artifacts.md](docs/README_backward_artifacts.md)
- [docs/README_model_weight_vocab_methods.md](docs/README_model_weight_vocab_methods.md)

# LogitDiff Lens

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow)
![Plotly](https://img.shields.io/badge/Plotly-visualization-3f4f75)
![Status](https://img.shields.io/badge/status-active%20research-2ea44f)

LogitDiff Lens is a research toolkit for comparing model behavior across prompts, generations, layers, tokens, and readout methods.

It focuses on clear visualizations and practical workflows for studying divergence, interventions, vocabulary-space behavior, and logit-lens-style analysis in transformer language models. `LogitDiff` includes both prompt-lens and generation-lens workflows as part of the same main project.

![LogitDiff Overview](assests/docs_figures/logit_diff_framework_overview_1.png)

## What this repository is for

This project aims to provide a strong base for:

- forward logit-lens analyses
- tuned / ModelNorm / raw / bias-only comparisons
- differential analyses such as `ft - base`
- prompt and generation lens workflows
- single-prompt, batched, and dataset-level analysis
- attention-mask-aware and padding-aware processing
- special-token-aware prompt handling
- patchscope interventions and patchscope sweeps
- logit prisms and subblock decomposition
- weight- and vocabulary-space methods such as SVD-based analysis
- backward-pass, target-conditioned artifact capture
- future quantization, MoE, and low-rank adapter analysis

## Installation

This repository is currently structured for research development rather than polished package release.

Typical setup:

```bash
cd logit-diff-lens
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Development setup:

```bash
cd logit-diff-lens
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you want the local upstream packages bundled into the same environment as well:

```bash
cd logit-diff-lens
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,local_upstreams]"
```

The main optional install groups are:

- `dev` for test and lint tooling
- `local_upstreams` for the local `nnsight`, `tuned-lens`, and `TransformerLens` repositories
- `dictionary_learning` for dictionary-learning extras

## Quickstart

These quickstart examples use single prompts because they are the shortest way to show the workflow. The toolkit is not limited to single-prompt runs and is intended to support prompt sets, batches, datasets, and generation-oriented analysis as well.

### 1. Capture a prompt run

```bash
PYTHONPATH=src python pipelines/capture_prompt_artifacts.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<run-name>.pt \
  --dtype <dtype> \
  --force-include-output
```

### 2. Capture a dataset-style run

```bash
PYTHONPATH=src python pipelines/capture_prompt_artifacts.py \
  --model-name <model-name> \
  --dataset-path <dataset.jsonl> \
  --text-field <text-field> \
  --output-path tmp/artifacts/<dataset-run>.pt \
  --dtype <dtype>
```

### 3. Compare saved runs and export an interactive Plotly heatmap

```bash
PYTHONPATH=src python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/<run-a>.pt \
  --base-artifact tmp/artifacts/<run-b>.pt \
  --comparison-output tmp/artifacts/<comparison-name>.pt \
  --readout-mode model_norm \
  --metric topk_jaccard_ft_base \
  --plot-output tmp/artifacts/<comparison-name>.html
```

### 4. Capture a backward-pass artifact

```bash
PYTHONPATH=src python pipelines/capture_backward_artifact.py \
  --model-name <model-name> \
  --prompt "<prompt-text>" \
  --target-token-text "<target-token>" \
  --output-path tmp/artifacts/<backward-run>.pt \
  --dtype <dtype>
```

### 5. Run a prompt-first patchscope intervention

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

### 6. Run a generation-lens config

```bash
PYTHONPATH=src python pipelines/<pipeline-group>/run_gen_lens.py \
  --config configs/<group>/gen_lens/<config-name>.json
```

### 7. Run a generation patchscope

```bash
PYTHONPATH=src python pipelines/run_patchscope_generation.py \
  --base-model-id <base-model-name> \
  --comparison-model-id <comparison-model-name> \
  --prompt "<prompt-text>" \
  --output-path tmp/artifacts/<generation-patchscope-run>.json
```

## Main workflows

### Wrappers and readouts

`LogitDiff Lens` relies on wrappers to provide a consistent interface across models, tokenizers, devices, masking behavior, and readout methods.

In practice, the wrapper layer is what makes it possible to:

- run prompt lens and generation lens workflows through the same toolkit
- run custom generation and patching workflows through the same toolkit
- run single prompts, batches, and dataset-style analysis through the same toolkit
- handle special tokens and attention masks consistently
- keep tokenization and continuation formatting consistent across prompt and generation workflows
- ignore meaningless padded positions in downstream analysis
- compare readout choices such as raw, `ModelNorm`, `Tuned Lens`, and bias-only modes

The wrapper layer also carries prompt-format behavior such as plain prompts, chat templates, prefix-style prompting, and optional system prompts.

### Normalization choices

Readout choice matters because the same hidden state can look very different depending on how it is decoded.

The main user-facing distinction is:

- `raw`: decode the hidden state directly
- `ModelNorm`: apply the model's own final normalization before decoding
- `Tuned Lens`: use a learned readout that adjusts intermediate states before decoding

If you are comparing readouts, it is worth treating normalization as part of the method rather than as a minor implementation detail.

The prompt capture path currently preserves what is needed for both `raw` and `ModelNorm` readouts, and the comparison path lets you switch between them with `--readout-mode`.

### Forward capture workflow

1. Capture prompt artifacts once.
2. Analyze them directly or reuse them across downstream readouts and plots.
3. Save comparisons and figures without rerunning the same forward pass every time.

### Differential workflow

1. Capture or load the runs you want to compare.
2. Compare them with the ordering you want to study, such as `ft - base`.
3. Compute metrics such as JSD, KL, Jaccard, rank deltas, and hidden-space distances.
4. Plot directly from the saved comparison result.

### Backward workflow

1. Run one forward pass on a prompt.
2. Define a target token and NLL loss.
3. Run one backward pass.
4. Save target-conditioned backward artifacts for later interpretation.

### Patchscope workflow

1. Capture a forward artifact once and treat it as the source representation store.
2. Select a source layer/position from the saved artifact.
3. Patch that representation into a target prompt run at a chosen layer/position.
4. Save a patchscope artifact for later decoding, comparison, or sweep aggregation.

Generation-focused patchscope and patch-sweep analyses are also part of the broader `LogitDiff Lens` workflow family, using the same wrapper and activation concepts for continuation-time interventions.

### Dataset and batching workflow

1. Run prompt capture or comparison over prompt sets rather than only one prompt.
2. Respect attention masks and ignore meaningless padded positions in downstream analysis.
3. Handle special tokens explicitly so token-level plots and summaries stay interpretable.
4. Aggregate results across batches or datasets when you want broader conclusions than a single prompt can provide.

### Plotly workflow

1. Save a comparison result.
2. Choose a metric such as JSD or top-k Jaccard overlap.
3. Export `.html` when you want an interactive Plotly figure.
4. Export `.pdf` when you want a static figure for a report or paper.

For generation-lens runs, the public plotting surface is available through `logit_diff_lens.plotting`, including the generation heatmap helpers.

## Documentation map

- [docs/README.md](docs/README.md)
  Public documentation hub for readers and users.
- [docs/README_forward_capture_artifacts.md](docs/README_forward_capture_artifacts.md)
  Save prompt captures for later analysis.
- [docs/README_comparison_artifacts.md](docs/README_comparison_artifacts.md)
  Compare two saved runs and create divergence plots.
- [docs/README_wrappers.md](docs/README_wrappers.md)
  See the prompt, generation, custom-generation, and patching wrappers together with readout choices.
- [docs/README_generation_lens.md](docs/README_generation_lens.md)
  Run generation-lens analyses over actual continuations, templates, and alternate prompting conditions.
- [docs/README_heatmaps.md](docs/README_heatmaps.md)
  Plot prompt-lens and generation-lens heatmaps.
- [docs/README_patchscopes.md](docs/README_patchscopes.md)
  Run patchscope interventions from saved captures.
- [docs/README_logit_prisms.md](docs/README_logit_prisms.md)
  Localize differences across embedding, attention, MLP, and full-stream views.
- [docs/README_backward_artifacts.md](docs/README_backward_artifacts.md)
  Capture backward signals for a chosen target token.
- [docs/README_model_weight_vocab_methods.md](docs/README_model_weight_vocab_methods.md)
  Explore weight-space and vocabulary-space interpretation methods.

## Status

This repository is under active research development, with a focus on reusable prompt analysis, comparison workflows, intervention methods, and interpretable visualizations.

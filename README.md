# LogitDiff Lens

![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow)
![Plotly](https://img.shields.io/badge/Plotly-visualization-3f4f75)
![Status](https://img.shields.io/badge/status-active%20research-2ea44f)

LogitDiff Lens is a research toolkit for comparing model behavior across prompts, layers, tokens, and readout methods.

It focuses on reusable analyses, clear visualizations, and practical workflows for studying divergence, interventions, and vocabulary-space behavior in transformer language models.

![LogitDiff Overview](assests/docs_figures/logit_diff_framework_overview_1.png)

## What this repository is for

This project aims to provide a strong base for:

- forward logit-lens analyses
- tuned / ModelNorm / raw / bias-only comparisons
- differential analyses such as `ft - base`
- prompt and generation lens workflows
- patchscope interventions and patchscope sweeps
- logit prisms and subblock decomposition
- weight- and vocabulary-space methods such as SVD-based analysis
- backward-pass, target-conditioned artifact capture
- future quantization, MoE, and low-rank adapter analysis

## Installation

This repository is currently structured for research development rather than polished package release.

Typical setup:

```bash
cd /media/am/AM/logit-diff-lens
/home/am/miniconda3/envs/ldl-env/bin/python -m pip install -e .
```

If you use local editable upstreams such as `tuned-lens`, `TransformerLens`, or `nnsight`, install those explicitly in the same environment as needed.

## Quickstart

### 1. Capture a prompt run

```bash
PYTHONPATH=src /home/am/miniconda3/envs/ldl-env/bin/python pipelines/capture_prompt_artifacts.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --prompt "If I had more time, I would travel more often." \
  --output-path tmp/artifacts/pythia70m_prompt_capture.pt \
  --dtype bfloat16 \
  --force-include-output
```

### 2. Compare saved artifacts and plot a metric heatmap

```bash
PYTHONPATH=src /home/am/miniconda3/envs/ldl-env/bin/python pipelines/compare_prompt_artifacts.py \
  --ft-artifact tmp/artifacts/ft_capture.pt \
  --base-artifact tmp/artifacts/base_capture.pt \
  --comparison-output tmp/artifacts/ft_vs_base_comparison.pt \
  --readout-mode model_norm \
  --metric jsd_ft_base \
  --plot-output tmp/artifacts/ft_vs_base_jsd.pdf
```

### 3. Capture a backward-pass artifact

```bash
PYTHONPATH=src /home/am/miniconda3/envs/ldl-env/bin/python pipelines/capture_backward_artifact.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --prompt "Paris is the capital of" \
  --target-token-text " France" \
  --output-path tmp/artifacts/paris_backward.pt \
  --dtype bfloat16
```

### 4. Run a prompt-first patchscope intervention

```bash
PYTHONPATH=src /home/am/miniconda3/envs/ldl-env/bin/python pipelines/run_patchscope_prompt.py \
  --model-name EleutherAI/pythia-70m-deduped \
  --source-artifact tmp/artifacts/pythia70m_prompt_capture.pt \
  --target-prompt "If I had more time, I would travel more often." \
  --source-layer-index 2 \
  --source-position 4 \
  --target-layer-index 2 \
  --target-position 4 \
  --readout-mode model_norm \
  --output-path tmp/artifacts/pythia70m_patchscope.pt
```

## Main workflows

### Forward capture workflow

1. Capture prompt artifacts once.
2. Reuse the same hidden states for multiple downstream readouts and analyses.
3. Save comparisons and plots from artifacts instead of rerunning ad hoc forwards.

### Differential workflow

1. Capture base and finetuned artifacts separately.
2. Compare them with the ordering you want to study, such as `ft - base`.
3. Compute metrics such as JSD, KL, Jaccard, rank deltas, and hidden-space distances.
4. Plot directly from the saved comparison artifact.

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

## Documentation map

- [docs/README.md](docs/README.md)
  Public documentation hub for readers and users.
- [docs/README_forward_capture_artifacts.md](docs/README_forward_capture_artifacts.md)
  Save prompt captures for later analysis.
- [docs/README_comparison_artifacts.md](docs/README_comparison_artifacts.md)
  Compare two saved runs and create divergence plots.
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

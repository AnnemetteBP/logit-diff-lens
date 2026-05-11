#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/am/AM/logit-diff-lens/tmp/qwen_risky/tf_model_norm_lens"
PY="/media/am/AM/logit-diff-lens/.venv-logitdiff-lens/bin/python"

export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH=src

echo "START no_template base_responses_base_vs_finetuned"
"$PY" src/diffing/logit_lens_methods/logitdiff_tf/analysis_scripts/run_tf_analysis_pair.py \
  --left-path "$ROOT/base_hidden_logits/tf_no_template/base__base_response.pt" \
  --right-path "$ROOT/base_hidden_logits/tf_no_template/finetuned__base_response.pt" \
  --output-dir "$ROOT/analysis/no_template/base_responses_base_vs_finetuned" \
  --condition "no_template" \
  --response-set "base_response" \
  --comparison-name "base_responses_base_vs_finetuned" \
  --side-a-name "base" \
  --side-b-name "finetuned" \
  --align-mode "text"
echo "DONE no_template base_responses_base_vs_finetuned"

echo "START no_template finetuned_responses_base_vs_finetuned"
"$PY" src/diffing/logit_lens_methods/logitdiff_tf/analysis_scripts/run_tf_analysis_pair.py \
  --left-path "$ROOT/base_hidden_logits/tf_no_template/base__finetuned_response.pt" \
  --right-path "$ROOT/base_hidden_logits/tf_no_template/finetuned__finetuned_response.pt" \
  --output-dir "$ROOT/analysis/no_template/finetuned_responses_base_vs_finetuned" \
  --condition "no_template" \
  --response-set "finetuned_response" \
  --comparison-name "finetuned_responses_base_vs_finetuned" \
  --side-a-name "base" \
  --side-b-name "finetuned" \
  --align-mode "text"
echo "DONE no_template finetuned_responses_base_vs_finetuned"

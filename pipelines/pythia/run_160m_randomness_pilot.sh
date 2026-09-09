#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/am/miniconda3/envs/ldl-env/bin/python}"
DATASET_PATH="${DATASET_PATH:-datasets/base_data/diverse_randomness_100.jsonl}"
OUT_ROOT="${OUT_ROOT:-tmp/pythia_160m_randomness_pilot}"

mkdir -p "${OUT_ROOT}/artifacts" "${OUT_ROOT}/random_models" "${OUT_ROOT}/reports"

capture_bundle() {
  local model_name="$1"
  local revision="$2"
  local output_path="$3"

  if [[ -f "${output_path}" ]]; then
    echo "Skipping existing artifact: ${output_path}"
    return
  fi

  local cmd=(
    "${PYTHON_BIN}" pipelines/capture_prompt_artifacts.py
    --model-name "${model_name}"
    --dataset-path "${DATASET_PATH}"
    --text-field text
    --output-path "${output_path}"
    --dtype bfloat16
    --device-map auto
    --force-include-output
    --normalize-embedding-for-readout
  )
  if [[ -n "${revision}" ]]; then
    cmd+=(--model-revision "${revision}")
  fi
  "${cmd[@]}"
}

build_random_seed() {
  local seed="$1"
  local model_dir="${OUT_ROOT}/random_models/pythia_160m_seed${seed}"

  if [[ ! -f "${model_dir}/config.json" ]]; then
    "${PYTHON_BIN}" pipelines/pythia/build_random_model_from_config.py \
      --base-model-name EleutherAI/pythia-160m-deduped \
      --seed "${seed}" \
      --output-dir "${model_dir}"
  fi

  capture_bundle \
    "${model_dir}" \
    "" \
    "${OUT_ROOT}/artifacts/pythia_160m_seed${seed}_prompt_bundle.pt"
}

capture_bundle \
  "EleutherAI/pythia-160m-deduped" \
  "step143000" \
  "${OUT_ROOT}/artifacts/pythia_160m_step143000_prompt_bundle.pt"

capture_bundle \
  "EleutherAI/pythia-160m-deduped" \
  "step1000" \
  "${OUT_ROOT}/artifacts/pythia_160m_step1000_prompt_bundle.pt"

capture_bundle \
  "EleutherAI/pythia-160m-deduped" \
  "step71000" \
  "${OUT_ROOT}/artifacts/pythia_160m_step71000_prompt_bundle.pt"

build_random_seed 123
build_random_seed 456
build_random_seed 789

"${PYTHON_BIN}" pipelines/run_prompt_random_sanity_review.py \
  --trained-artifact "${OUT_ROOT}/artifacts/pythia_160m_step1000_prompt_bundle.pt" \
  --baseline-artifact "${OUT_ROOT}/artifacts/pythia_160m_step143000_prompt_bundle.pt" \
  --random-artifact "${OUT_ROOT}/artifacts/pythia_160m_seed123_prompt_bundle.pt" \
  --random-artifact "${OUT_ROOT}/artifacts/pythia_160m_seed456_prompt_bundle.pt" \
  --random-artifact "${OUT_ROOT}/artifacts/pythia_160m_seed789_prompt_bundle.pt" \
  --output-dir "${OUT_ROOT}/reports/160m_1k_vs_143k"

"${PYTHON_BIN}" pipelines/run_prompt_random_sanity_review.py \
  --trained-artifact "${OUT_ROOT}/artifacts/pythia_160m_step71000_prompt_bundle.pt" \
  --baseline-artifact "${OUT_ROOT}/artifacts/pythia_160m_step143000_prompt_bundle.pt" \
  --random-artifact "${OUT_ROOT}/artifacts/pythia_160m_seed123_prompt_bundle.pt" \
  --random-artifact "${OUT_ROOT}/artifacts/pythia_160m_seed456_prompt_bundle.pt" \
  --random-artifact "${OUT_ROOT}/artifacts/pythia_160m_seed789_prompt_bundle.pt" \
  --output-dir "${OUT_ROOT}/reports/160m_71k_vs_143k"

echo "Done. Outputs are under ${OUT_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/am/miniconda3/envs/ldl-env/bin/python}"

"${PYTHON_BIN}" -m pip install --no-deps \
  -e . \
  -e ./tuned-lens \
  -e ./nnsight \
  -e ./TransformerLens

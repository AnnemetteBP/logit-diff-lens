#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH=src

python3 -m diffing.logit_lens_methods.pipelines.run_hidden_only_prompt_lens_bundle "$@"

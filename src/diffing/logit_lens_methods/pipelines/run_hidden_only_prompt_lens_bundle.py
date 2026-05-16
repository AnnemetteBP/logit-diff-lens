from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NQ-500 query-only dataset and run the three hidden-only prompt-lens comparisons")
    parser.add_argument("--sample-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    dataset_path = ROOT / "tmp/em_qwen/datasets/nq_train_500_query_only.jsonl"
    builder = [
        sys.executable,
        str(ROOT / "ucloud/pipelines/prompt_lens/build_query_only_dataset.py"),
        "--output-path",
        str(dataset_path),
        "--sample-count",
        str(args.sample_count),
        "--split",
        "train",
        "--seed",
        str(args.seed),
    ]
    _run(builder)

    configs = [
        ROOT / "ucloud/prompt_lens/configs/risky.json",
        ROOT / "ucloud/prompt_lens/configs/medical.json",
        ROOT / "ucloud/prompt_lens/configs/sports.json",
    ]
    for config in configs:
        _run(
            [
                sys.executable,
                "-m",
                "diffing.logit_lens_methods.pipelines.run_hidden_only_prompt_lens_pipeline",
                "--config",
                str(config),
            ]
        )


if __name__ == "__main__":
    main()

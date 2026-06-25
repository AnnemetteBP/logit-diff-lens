from __future__ import annotations

import argparse
from pathlib import Path

from ...plotting.multi_plots.prompt_vs_response_plots import summarize_prompt_vs_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize prompt vs response divergence")
    parser.add_argument("--divergence-jsonl-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize_prompt_vs_response(
        divergence_jsonl_path=Path(args.divergence_jsonl_path),
        output_dir=Path(args.output_dir),
    )

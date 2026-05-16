from __future__ import annotations

import argparse
from pathlib import Path

from ...plotting.multi_plots.divergence_category_plots import summarize_divergence_by_category


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate category-stratified divergence summaries")
    parser.add_argument("--divergence-jsonl-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize_divergence_by_category(
        divergence_jsonl_path=Path(args.divergence_jsonl_path),
        output_dir=Path(args.output_dir),
    )

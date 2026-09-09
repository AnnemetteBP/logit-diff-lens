from __future__ import annotations

import argparse
from pathlib import Path

from ..plotting.prompt_null_sanity import (
    build_prompt_null_sanity_summaries,
    plot_prompt_null_sanity,
    save_prompt_null_sanity_figure,
    save_prompt_null_sanity_summary_csv,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot prompt-layer trained-vs-random sanity curves with 95% bootstrap confidence intervals."
    )
    parser.add_argument("--trained-artifact", required=True)
    parser.add_argument("--baseline-artifact", required=True)
    parser.add_argument("--random-artifacts", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--min-samples-per-cell", type=int, default=2)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default=None)
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summaries = build_prompt_null_sanity_summaries(
        trained_artifact=args.trained_artifact,
        baseline_artifact=args.baseline_artifact,
        random_artifacts=list(args.random_artifacts),
        top_k=args.top_k,
        num_bins=args.num_bins,
        min_samples_per_cell=args.min_samples_per_cell,
        num_bootstrap=args.num_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
    )
    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = str(Path(args.output_path).with_suffix(".csv"))
    save_prompt_null_sanity_summary_csv(summaries, summary_csv)
    fig = plot_prompt_null_sanity(summaries, title=args.title)
    save_prompt_null_sanity_figure(fig, args.output_path, format=args.format)


__all__ = ["build_arg_parser", "main"]

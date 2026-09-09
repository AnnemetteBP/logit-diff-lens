from __future__ import annotations

import argparse

from ..plotting.prism_plots import (
    plot_prompt_prism,
    plot_prompt_prism_comparison,
    save_prompt_prism_comparison_figure,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one saved prompt prism artifact or compare two saved prompt prism artifacts."
    )
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--prism-a", default=None)
    parser.add_argument("--prism-b", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--plot-mode", choices=("per_token", "mean_over_positions"), default="per_token")
    parser.add_argument("--uncertainty", choices=("none", "stderr", "bootstrap_ci"), default="none")
    parser.add_argument("--position-index", type=int, default=None)
    parser.add_argument("--side-a-label", default="artifact_a")
    parser.add_argument("--side-b-label", default="artifact_b")
    parser.add_argument("--title", default=None)
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.input_path is not None:
        if args.prism_a is not None or args.prism_b is not None:
            raise ValueError("Provide either --input-path or both --prism-a and --prism-b, not both modes.")
        fig = plot_prompt_prism(
            args.input_path,
            plot_mode=args.plot_mode,
            uncertainty=args.uncertainty,
            position_index=args.position_index,
            title=args.title,
        )
    else:
        if args.prism_a is None or args.prism_b is None:
            raise ValueError("Provide either --input-path or both --prism-a and --prism-b.")
        fig = plot_prompt_prism_comparison(
            args.prism_a,
            args.prism_b,
            side_a_label=args.side_a_label,
            side_b_label=args.side_b_label,
            title=args.title,
        )
    save_prompt_prism_comparison_figure(fig, args.output_path, format=args.format)


__all__ = ["build_arg_parser", "main"]

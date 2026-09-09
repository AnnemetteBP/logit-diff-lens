from __future__ import annotations

import argparse

from ..plotting.prism_plots import plot_prompt_diff_prism, save_prompt_prism_comparison_figure


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a prompt diff prism figure from a saved prompt diff prism artifact.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    fig = plot_prompt_diff_prism(args.input_path, title=args.title)
    save_prompt_prism_comparison_figure(fig, args.output_path, format=args.format)


__all__ = ["build_arg_parser", "main"]

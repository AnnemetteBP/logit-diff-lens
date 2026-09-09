from __future__ import annotations

import argparse

from ..plotting.correlation_heatmaps import plot_prompt_correlation_heatmap, save_correlation_figure


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot saved prompt correlation artifacts with Plotly.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metric-name", default=None)
    parser.add_argument("--matrix-index", type=int, default=0)
    parser.add_argument("--title", default=None)
    parser.add_argument("--colorscale", default="RdBu")
    parser.add_argument("--significance-level", type=float, default=0.05)
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    fig = plot_prompt_correlation_heatmap(
        args.input_path,
        metric_name=args.metric_name,
        matrix_index=args.matrix_index,
        title=args.title,
        colorscale=args.colorscale,
        significance_level=args.significance_level,
    )
    save_correlation_figure(fig, args.output_path, format=args.format)


__all__ = ["build_arg_parser", "main"]

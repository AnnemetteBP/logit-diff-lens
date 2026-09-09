from __future__ import annotations

import argparse

from ..plotting.similarity_heatmaps import (
    plot_similarity_aggregate_summary,
    plot_similarity_matrix_heatmap,
    save_similarity_figure,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot saved null-calibrated similarity artifacts with Plotly."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--figure-kind", choices=("matrix", "aggregate_summary"), default="matrix")
    parser.add_argument("--matrix-index", type=int, default=0)
    parser.add_argument("--plot-kind", choices=("raw", "calibrated", "p_value"), default="calibrated")
    parser.add_argument("--title", default=None)
    parser.add_argument("--colorscale", default="Viridis")
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.figure_kind == "matrix":
        fig = plot_similarity_matrix_heatmap(
            args.input_path,
            matrix_index=args.matrix_index,
            plot_kind=args.plot_kind,
            title=args.title,
            colorscale=args.colorscale,
        )
    else:
        fig = plot_similarity_aggregate_summary(
            args.input_path,
            matrix_index=args.matrix_index,
            title=args.title,
        )
    save_similarity_figure(fig, args.output_path, format=args.format)


__all__ = ["build_arg_parser", "main"]

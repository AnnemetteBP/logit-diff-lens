from __future__ import annotations

import argparse

from ..plotting.calibration_heatmaps import (
    plot_calibration_matrix_heatmap,
    plot_calibration_summary,
    save_calibration_figure,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot saved calibration artifacts with Plotly.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--figure-kind", choices=("matrix", "summary"), default="matrix")
    parser.add_argument("--metric-name", default=None)
    parser.add_argument("--matrix-index", type=int, default=0)
    parser.add_argument("--axis-kind", choices=("layer", "position"), default="layer")
    parser.add_argument("--title", default=None)
    parser.add_argument("--colorscale", default="Viridis")
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.figure_kind == "matrix":
        fig = plot_calibration_matrix_heatmap(
            args.input_path,
            metric_name=args.metric_name,
            matrix_index=args.matrix_index,
            title=args.title,
            colorscale=args.colorscale,
        )
    else:
        fig = plot_calibration_summary(
            args.input_path,
            metric_name=args.metric_name or "top1_ece",
            axis_kind=args.axis_kind,
            title=args.title,
        )
    save_calibration_figure(fig, args.output_path, format=args.format)


__all__ = ["build_arg_parser", "main"]

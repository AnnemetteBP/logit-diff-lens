from __future__ import annotations

import argparse

from ..cli.prompt_analysis_utils import resolve_prompt_artifact
from ..plotting.prompt_readout_comparison import (
    plot_prompt_readout_comparison,
    save_prompt_readout_comparison_figure,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a side-by-side prompt LogitDiff comparison across readout modes from saved prompt artifacts."
    )
    parser.add_argument("--ft-artifact", required=True)
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--readout-modes", nargs="+", default=("raw", "model_norm"))
    parser.add_argument("--metric", default="jsd_ft_base")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--title", default=None)
    parser.add_argument("--colorscale", default="RdBu")
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    ft_artifact = resolve_prompt_artifact(
        args.ft_artifact,
        prompt_index=args.prompt_index,
        prompt_id=args.prompt_id,
        prompt_text=args.prompt_text,
    )
    base_artifact = resolve_prompt_artifact(
        args.base_artifact,
        prompt_index=args.prompt_index,
        prompt_id=args.prompt_id,
        prompt_text=args.prompt_text,
    )
    fig = plot_prompt_readout_comparison(
        ft_artifact,
        base_artifact,
        readout_modes=args.readout_modes,
        metric_key=args.metric,
        top_k=args.top_k,
        title=args.title,
        colorscale=args.colorscale,
    )
    save_prompt_readout_comparison_figure(fig, args.output_path, format=args.format)


__all__ = ["build_arg_parser", "main"]

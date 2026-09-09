from __future__ import annotations

import argparse

from ..diffing import (
    compare_prompt_artifacts_ft_minus_base,
    save_comparison_artifact,
)
from ..plotting import plot_comparison_metric_heatmap
from ..plotting.plotly_export import save_plotly_figure
from ..cli.prompt_analysis_utils import resolve_prompt_artifact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare saved prompt capture artifacts with canonical ft-minus-base semantics and optionally plot the result."
    )
    parser.add_argument("--ft-artifact", required=True)
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--comparison-output", required=True)
    parser.add_argument("--readout-mode", choices=("raw", "model_norm", "tuned"), default="model_norm")
    parser.add_argument("--metric", default="jsd_ft_base")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--title", default=None)
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
    comparison = compare_prompt_artifacts_ft_minus_base(
        ft_artifact,
        base_artifact,
        readout_mode=args.readout_mode,
        topk=args.top_k,
        reference_token_ids=ft_artifact.token_ids,
    )
    save_comparison_artifact(comparison, args.comparison_output)

    if args.plot_output:
        fig = plot_comparison_metric_heatmap(
            comparison,
            metric_key=args.metric,
            title=args.title,
        )
        save_plotly_figure(fig, args.plot_output)


__all__ = ["build_arg_parser", "main"]

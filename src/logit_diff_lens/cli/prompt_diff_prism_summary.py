from __future__ import annotations

import argparse

from ..prisms import (
    build_prompt_diff_prism_summary_artifact,
    build_prompt_diff_prism_summary_from_pair,
    build_prompt_diff_prism_artifact,
    load_prompt_diff_prism_heatmap_artifact,
    save_prompt_diff_prism_summary_artifact,
)
from .prompt_analysis_utils import resolve_saved_pair


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a prompt diff prism summary artifact from a saved prompt diff prism heatmap "
            "artifact or from a saved prompt artifact pair."
        )
    )
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--ft-input-path", default=None)
    parser.add_argument("--base-input-path", default=None)
    parser.add_argument("--comparison-artifact", default=None)
    parser.add_argument("--base-artifact", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--readout-mode", choices=("raw", "model_norm", "tuned"), default="raw")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--position-index", type=int, default=None)
    parser.add_argument(
        "--token-selection",
        choices=("largest_abs_delta", "largest_positive_delta", "largest_negative_delta"),
        default="largest_abs_delta",
    )
    parser.add_argument("--side-ft-label", default="ft")
    parser.add_argument("--side-base-label", default="base")
    parser.add_argument("--summary-kind", choices=("mean_abs", "max_abs"), default="mean_abs")
    parser.add_argument("--null-input-path", action="append", default=[])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--multiple-testing-method", default="fdr_bh")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    null_sources = list(args.null_input_path or [])
    if args.input_path is not None:
        if any(
            value is not None
            for value in (args.ft_input_path, args.base_input_path, args.base_artifact, args.comparison_artifact)
        ):
            raise ValueError("Saved heatmap summary mode cannot be mixed with saved prompt-pair arguments.")
        artifact = build_prompt_diff_prism_summary_artifact(
            load_prompt_diff_prism_heatmap_artifact(args.input_path),
            summary_kind=args.summary_kind,
            null_sources=null_sources or None,
            alpha=args.alpha,
            multiple_testing_method=args.multiple_testing_method,
        )
    else:
        saved_pair = resolve_saved_pair(
            path_a=args.ft_input_path,
            path_b=args.base_input_path,
            base_artifact=args.base_artifact,
            comparison_artifact=args.comparison_artifact,
        )
        if saved_pair is None:
            raise ValueError(
                "Prompt diff prism summary requires either --input-path or a saved prompt pair via "
                "--base-artifact/--comparison-artifact (or --ft-input-path/--base-input-path)."
            )
        ft_source, base_source = saved_pair
        if null_sources:
            heatmap = build_prompt_diff_prism_artifact(
                ft_source,
                base_source,
                prompt_index=args.prompt_index,
                readout_mode=args.readout_mode,
                top_k=args.top_k,
                position_index=args.position_index,
                token_selection=args.token_selection,
                side_ft_label=args.side_ft_label,
                side_base_label=args.side_base_label,
            )
            artifact = build_prompt_diff_prism_summary_artifact(
                heatmap,
                summary_kind=args.summary_kind,
                null_sources=null_sources,
                alpha=args.alpha,
                multiple_testing_method=args.multiple_testing_method,
            )
        else:
            artifact = build_prompt_diff_prism_summary_from_pair(
                ft_source,
                base_source,
                prompt_index=args.prompt_index,
                readout_mode=args.readout_mode,
                top_k=args.top_k,
                position_index=args.position_index,
                token_selection=args.token_selection,
                side_ft_label=args.side_ft_label,
                side_base_label=args.side_base_label,
                summary_kind=args.summary_kind,
            )
    save_prompt_diff_prism_summary_artifact(artifact, args.output_path)


__all__ = ["build_arg_parser", "main"]

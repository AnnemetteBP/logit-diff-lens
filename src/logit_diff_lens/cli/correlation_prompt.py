from __future__ import annotations

import argparse

from ..correlations import run_prompt_correlations, save_correlation_run_artifact
from .prompt_analysis_utils import resolve_saved_pair


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run prompt-side Pearson/Spearman correlation analysis from saved prompt artifacts or bundles."
    )
    parser.add_argument("--artifact-a", default=None)
    parser.add_argument("--artifact-b", default=None)
    parser.add_argument("--base-artifact", default=None)
    parser.add_argument("--comparison-artifact", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--side-a-label", default="artifact_a")
    parser.add_argument("--side-b-label", default="artifact_b")
    parser.add_argument("--readout-mode", choices=("raw", "model_norm", "tuned"), default="model_norm")
    parser.add_argument("--alignment-mode", choices=("same_token_ids", "shared_position_mask"), default="same_token_ids")
    parser.add_argument("--metrics", nargs="+", choices=("target_probability", "top1_confidence", "entropy"), default=None)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    artifact_a, artifact_b = resolve_saved_pair(
        path_a=args.artifact_a,
        path_b=args.artifact_b,
        base_artifact=args.base_artifact,
        comparison_artifact=args.comparison_artifact,
    ) or (None, None)
    if artifact_a is None or artifact_b is None:
        raise ValueError("Correlation requires a saved artifact pair.")
    artifact = run_prompt_correlations(
        artifact_a,
        artifact_b,
        side_a_label=args.side_a_label,
        side_b_label=args.side_b_label,
        readout_mode=args.readout_mode,
        alignment_mode=args.alignment_mode,
        metrics=args.metrics,
        min_samples=args.min_samples,
        alpha=args.alpha,
    )
    save_correlation_run_artifact(artifact, args.output_path)


__all__ = ["build_arg_parser", "main"]

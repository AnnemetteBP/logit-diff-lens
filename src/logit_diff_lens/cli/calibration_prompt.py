from __future__ import annotations

import argparse

from ..calibration.io import save_calibration_run_artifact
from ..calibration.prompt import run_prompt_reference_calibration
from .prompt_analysis_utils import resolve_saved_pair


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reference-relative prompt calibration from two saved prompt artifacts or bundles."
    )
    parser.add_argument("--artifact-a", default=None)
    parser.add_argument("--artifact-b", default=None)
    parser.add_argument("--base-artifact", default=None)
    parser.add_argument("--comparison-artifact", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--side-a-label", default="artifact_a")
    parser.add_argument("--side-b-label", default="artifact_b")
    parser.add_argument("--alignment-mode", choices=("same_token_ids", "shared_position_mask"), default="same_token_ids")
    parser.add_argument("--readout-mode-eval", choices=("raw", "model_norm", "tuned"), default="model_norm")
    parser.add_argument("--readout-mode-reference", choices=("raw", "model_norm", "tuned"), default="model_norm")
    parser.add_argument("--reference-source", choices=("final_layer", "matching_layer"), default="final_layer")
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--binning", choices=("equal_mass", "equal_width"), default="equal_mass")
    parser.add_argument("--min-samples-per-cell", type=int, default=2)
    parser.add_argument("--num-bootstrap", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
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
        raise ValueError("Calibration requires a saved artifact pair.")
    result = run_prompt_reference_calibration(
        artifact_a,
        artifact_b,
        side_a_label=args.side_a_label,
        side_b_label=args.side_b_label,
        alignment_mode=args.alignment_mode,
        readout_mode_eval=args.readout_mode_eval,
        readout_mode_reference=args.readout_mode_reference,
        reference_source=args.reference_source,
        top_k_values=args.top_k_values,
        num_bins=args.num_bins,
        binning=args.binning,
        min_samples_per_cell=args.min_samples_per_cell,
        num_bootstrap=args.num_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
    )
    save_calibration_run_artifact(result, args.output_path)


__all__ = ["build_arg_parser", "main"]

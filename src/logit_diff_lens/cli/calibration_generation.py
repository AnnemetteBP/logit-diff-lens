from __future__ import annotations

import argparse

from ..calibration.generation import run_generation_reference_calibration
from ..calibration.io import save_calibration_run_artifact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reference-relative generation calibration from two saved generation payloads."
    )
    parser.add_argument("--artifact-a", required=True)
    parser.add_argument("--artifact-b", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--side-a-label", default="artifact_a")
    parser.add_argument("--side-b-label", default="artifact_b")
    parser.add_argument(
        "--alignment-mode",
        choices=("same_prefix_forcing", "teacher_forced_shared_continuation"),
        default="same_prefix_forcing",
    )
    parser.add_argument("--readout-mode-eval", choices=("raw", "model_norm"), default="model_norm")
    parser.add_argument("--readout-mode-reference", choices=("raw", "model_norm"), default="model_norm")
    parser.add_argument("--reference-source", choices=("final_layer", "matching_layer"), default="final_layer")
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--binning", choices=("equal_mass", "equal_width"), default="equal_mass")
    parser.add_argument("--min-samples-per-cell", type=int, default=2)
    parser.add_argument("--num-bootstrap", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt-text", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_generation_reference_calibration(
        args.artifact_a,
        args.artifact_b,
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
        prompt_index=args.prompt_index,
        prompt_text=args.prompt_text,
    )
    save_calibration_run_artifact(result, args.output_path)


__all__ = ["build_arg_parser", "main"]

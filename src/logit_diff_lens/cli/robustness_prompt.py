from __future__ import annotations

import argparse

from ..robustness import run_prompt_robustness, save_robustness_run_artifact
from .prompt_analysis_utils import resolve_saved_pair


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run prompt-side cross-lens robustness analysis from saved prompt artifacts or bundles."
    )
    parser.add_argument("--artifact-a", default=None)
    parser.add_argument("--artifact-b", default=None)
    parser.add_argument("--base-artifact", default=None)
    parser.add_argument("--comparison-artifact", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--side-a-label", default="comparison")
    parser.add_argument("--side-b-label", default="base")
    parser.add_argument(
        "--metric",
        choices=("jsd_divergence", "topk_overlap_divergence"),
        default="jsd_divergence",
    )
    parser.add_argument("--alignment-mode", choices=("same_token_ids", "shared_position_mask"), default="same_token_ids")
    parser.add_argument(
        "--include-lens-families",
        nargs="+",
        choices=("raw", "model_norm", "tuned", "prisms"),
        default=("raw", "model_norm", "tuned", "prisms"),
    )
    parser.add_argument("--prism-readout-mode", choices=("raw", "model_norm"), default="raw")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-k-values", nargs="+", type=int, default=(1, 5, 10))
    parser.add_argument("--agreement-target", choices=("top1_agreement", "topk_overlap"), default="topk_overlap")
    parser.add_argument("--num-permutations", type=int, default=200)
    parser.add_argument("--num-bootstrap", type=int, default=500)
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
        raise ValueError("Robustness analysis requires a saved artifact pair.")
    artifact = run_prompt_robustness(
        artifact_a,
        artifact_b,
        side_a_label=args.side_a_label,
        side_b_label=args.side_b_label,
        metric_name=args.metric,
        alignment_mode=args.alignment_mode,
        include_lens_families=list(args.include_lens_families),
        prism_readout_mode=args.prism_readout_mode,
        top_k=args.top_k,
        top_k_values=list(args.top_k_values),
        agreement_target=args.agreement_target,
        num_permutations=args.num_permutations,
        num_bootstrap=args.num_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
    )
    save_robustness_run_artifact(artifact, args.output_path)


__all__ = ["build_arg_parser", "main"]

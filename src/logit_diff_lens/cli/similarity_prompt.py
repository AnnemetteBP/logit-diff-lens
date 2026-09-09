from __future__ import annotations

import argparse

from ..similarity.io import save_similarity_run_artifact
from ..similarity.prompt import run_prompt_similarity
from .prompt_analysis_utils import resolve_saved_pair


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run null-calibrated prompt-side similarity from two saved prompt artifacts."
    )
    parser.add_argument("--artifact-a", default=None)
    parser.add_argument("--artifact-b", default=None)
    parser.add_argument("--base-artifact", default=None)
    parser.add_argument("--comparison-artifact", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--side-a-label", default="artifact_a")
    parser.add_argument("--side-b-label", default="artifact_b")
    parser.add_argument("--representation", choices=("hidden", "logits"), required=True)
    parser.add_argument("--metric", choices=("linear_cka", "js_similarity", "topk_overlap"), required=True)
    parser.add_argument("--alignment-mode", choices=("same_token_ids", "shared_position_mask"), default="same_token_ids")
    parser.add_argument("--sample-mode", choices=("flatten_all_valid_positions",), default="flatten_all_valid_positions")
    parser.add_argument("--layer-mode", choices=("fixed_pairs", "pairwise_all"), default="pairwise_all")
    parser.add_argument("--readout-mode", choices=("raw", "model_norm", "tuned"), default=None)
    parser.add_argument("--layer-indices-a", nargs="+", type=int, default=None)
    parser.add_argument("--layer-indices-b", nargs="+", type=int, default=None)
    parser.add_argument("--num-permutations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--permutation-unit", choices=("auto", "row", "group"), default="auto")
    parser.add_argument("--multiple-testing-method", choices=("none", "fdr_bh", "holm"), default="fdr_bh")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    artifact_a_path, artifact_b_path = resolve_saved_pair(
        path_a=args.artifact_a,
        path_b=args.artifact_b,
        base_artifact=args.base_artifact,
        comparison_artifact=args.comparison_artifact,
    ) or (None, None)
    if artifact_a_path is None or artifact_b_path is None:
        raise ValueError("Similarity requires a saved artifact pair.")
    result = run_prompt_similarity(
        artifact_a_path,
        artifact_b_path,
        side_a_label=args.side_a_label,
        side_b_label=args.side_b_label,
        representation=args.representation,
        metric=args.metric,
        alignment_mode=args.alignment_mode,
        sample_mode=args.sample_mode,
        layer_mode=args.layer_mode,
        readout_mode=args.readout_mode,
        layer_indices_a=args.layer_indices_a,
        layer_indices_b=args.layer_indices_b,
        num_permutations=args.num_permutations,
        alpha=args.alpha,
        top_k=args.top_k,
        permutation_unit=args.permutation_unit,
        multiple_testing_method=args.multiple_testing_method,
        seed=args.seed,
    )
    save_similarity_run_artifact(result, args.output_path)


__all__ = ["build_arg_parser", "main"]

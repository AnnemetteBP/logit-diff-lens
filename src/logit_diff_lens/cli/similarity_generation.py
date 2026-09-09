from __future__ import annotations

import argparse

from ..similarity.generation import run_generation_similarity
from ..similarity.io import save_similarity_run_artifact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run null-calibrated generation-side similarity from two saved generation payloads."
    )
    parser.add_argument("--artifact-a", required=True)
    parser.add_argument("--artifact-b", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--side-a-label", default="artifact_a")
    parser.add_argument("--side-b-label", default="artifact_b")
    parser.add_argument("--representation", choices=("hidden", "logits"), required=True)
    parser.add_argument("--metric", choices=("linear_cka", "js_similarity", "topk_overlap"), required=True)
    parser.add_argument(
        "--alignment-mode",
        choices=("same_prefix_forcing", "teacher_forced_shared_continuation", "own_trajectory"),
        default="same_prefix_forcing",
    )
    parser.add_argument("--sample-mode", choices=("flatten_all_valid_positions",), default="flatten_all_valid_positions")
    parser.add_argument("--layer-mode", choices=("fixed_pairs", "pairwise_all"), default="pairwise_all")
    parser.add_argument("--readout-mode", choices=("raw", "unit_norm", "eps_norm", "model_norm"), default=None)
    parser.add_argument("--layer-indices-a", nargs="+", type=int, default=None)
    parser.add_argument("--layer-indices-b", nargs="+", type=int, default=None)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--num-permutations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--permutation-unit", choices=("auto", "row", "group"), default="auto")
    parser.add_argument("--multiple-testing-method", choices=("none", "fdr_bh", "holm"), default="fdr_bh")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_generation_similarity(
        args.artifact_a,
        args.artifact_b,
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
        prompt_index=args.prompt_index,
        prompt_text=args.prompt_text,
        num_permutations=args.num_permutations,
        alpha=args.alpha,
        top_k=args.top_k,
        permutation_unit=args.permutation_unit,
        multiple_testing_method=args.multiple_testing_method,
        seed=args.seed,
    )
    save_similarity_run_artifact(result, args.output_path)


__all__ = ["build_arg_parser", "main"]

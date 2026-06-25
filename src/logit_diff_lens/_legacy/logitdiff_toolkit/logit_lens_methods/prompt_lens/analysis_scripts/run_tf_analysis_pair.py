from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diffing.logit_lens_methods.prompt_lens.activation_dataset_analysis import (
    align_model_activation_datasets,
    align_model_activation_datasets_by_group,
    analyze_paired_activation_svd,
    summarize_paired_teacher_forced_divergence,
)


def run_analysis(
    *,
    left_path: Path,
    right_path: Path,
    output_dir: Path,
    condition: str,
    response_set: str,
    comparison_name: str,
    side_a_name: str,
    side_b_name: str,
    align_mode: str = "text",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    div_dir = output_dir / "divergence"
    svd_dir = output_dir / "svd"
    div_dir.mkdir(parents=True, exist_ok=True)
    svd_dir.mkdir(parents=True, exist_ok=True)

    if align_mode == "group":
        paired = align_model_activation_datasets_by_group(left_path, right_path)
    elif align_mode == "text":
        paired = align_model_activation_datasets(left_path, right_path)
    else:
        raise ValueError(f"Unsupported align_mode: {align_mode}")

    torch.save(paired, output_dir / "aligned_pair.pt")

    divergence_result = summarize_paired_teacher_forced_divergence(
        paired,
        output_dir=div_dir,
        top_k=5,
    )
    svd_result = analyze_paired_activation_svd(
        paired,
        output_dir=svd_dir,
        side_a_name=side_a_name,
        side_b_name=side_b_name,
        norm_mode="model_norm",
        top_components=8,
    )

    summary = {
        "comparison_name": comparison_name,
        "condition": condition,
        "response_set": response_set,
        "side_a_name": side_a_name,
        "side_b_name": side_b_name,
        "delta_definition": f"{side_b_name} - {side_a_name}",
        "align_mode": align_mode,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "aligned_pair_path": str(output_dir / "aligned_pair.pt"),
        "divergence": divergence_result,
        "svd": {
            "summary_path": str(svd_dir / "svd_summary.json"),
            "num_examples": svd_result["num_examples"],
            "num_layers": svd_result["num_layers"],
            "logit_source": svd_result["logit_source"],
            "norm_mode": svd_result["norm_mode"],
        },
    }
    (output_dir / "run_summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run paired TF divergence + SVD analysis")
    parser.add_argument("--left-path", required=True)
    parser.add_argument("--right-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--response-set", required=True)
    parser.add_argument("--comparison-name", required=True)
    parser.add_argument("--side-a-name", default="base")
    parser.add_argument("--side-b-name", default="finetuned")
    parser.add_argument("--align-mode", choices=("text", "group"), default="text")
    args = parser.parse_args()

    run_analysis(
        left_path=Path(args.left_path),
        right_path=Path(args.right_path),
        output_dir=Path(args.output_dir),
        condition=args.condition,
        response_set=args.response_set,
        comparison_name=args.comparison_name,
        side_a_name=args.side_a_name,
        side_b_name=args.side_b_name,
        align_mode=args.align_mode,
    )

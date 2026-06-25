from __future__ import annotations

import argparse
from pathlib import Path

import plotly.io as pio
import torch

from ..diffing import (
    compare_prompt_artifacts_ft_minus_base,
    load_prompt_decode_artifact,
    load_prompt_decode_artifact_bundle,
    save_comparison_artifact,
)
from ..plotting import plot_comparison_metric_heatmap


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare saved prompt capture artifacts with canonical ft-minus-base semantics and optionally plot the result."
    )
    parser.add_argument("--ft-artifact", required=True)
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--comparison-output", required=True)
    parser.add_argument("--readout-mode", choices=("raw", "model_norm"), default="model_norm")
    parser.add_argument("--metric", default="jsd_ft_base")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--title", default=None)
    return parser


def _load_first_prompt_artifact(path: str | Path):
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "artifacts" in payload:
        bundle = load_prompt_decode_artifact_bundle(path)
        if not bundle["artifacts"]:
            raise ValueError(f"No artifacts found in bundle: {path}")
        return bundle["artifacts"][0]
    return load_prompt_decode_artifact(path)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    ft_artifact = _load_first_prompt_artifact(args.ft_artifact)
    base_artifact = _load_first_prompt_artifact(args.base_artifact)
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
        output_path = Path(args.plot_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".html":
            fig.write_html(str(output_path))
        elif suffix == ".pdf":
            pio.write_image(fig, str(output_path), format="pdf")
        else:
            raise ValueError("--plot-output must end in .html or .pdf")


__all__ = ["build_arg_parser", "main"]

from __future__ import annotations

import argparse
from pathlib import Path

import plotly.io as pio

from logit_diff_lens.diffing import load_comparison_artifact
from logit_diff_lens.plotting import plot_comparison_metric_heatmap
from logit_diff_lens.plotting.prompt_heatmaps import (
    save_jaccard_heatmap_html,
    save_jaccard_heatmap_pdf,
    save_logitdiff_next_token_verification_html,
    save_logitdiff_next_token_verification_pdf,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Plotly prompt heatmap from a saved LogitDiff comparison artifact."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--plot-kind",
        choices=("jaccard", "next_token_verification", "comparison_metric"),
        default="jaccard",
    )
    parser.add_argument("--metric", default="jsd_ft_base")
    parser.add_argument("--title", default=None)
    parser.add_argument("--colorscale", default="RdBu")
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--display-top-tokens", type=int, default=10)
    parser.add_argument("--visible-cell-tokens", type=int, default=None)
    parser.add_argument("--max-divergent-layers", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--keep-last-layer-fraction", type=float, default=0.5)
    parser.add_argument("--exclude-prompt-tokens", action="store_true")
    parser.add_argument("--exclude-generated-tokens", action="store_true")
    parser.add_argument("--start-position", type=int, default=None)
    parser.add_argument("--end-position", type=int, default=None)
    parser.add_argument("--max-token-chars", type=int, default=18)
    parser.add_argument(
        "--layer-selection",
        choices=("all", "most_divergent", "least_divergent"),
        default="most_divergent",
    )
    parser.add_argument(
        "--x-tick-mode",
        choices=("prompt", "base_generated", "position"),
        default="base_generated",
    )
    parser.add_argument(
        "--x-tick-mode-secondary",
        choices=("prompt", "base_generated", "position", "none"),
        default="none",
    )
    parser.add_argument("--show-marginals", action="store_true")
    parser.add_argument("--analysis-topk", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.format or output_path.suffix.lower().lstrip(".")
    layer_limit = args.max_divergent_layers if args.max_divergent_layers is not None else args.max_layers
    token_limit = args.visible_cell_tokens if args.visible_cell_tokens is not None else args.display_top_tokens
    analysis_topk = args.analysis_topk if args.analysis_topk is not None else args.top_k

    if args.plot_kind == "jaccard":
        max_layers = layer_limit
        if args.layer_selection == "all":
            max_layers = None
        common_kwargs = {
            "prompt_index": args.prompt_index,
            "prompt_text": args.prompt_text,
            "include_prompt_tokens": not bool(args.exclude_prompt_tokens),
            "include_generated_tokens": not bool(args.exclude_generated_tokens),
            "start_idx": args.start_position,
            "end_idx": args.end_position,
            "title": args.title,
            "colorscale": args.colorscale,
            "display_top_tokens": token_limit,
            "visible_cell_tokens": args.visible_cell_tokens,
            "max_token_chars": args.max_token_chars,
            "show_marginals": bool(args.show_marginals),
            "max_layers": max_layers,
            "layer_selection": args.layer_selection,
            "analysis_topk": analysis_topk,
            "x_tick_mode": args.x_tick_mode,
        }
        if output_format == "html":
            save_jaccard_heatmap_html(args.input_path, output_path, **common_kwargs)
            return
        if output_format == "pdf":
            save_jaccard_heatmap_pdf(args.input_path, output_path, **common_kwargs)
            return
        raise ValueError("--output-path or --format must specify html or pdf")

    if args.plot_kind == "next_token_verification":
        common_kwargs = {
            "prompt_index": args.prompt_index,
            "prompt_text": args.prompt_text,
            "top_k": token_limit,
            "max_divergent_layers": 5 if layer_limit is None else layer_limit,
            "keep_last_layer_fraction": args.keep_last_layer_fraction,
            "include_prompt_tokens": not bool(args.exclude_prompt_tokens),
            "include_generated_tokens": not bool(args.exclude_generated_tokens),
            "start_idx": args.start_position,
            "end_idx": args.end_position,
            "max_token_chars": args.max_token_chars,
            "title": args.title,
            "colorscale": args.colorscale,
        }
        if output_format == "html":
            save_logitdiff_next_token_verification_html(args.input_path, output_path, **common_kwargs)
            return
        if output_format == "pdf":
            save_logitdiff_next_token_verification_pdf(args.input_path, output_path, **common_kwargs)
            return
        raise ValueError("--output-path or --format must specify html or pdf")

    comparison = load_comparison_artifact(args.input_path)
    fig = plot_comparison_metric_heatmap(
        comparison,
        metric_key=args.metric,
        title=args.title,
        colorscale=args.colorscale,
    )
    if output_format == "html":
        fig.write_html(str(output_path))
        return
    if output_format == "pdf":
        pio.write_image(fig, str(output_path), format="pdf")
        return
    raise ValueError("--output-path or --format must specify html or pdf")


if __name__ == "__main__":
    main()

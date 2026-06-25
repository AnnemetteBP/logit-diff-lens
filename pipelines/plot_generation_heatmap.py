from __future__ import annotations

import argparse

from logit_diff_lens.plotting.logitdiff_gen_plotter import (
    save_logitdiff_next_token_verification_html,
    save_logitdiff_next_token_verification_pdf,
    save_logitdiff_heatmap_html,
    save_logitdiff_heatmap_pdf,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Plotly generation heatmap from a saved LogitDiff generation-lens JSON payload."
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--format", choices=("html", "pdf"), default="html")
    parser.add_argument(
        "--plot-kind",
        choices=("jaccard", "next_token_verification"),
        default="jaccard",
    )
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--display-top-tokens", type=int, default=10)
    parser.add_argument("--visible-cell-tokens", type=int, default=None)
    parser.add_argument("--max-token-chars", type=int, default=18)
    parser.add_argument("--exclude-prompt-tokens", action="store_true")
    parser.add_argument("--exclude-generated-tokens", action="store_true")
    parser.add_argument("--start-position", type=int, default=None)
    parser.add_argument("--end-position", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--max-divergent-layers", type=int, default=None)
    parser.add_argument(
        "--layer-selection",
        choices=("all", "most_divergent", "least_divergent"),
        default="most_divergent",
    )
    parser.add_argument(
        "--x-tick-mode",
        choices=("input_tokens", "base_generated", "ft_generated", "base_top1", "ft_top1", "position"),
        default="ft_generated",
    )
    parser.add_argument(
        "--x-tick-mode-secondary",
        choices=("input_tokens", "base_generated", "ft_generated", "base_top1", "ft_top1", "position", "none"),
        default="base_generated",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--colorscale", default="RdBu")
    parser.add_argument("--show-marginals", action="store_true")
    parser.add_argument("--keep-last-layer-fraction", type=float, default=0.5)
    parser.add_argument("--analysis-topk", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    layer_limit = args.max_layers if args.max_layers is not None else args.max_divergent_layers
    token_limit = args.visible_cell_tokens if args.visible_cell_tokens is not None else args.display_top_tokens
    if args.top_k is not None and args.visible_cell_tokens is None:
        token_limit = args.top_k
    analysis_topk = args.analysis_topk if args.analysis_topk is not None else args.top_k
    common_kwargs = {
        "prompt_index": args.prompt_index,
        "prompt_text": args.prompt_text,
        "display_top_tokens": token_limit,
        "visible_cell_tokens": args.visible_cell_tokens,
        "max_token_chars": args.max_token_chars,
        "include_prompt_tokens": not bool(args.exclude_prompt_tokens),
        "include_generated_tokens": not bool(args.exclude_generated_tokens),
        "start_idx": args.start_position,
        "end_idx": args.end_position,
        "title": args.title,
        "colorscale": args.colorscale,
        "show_marginals": bool(args.show_marginals),
    }

    if args.plot_kind == "jaccard":
        common_kwargs.update(
            {
                "max_layers": layer_limit,
                "layer_selection": args.layer_selection,
                "analysis_topk": analysis_topk,
                "x_tick_mode": args.x_tick_mode,
                "x_tick_mode_secondary": None
                if args.x_tick_mode_secondary == "none"
                else args.x_tick_mode_secondary,
            }
        )
        if args.format == "pdf":
            save_logitdiff_heatmap_pdf(args.input_path, args.output_path, **common_kwargs)
            return
        save_logitdiff_heatmap_html(args.input_path, args.output_path, **common_kwargs)
        return

    common_kwargs.update(
        {
            "max_layers": 5 if layer_limit is None else layer_limit,
            "keep_last_layer_fraction": args.keep_last_layer_fraction,
        }
    )
    if args.format == "pdf":
        save_logitdiff_next_token_verification_pdf(args.input_path, args.output_path, **common_kwargs)
        return
    save_logitdiff_next_token_verification_html(args.input_path, args.output_path, **common_kwargs)


if __name__ == "__main__":
    main()

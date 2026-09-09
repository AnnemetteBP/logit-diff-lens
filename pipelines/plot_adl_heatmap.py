from __future__ import annotations

import argparse
from pathlib import Path

from logit_diff_lens.collectors.prompt import _format_generation_prompt
from logit_diff_lens.logit_lens.capture import _load_model_and_tokenizer
from logit_diff_lens.wrappers import LogitLensWrapper


METRIC_CHOICES = (
    "logit_max",
    "delta_norm",
    "ground_truth_probs",
    "entropy",
    "kl_div",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an ADL heatmap from a saved ADL payload or compute it live "
            "from two wrappers on one prompt."
        )
    )
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    parser.add_argument("--metric", choices=METRIC_CHOICES, default="logit_max")
    parser.add_argument("--title", default=None)
    parser.add_argument("--cmap", default=None)
    parser.add_argument("--norm-mode", choices=("raw", "unit_norm", "eps_norm", "model_norm"), default="model_norm")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--block-steps", type=int, default=1)
    parser.add_argument("--start-position", type=int, default=None)
    parser.add_argument("--end-position", type=int, default=None)
    parser.add_argument("--force-include-input", action="store_true", default=True)
    parser.add_argument("--no-force-include-input", dest="force_include_input", action="store_false")
    parser.add_argument("--force-include-output", action="store_true", default=True)
    parser.add_argument("--no-force-include-output", dest="force_include_output", action="store_false")
    parser.add_argument("--mark-correct-preds", action="store_true", default=True)
    parser.add_argument("--no-mark-correct-preds", dest="mark_correct_preds", action="store_false")
    parser.add_argument("--show-marginals", action="store_true")
    parser.add_argument("--fig-width", type=int, default=None)
    parser.add_argument("--fig-height", type=int, default=None)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--auto-vmin-vmax", action="store_true")
    parser.add_argument("--lm-head-a", action="store_true")
    parser.add_argument("--analyze-special-tokens", action="store_true")

    parser.add_argument("--model-name", required=True)
    parser.add_argument("--comparison-model-name", default=None)
    parser.add_argument("--comparison-adapter-path", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument(
        "--prompt-format",
        choices=("plain", "chat_template", "user_assistant_prefix"),
        default="plain",
    )
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--no-add-special-tokens", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stable-analysis", action="store_true", default=True)
    parser.add_argument("--no-stable-analysis", dest="stable_analysis", action="store_false")
    return parser


def _build_wrapper(
    *,
    model_name: str,
    tokenizer_name: str | None,
    adapter_path: str | None,
    dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
    load_in_4bit: bool,
    load_in_8bit: bool,
    debug: bool,
    stable_analysis: bool,
) -> LogitLensWrapper:
    model, tokenizer = _load_model_and_tokenizer(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        precision=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        adapter_path=adapter_path,
    )
    return LogitLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=debug,
        stable_analysis=stable_analysis,
    )


def _format_prompt(wrapper: LogitLensWrapper, args: argparse.Namespace) -> str:
    if args.prompt is None:
        raise ValueError("Live ADL plotting requires --prompt.")
    return _format_generation_prompt(
        wrapper,
        args.prompt,
        prompt_format=args.prompt_format,
        use_chat_template=bool(args.use_chat_template),
        system_prompt=args.system_prompt,
    )


def _metric_kwargs(metric: str) -> dict[str, bool]:
    return {name: name == metric for name in METRIC_CHOICES}


def _saved_mode_uses_live_adl_inputs(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.comparison_model_name,
            args.comparison_adapter_path,
            args.prompt,
            args.system_prompt,
            args.device_map,
        )
    ) or args.prompt_format != "plain" or any(
        bool(value)
        for value in (
            args.trust_remote_code,
            args.load_in_4bit,
            args.load_in_8bit,
            args.use_chat_template,
            args.no_add_special_tokens,
            args.debug,
            not args.stable_analysis,
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.input_path is not None and _saved_mode_uses_live_adl_inputs(args):
        raise ValueError("Saved ADL heatmap mode cannot be mixed with live comparison arguments.")
    from logit_diff_lens.plotting import save_adl_case_heatmap

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.format or output_path.suffix.lower().lstrip(".")
    if output_format not in {"html", "pdf"}:
        raise ValueError("--output-path or --format must specify html or pdf")

    wrapper_a = _build_wrapper(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        adapter_path=args.adapter_path,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.device_map,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        debug=bool(args.debug),
        stable_analysis=bool(args.stable_analysis),
    )

    source = args.input_path
    prompt = None
    if source is None:
        if not args.comparison_model_name and not args.comparison_adapter_path:
            raise ValueError(
                "Live ADL plotting requires --comparison-model-name or --comparison-adapter-path."
            )
        wrapper_b = _build_wrapper(
            model_name=args.comparison_model_name or args.model_name,
            tokenizer_name=args.tokenizer_name,
            adapter_path=args.comparison_adapter_path,
            dtype=args.dtype,
            trust_remote_code=bool(args.trust_remote_code),
            device_map=args.device_map,
            load_in_4bit=bool(args.load_in_4bit),
            load_in_8bit=bool(args.load_in_8bit),
            debug=bool(args.debug),
            stable_analysis=bool(args.stable_analysis),
        )
        source = (wrapper_a, wrapper_b)
        prompt = _format_prompt(wrapper_a, args)
    else:
        source = wrapper_a

    save_adl_case_heatmap(
        source,
        output_path,
        prompt=prompt,
        format=output_format,
        norm_mode=args.norm_mode,
        lm_head_A=bool(args.lm_head_a),
        add_special_tokens=not bool(args.no_add_special_tokens),
        analyze_special_tokens=bool(args.analyze_special_tokens),
        topk=args.top_k,
        force_include_input=bool(args.force_include_input),
        force_include_output=bool(args.force_include_output),
        mark_correct_preds=bool(args.mark_correct_preds),
        show_marginals=bool(args.show_marginals),
        block_steps=args.block_steps,
        start_idx=args.start_position,
        end_idx=args.end_position,
        cmap=args.cmap,
        title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        auto_vmin_vmax=bool(args.auto_vmin_vmax),
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        plot_data_from_file=args.input_path,
        **_metric_kwargs(args.metric),
    )


if __name__ == "__main__":
    main()

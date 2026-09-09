from __future__ import annotations

import argparse

from ..prisms import build_prompt_diff_prism_artifact, save_prompt_diff_prism_heatmap_artifact
from .prompt_analysis_utils import (
    add_prompt_runtime_args,
    build_prompt_wrapper,
    collect_prompt_artifacts_live,
    resolve_saved_pair,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a prompt diff prism artifact from saved prompt artifacts or from a live baseline/comparison prompt collection."
    )
    parser.add_argument("--ft-input-path", default=None)
    parser.add_argument("--base-input-path", default=None)
    parser.add_argument("--comparison-artifact", default=None)
    parser.add_argument("--base-artifact", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--readout-mode", choices=("raw", "model_norm"), default="raw")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--position-index", type=int, default=None)
    parser.add_argument(
        "--token-selection",
        choices=("largest_abs_delta", "largest_positive_delta", "largest_negative_delta"),
        default="largest_abs_delta",
    )
    parser.add_argument("--side-ft-label", default="ft")
    parser.add_argument("--side-base-label", default="base")
    add_prompt_runtime_args(parser, include_comparison=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    saved_pair = resolve_saved_pair(
        path_a=args.ft_input_path,
        path_b=args.base_input_path,
        base_artifact=args.base_artifact,
        comparison_artifact=args.comparison_artifact,
    )
    ft_source = None
    base_source = None
    if saved_pair is not None:
        if any(
            value is not None
            for value in (
                args.model_name,
                args.comparison_model_name,
                args.prompt,
                args.dataset_path,
                args.adapter_path,
                args.comparison_adapter_path,
                args.tokenizer_name,
            )
        ):
            raise ValueError("Saved prompt diff prism mode cannot be mixed with live prompt collection arguments.")
        ft_source, base_source = saved_pair
    else:
        if args.model_name is None:
            raise ValueError("Live prompt diff prism mode requires --model-name.")
        if not args.comparison_model_name and not args.comparison_adapter_path:
            raise ValueError(
                "Live prompt diff prism mode requires --comparison-model-name or --comparison-adapter-path."
            )
        base_wrapper = build_prompt_wrapper(
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
        comparison_wrapper = build_prompt_wrapper(
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
        base_artifacts = collect_prompt_artifacts_live(
            base_wrapper,
            prompt=args.prompt,
            dataset_path=args.dataset_path,
            text_field=args.text_field,
            use_chat_template=bool(args.use_chat_template),
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
            add_special_tokens=not bool(args.no_add_special_tokens),
            truncation=bool(args.truncate),
            max_length=args.max_length,
            force_include_input=bool(args.force_include_input),
            force_include_output=bool(args.force_include_output),
            normalize_embedding_for_readout=bool(args.normalize_embedding_for_readout),
            norm_modes=args.norm_modes,
            collect_components=bool(args.collect_components),
            project_component_logits=bool(args.project_component_logits),
            save_logits=bool(args.save_logits),
            tuned_lens_resource_id=args.tuned_lens_resource_id,
        )
        comparison_artifacts = collect_prompt_artifacts_live(
            comparison_wrapper,
            prompt=args.prompt,
            dataset_path=args.dataset_path,
            text_field=args.text_field,
            use_chat_template=bool(args.comparison_use_chat_template or args.use_chat_template),
            prompt_format=args.comparison_prompt_format or args.prompt_format,
            system_prompt=args.comparison_system_prompt if args.comparison_system_prompt is not None else args.system_prompt,
            add_special_tokens=not bool(args.no_add_special_tokens),
            truncation=bool(args.truncate),
            max_length=args.max_length,
            force_include_input=bool(args.force_include_input),
            force_include_output=bool(args.force_include_output),
            normalize_embedding_for_readout=bool(args.normalize_embedding_for_readout),
            norm_modes=args.norm_modes,
            collect_components=bool(args.collect_components),
            project_component_logits=bool(args.project_component_logits),
            save_logits=bool(args.save_logits),
            tuned_lens_resource_id=args.tuned_lens_resource_id,
        )
        if len(base_artifacts) != len(comparison_artifacts):
            raise ValueError("Live prompt diff prism requires baseline and comparison collections with the same number of prompts.")
        try:
            base_source = base_artifacts[args.prompt_index]
            ft_source = comparison_artifacts[args.prompt_index]
        except IndexError as exc:
            raise ValueError(
                f"prompt_index={args.prompt_index} out of range for {len(base_artifacts)} collected prompts"
            ) from exc
    artifact = build_prompt_diff_prism_artifact(
        ft_source,
        base_source,
        prompt_index=args.prompt_index,
        readout_mode=args.readout_mode,
        top_k=args.top_k,
        position_index=args.position_index,
        token_selection=args.token_selection,
        side_ft_label=args.side_ft_label,
        side_base_label=args.side_base_label,
    )
    save_prompt_diff_prism_heatmap_artifact(artifact, args.output_path)


__all__ = ["build_arg_parser", "main"]

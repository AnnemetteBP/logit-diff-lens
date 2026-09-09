from __future__ import annotations

import argparse

from ..prisms import build_prompt_prism_artifact, save_prompt_prism_artifact
from .prompt_analysis_utils import add_prompt_runtime_args, build_prompt_wrapper, collect_prompt_artifacts_live


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a package-owned prompt prism artifact from a saved prompt capture artifact or from a live prompt capture run."
    )
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--readout-mode", choices=("raw", "model_norm"), default="model_norm")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument(
        "--token-selection",
        choices=("largest_positive_logit", "largest_negative_logit", "largest_abs_logit"),
        default="largest_positive_logit",
    )
    add_prompt_runtime_args(parser, include_comparison=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    source = args.input_path
    if source is None:
        if args.model_name is None:
            raise ValueError("Live prompt prism mode requires --model-name.")
        wrapper = build_prompt_wrapper(
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
        artifacts = collect_prompt_artifacts_live(
            wrapper,
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
        try:
            source = artifacts[args.prompt_index]
        except IndexError as exc:
            raise ValueError(f"prompt_index={args.prompt_index} out of range for {len(artifacts)} collected prompts") from exc
    else:
        if any(
            value is not None
            for value in (args.model_name, args.prompt, args.dataset_path, args.adapter_path, args.tokenizer_name)
        ):
            raise ValueError("Saved prompt prism mode cannot be mixed with live prompt collection arguments.")
    artifact = build_prompt_prism_artifact(
        source,
        prompt_index=args.prompt_index,
        readout_mode=args.readout_mode,
        top_k=args.top_k,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        token_selection=args.token_selection,
    )
    save_prompt_prism_artifact(artifact, args.output_path)


__all__ = ["build_arg_parser", "main"]

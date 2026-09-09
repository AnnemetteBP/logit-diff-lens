from __future__ import annotations

import argparse

from ..collectors.patchscope import PatchscopePromptConfig, collect_patchscope_prompt_artifact
from ..diffing.io import load_prompt_decode_artifact, save_patchscope_prompt_artifact
from .runtime_args import add_stable_analysis_args
from ..wrappers import PatchingLensWrapper
from .capture import _load_model_and_tokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a prompt-first identity patchscope using a saved forward capture artifact as the source representation."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--target-prompt", required=True)
    parser.add_argument("--source-layer-index", type=int, required=True)
    parser.add_argument("--source-position", type=int, required=True)
    parser.add_argument("--target-layer-index", type=int, required=True)
    parser.add_argument("--target-position", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--readout-mode", choices=("raw", "model_norm"), default="model_norm")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    add_stable_analysis_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    source_artifact = load_prompt_decode_artifact(args.source_artifact)
    model, tokenizer = _load_model_and_tokenizer(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        precision=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.device_map,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        adapter_path=args.adapter_path,
    )
    wrapper = PatchingLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=bool(args.debug),
        stable_analysis=bool(args.stable_analysis),
    )
    artifact = collect_patchscope_prompt_artifact(
        wrapper,
        source_artifact,
        PatchscopePromptConfig(
            target_prompt=args.target_prompt,
            source_layer_index=args.source_layer_index,
            source_position=args.source_position,
            target_layer_index=args.target_layer_index,
            target_position=args.target_position,
            readout_mode=args.readout_mode,
            top_k=args.top_k,
        ),
    )
    save_patchscope_prompt_artifact(artifact, args.output_path)


__all__ = ["build_arg_parser", "main"]

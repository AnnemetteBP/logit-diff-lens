from __future__ import annotations

import argparse

from ..collectors.backward import BackwardLensCollectorConfig, collect_backward_prompt_artifact
from ..diffing.io import save_backward_prompt_artifact
from .runtime_args import add_stable_analysis_args
from ..wrappers import LogitLensWrapper
from .capture import _load_model_and_tokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a target-conditioned backward-pass artifact from a single prompt."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target-token-id", type=int, default=None)
    parser.add_argument("--target-token-text", default=None)
    parser.add_argument("--target-position", default="last")
    parser.add_argument("--output-path", required=True)
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
    parser.add_argument("--no-collect-attention-vjp", dest="collect_attention_vjp", action="store_false")
    parser.add_argument("--no-collect-mlp-vjp", dest="collect_mlp_vjp", action="store_false")
    parser.add_argument("--debug", action="store_true")
    add_stable_analysis_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if (args.target_token_id is None) == (args.target_token_text is None):
        raise ValueError("Provide exactly one of --target-token-id or --target-token-text.")

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
    wrapper = LogitLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=bool(args.debug),
        stable_analysis=bool(args.stable_analysis),
    )
    target_position = args.target_position
    if target_position != "last":
        target_position = int(target_position)
    artifact = collect_backward_prompt_artifact(
        wrapper,
        BackwardLensCollectorConfig(
            prompt=args.prompt,
            target_token_id=args.target_token_id,
            target_token_text=args.target_token_text,
            target_position=target_position,
            use_chat_template=bool(args.use_chat_template),
            prompt_format=args.prompt_format,
            system_prompt=args.system_prompt,
            add_special_tokens=not bool(args.no_add_special_tokens),
            collect_attention_vjp=bool(args.collect_attention_vjp),
            collect_mlp_vjp=bool(args.collect_mlp_vjp),
        ),
    )
    save_backward_prompt_artifact(artifact, args.output_path)


__all__ = ["build_arg_parser", "main"]

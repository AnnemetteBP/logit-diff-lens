from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an architecture-matched random model by instantiating a "
            "causal LM directly from its pretrained config."
        )
    )
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))

    config = AutoConfig.from_pretrained(
        args.base_model_name,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=bool(args.trust_remote_code),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.base_model_name,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

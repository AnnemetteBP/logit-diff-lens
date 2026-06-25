from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from logit_diff_lens.collectors.generation import (
    GenerationActivationCollectorConfig,
    collect_generation_activation_dataset_incremental,
    collect_generation_activations,
)
from logit_diff_lens.logit_lens.capture import _resolve_torch_dtype
from logit_diff_lens.wrappers import CustomGenerationLensWrapper, GenerateLensWrapper


def _load_model_and_tokenizer(
    *,
    model_name: str,
    tokenizer_name: str | None,
    precision: str,
    trust_remote_code: bool,
    device_map: str | None,
    load_in_4bit: bool,
    load_in_8bit: bool,
    adapter_path: str | None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    torch_dtype = _resolve_torch_dtype(precision)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    if device_map:
        model_kwargs["device_map"] = device_map
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture generation-lens artifacts for a single prompt or a dataset."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--text-field", default="analysis_text")
    parser.add_argument("--label-field", default="label")
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
    parser.add_argument("--analyze-special-tokens", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--padding",
        choices=("auto", "longest", "max_length", "do_not_pad"),
        default="auto",
    )
    parser.add_argument("--force-include-input", action="store_true", default=True)
    parser.add_argument("--no-force-include-input", dest="force_include_input", action="store_false")
    parser.add_argument("--force-include-output", action="store_true", default=True)
    parser.add_argument("--no-force-include-output", dest="force_include_output", action="store_false")
    parser.add_argument(
        "--norm-modes",
        nargs="+",
        default=("raw", "unit_norm", "eps_norm", "model_norm"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--collect-components", action="store_true")
    parser.add_argument("--project-component-logits", action="store_true")
    parser.add_argument("--custom-generate", action="store_true")
    return parser


def _resolve_padding(value: str) -> bool | str | None:
    if value == "auto":
        return None
    if value == "do_not_pad":
        return False
    return value


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if bool(args.prompt) == bool(args.dataset_path):
        raise ValueError("Provide exactly one of --prompt or --dataset-path.")
    if args.project_component_logits and not args.collect_components:
        raise ValueError("--project-component-logits requires --collect-components.")

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
    wrapper_cls = CustomGenerationLensWrapper if args.custom_generate else GenerateLensWrapper
    wrapper = wrapper_cls(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=False,
        stable_analysis=True,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.prompt is not None:
        payload = collect_generation_activations(
            wrapper,
            GenerationActivationCollectorConfig(
                prompt=args.prompt,
                use_chat_template=bool(args.use_chat_template),
                prompt_format=args.prompt_format,
                system_prompt=args.system_prompt,
                add_special_tokens=not bool(args.no_add_special_tokens),
                analyze_special_tokens=bool(args.analyze_special_tokens),
                truncation=bool(args.truncate),
                max_length=args.max_length,
                padding=_resolve_padding(args.padding),
                force_include_input=bool(args.force_include_input),
                force_include_output=bool(args.force_include_output),
                norm_modes=tuple(args.norm_modes),
                collect_components=bool(args.collect_components),
                project_component_logits=bool(args.project_component_logits),
                max_new_tokens=int(args.max_new_tokens),
            ),
        )
        torch.save(payload, output_path)
        return

    payload = collect_generation_activation_dataset_incremental(
        wrapper=wrapper,
        dataset_path=args.dataset_path,
        output_path=output_path,
        text_field=args.text_field,
        label_field=args.label_field,
        use_chat_template=bool(args.use_chat_template),
        prompt_format=args.prompt_format,
        system_prompt=args.system_prompt,
        add_special_tokens=not bool(args.no_add_special_tokens),
        analyze_special_tokens=bool(args.analyze_special_tokens),
        truncation=bool(args.truncate),
        max_length=args.max_length,
        padding=_resolve_padding(args.padding),
        force_include_input=bool(args.force_include_input),
        force_include_output=bool(args.force_include_output),
        norm_modes=tuple(args.norm_modes),
        collect_components=bool(args.collect_components),
        project_component_logits=bool(args.project_component_logits),
        max_new_tokens=int(args.max_new_tokens),
        batch_size=int(args.batch_size),
    )
    torch.save(payload, output_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..collectors.prompt import (
    PromptLensActivationCollectorConfig,
    _build_collection_text_and_kind,
    _load_tuned_lens,
    collect_prompt_lens_activations,
)
from ..diffing.io import save_prompt_decode_artifact, save_prompt_decode_artifact_bundle
from ..wrappers import LogitLensWrapper
from .training import parse_precision


def _resolve_torch_dtype(precision: str) -> torch.dtype | str:
    normalized = parse_precision(precision)
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    if normalized == "auto":
        return "auto"
    if normalized == "int8":
        return torch.float16
    raise ValueError(f"Unsupported precision: {precision}")


def _resolve_padding(value: str | None) -> bool | str | None:
    if value in (None, "auto"):
        return None
    if value == "do_not_pad":
        return False
    return value


def _load_model_and_tokenizer(
    *,
    model_name: str,
    tokenizer_name: str | None,
    model_revision: str | None,
    precision: str,
    trust_remote_code: bool,
    device_map: str | None,
    load_in_4bit: bool,
    load_in_8bit: bool,
    adapter_path: str | None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or model_name,
        revision=model_revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if model_revision:
        model_kwargs["revision"] = model_revision
    torch_dtype = _resolve_torch_dtype(precision)
    if torch_dtype != "auto":
        model_kwargs["dtype"] = torch_dtype
    if device_map:
        model_kwargs["device_map"] = device_map
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture canonical prompt hidden-state artifacts for reuse across downstream lens analyses."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--label-field", default=None)
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
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--padding",
        choices=("auto", "longest", "max_length", "do_not_pad"),
        default="auto",
    )
    parser.add_argument("--force-include-input", action="store_true", default=True)
    parser.add_argument("--no-force-include-input", dest="force_include_input", action="store_false")
    parser.add_argument("--force-include-output", action="store_true")
    parser.add_argument("--normalize-embedding-for-readout", action="store_true")
    parser.add_argument(
        "--norm-modes",
        nargs="+",
        default=("raw", "model_norm"),
    )
    parser.add_argument("--collect-components", action="store_true")
    parser.add_argument("--project-component-logits", action="store_true")
    parser.add_argument("--tuned-lens-resource-id", default=None)
    parser.add_argument("--save-logits", action="store_true", default=True)
    parser.add_argument("--no-save-logits", dest="save_logits", action="store_false")
    parser.add_argument("--stable-analysis", action="store_true", default=True)
    parser.add_argument("--no-stable-analysis", dest="stable_analysis", action="store_false")
    parser.add_argument("--debug", action="store_true")
    return parser


def _build_collector_config_from_args(args: argparse.Namespace, prompt: str) -> PromptLensActivationCollectorConfig:
    return PromptLensActivationCollectorConfig(
        prompt=prompt,
        use_chat_template=bool(args.use_chat_template),
        prompt_format=args.prompt_format,
        system_prompt=args.system_prompt,
        add_special_tokens=not bool(args.no_add_special_tokens),
        truncation=bool(args.truncate),
        max_length=args.max_length,
        padding=_resolve_padding(args.padding),
        force_include_input=bool(args.force_include_input),
        force_include_output=bool(args.force_include_output),
        normalize_embedding_for_readout=bool(args.normalize_embedding_for_readout),
        norm_modes=tuple(args.norm_modes),
        collect_components=bool(args.collect_components),
        project_component_logits=bool(args.project_component_logits),
        save_logits=bool(args.save_logits),
        tuned_lens_resource_id=args.tuned_lens_resource_id,
    )


def _capture_single_prompt(
    wrapper: LogitLensWrapper,
    args: argparse.Namespace,
) -> None:
    config = _build_collector_config_from_args(args, args.prompt)
    tuned_lens = None
    if args.tuned_lens_resource_id is not None:
        tuned_lens = _load_tuned_lens(wrapper, resource_id=args.tuned_lens_resource_id)
    artifact = collect_prompt_lens_activations(wrapper, config, tuned_lens=tuned_lens)["artifact"]
    save_prompt_decode_artifact(artifact, args.output_path)


def _capture_dataset(
    wrapper: LogitLensWrapper,
    args: argparse.Namespace,
) -> None:
    dataset_path = Path(args.dataset_path)
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    artifacts = []
    tuned_lens = None
    if args.tuned_lens_resource_id is not None:
        tuned_lens = _load_tuned_lens(wrapper, resource_id=args.tuned_lens_resource_id)
    for row in rows:
        prompt_text, continuation_kind = _build_collection_text_and_kind(
            row,
            text_field=args.text_field,
        )
        config = _build_collector_config_from_args(args, prompt_text)
        artifact = collect_prompt_lens_activations(wrapper, config, tuned_lens=tuned_lens)["artifact"]
        artifact.prompt_id = str(row.get("id")) if row.get("id") is not None else None
        artifact.metadata.update(
            {
                "row_id": row.get("id"),
                "group_id": row.get("group_id"),
                "variant": row.get("variant"),
                "language": row.get("language"),
                "label": row.get(args.label_field) if args.label_field else None,
                "continuation_kind": continuation_kind,
                "dataset_path": str(dataset_path),
                "text_field": args.text_field,
            }
        )
        artifacts.append(artifact)

    save_prompt_decode_artifact_bundle(
        artifacts,
        args.output_path,
        metadata={
            "dataset_path": str(dataset_path),
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name or args.model_name,
            "adapter_path": args.adapter_path,
            "model_revision": args.model_revision,
            "dtype": args.dtype,
            "trust_remote_code": bool(args.trust_remote_code),
            "device_map": args.device_map,
            "load_in_4bit": bool(args.load_in_4bit),
            "load_in_8bit": bool(args.load_in_8bit),
            "stable_analysis": bool(args.stable_analysis),
            "text_field": args.text_field,
            "label_field": args.label_field,
            "use_chat_template": bool(args.use_chat_template),
            "prompt_format": args.prompt_format,
            "system_prompt": args.system_prompt,
            "add_special_tokens": not bool(args.no_add_special_tokens),
            "truncation": bool(args.truncate),
            "max_length": args.max_length,
            "padding": _resolve_padding(args.padding),
            "force_include_input": bool(args.force_include_input),
            "force_include_output": bool(args.force_include_output),
            "normalize_embedding_for_readout": bool(args.normalize_embedding_for_readout),
            "norm_modes": list(args.norm_modes),
            "collect_components": bool(args.collect_components),
            "project_component_logits": bool(args.project_component_logits),
            "save_logits": bool(args.save_logits),
            "tuned_lens_resource_id": args.tuned_lens_resource_id,
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if bool(args.prompt) == bool(args.dataset_path):
        raise ValueError("Provide exactly one of --prompt or --dataset-path.")
    if args.project_component_logits and not args.collect_components:
        raise ValueError("--project-component-logits requires --collect-components.")
    if args.tuned_lens_resource_id and not args.save_logits:
        raise ValueError("--tuned-lens-resource-id requires logits to be saved; remove --no-save-logits.")

    model, tokenizer = _load_model_and_tokenizer(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        model_revision=args.model_revision,
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

    if args.prompt is not None:
        _capture_single_prompt(wrapper, args)
    else:
        _capture_dataset(wrapper, args)


if __name__ == "__main__":
    main()


__all__ = ["build_arg_parser", "main"]

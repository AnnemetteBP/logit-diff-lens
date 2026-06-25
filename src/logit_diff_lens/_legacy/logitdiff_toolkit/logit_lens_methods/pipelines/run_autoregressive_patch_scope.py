from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from diffing.logit_lens_methods.tokenizer_loading import load_tokenizer
from diffing.logit_lens_methods.wrapper.lens_wrappers.logit_lens_wrapper import LogitLensWrapper
from diffing.logit_lens_methods.wrapper.lens_wrappers.patching_lens_wrapper import PatchingLensWrapper


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return mapping[dtype_name]


def _clear_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_quantization_config(
    *,
    config_name: str | None,
    config_source: str | None,
) -> Any | None:
    if not config_name:
        return None
    if not config_source:
        raise ValueError("quantization config source is required when config name is set")
    source_path = Path(config_source)
    if not source_path.exists():
        raise FileNotFoundError(f"quantization config source not found: {source_path}")
    spec = importlib.util.spec_from_file_location("patchscope_quant_config", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load quantization config source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, config_name):
        raise AttributeError(f"{source_path} does not define {config_name}")
    return getattr(module, config_name)


def _load_model(
    *,
    model_id: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    force_cpu: bool,
    adapter_path: str | None = None,
    quantization_config: Any | None = None,
    device_map_override: Any = None,
):
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "device_map": device_map_override if device_map_override is not None else ("cpu" if force_cpu else "auto"),
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def _topk_tokens(logits: torch.Tensor, tokenizer, k: int = 5) -> list[dict[str, Any]]:
    probs = torch.softmax(logits.float(), dim=-1)
    vals, idxs = torch.topk(probs, k=k)
    out = []
    for prob, tok_id in zip(vals.tolist(), idxs.tolist()):
        out.append(
            {
                "token_id": int(tok_id),
                "token_str": tokenizer.decode([tok_id]),
                "prob": float(prob),
            }
        )
    return out


def _build_prompt_text(
    *,
    tokenizer,
    prompt: str,
    system_prompt: str | None,
    use_chat_template: bool,
    chat_template_path: str | None,
) -> str:
    if chat_template_path:
        tokenizer.chat_template = Path(chat_template_path).read_text(encoding="utf-8")

    if not use_chat_template:
        return prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _first_reversion_layer(records: list[dict[str, Any]]) -> int | None:
    for record in records:
        if record["reverted_to_base_top1"]:
            return int(record["layer_idx"])
    return None


def run_autoregressive_patch_scope(
    *,
    base_model_id: str,
    adapter_path: str | None = None,
    comparison_model_id: str | None = None,
    prompt: str,
    output_path: Path,
    tokenizer_id: str | None = None,
    system_prompt: str | None = None,
    use_chat_template: bool = False,
    chat_template_path: str | None = None,
    dtype_name: str = "bfloat16",
    trust_remote_code: bool = False,
    force_cpu: bool = False,
    base_force_cpu: bool | None = None,
    comparison_force_cpu: bool | None = None,
    comparison_force_single_gpu: bool = False,
    num_generated_positions: int = 4,
    base_quantization_config_name: str | None = None,
    base_quantization_config_source: str | None = None,
    comparison_quantization_config_name: str | None = None,
    comparison_quantization_config_source: str | None = None,
) -> dict[str, Any]:
    _clear_cache()
    dtype = _resolve_dtype(dtype_name)
    tokenizer = load_tokenizer(tokenizer_id or base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_quantization_config = _load_quantization_config(
        config_name=base_quantization_config_name,
        config_source=base_quantization_config_source,
    )
    comparison_quantization_config = _load_quantization_config(
        config_name=comparison_quantization_config_name,
        config_source=comparison_quantization_config_source,
    )

    prompt_text = _build_prompt_text(
        tokenizer=tokenizer,
        prompt=prompt,
        system_prompt=system_prompt,
        use_chat_template=use_chat_template,
        chat_template_path=chat_template_path,
    )

    base_model = _load_model(
        model_id=base_model_id,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        force_cpu=force_cpu if base_force_cpu is None else base_force_cpu,
        quantization_config=base_quantization_config,
    )
    ft_model = _load_model(
        model_id=comparison_model_id or base_model_id,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        force_cpu=force_cpu if comparison_force_cpu is None else comparison_force_cpu,
        adapter_path=adapter_path,
        quantization_config=comparison_quantization_config,
        device_map_override={"": 0} if comparison_force_single_gpu else None,
    )

    base_wrapper = LogitLensWrapper(
        model=base_model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=False,
        stable_analysis=True,
    )
    ft_wrapper = PatchingLensWrapper(
        model=ft_model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=False,
        stable_analysis=True,
        patch_config=None,
    )

    tokenized = base_wrapper.tokenize_inputs(
        prompt_text,
        add_special_tokens=not use_chat_template,
    )
    current_input_ids = tokenized["input_ids"].clone()
    current_attention_mask = tokenized["attention_mask"].clone()

    position_sweeps: list[dict[str, Any]] = []

    for generated_pos in range(num_generated_positions):
        patch_token_idx = int(current_attention_mask[0].sum().item()) - 1
        ft_input_ids = current_input_ids.to(ft_wrapper.model_device)
        ft_attention_mask = current_attention_mask.to(ft_wrapper.model_device)

        base_acts, base_outputs = base_wrapper.forward_pass(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
        )
        _, ft_outputs = ft_wrapper.forward_pass(
            input_ids=ft_input_ids,
            attention_mask=ft_attention_mask,
        )

        base_next_logits = base_outputs.logits[0, patch_token_idx, :].detach().cpu()
        ft_next_logits = ft_outputs.logits[0, patch_token_idx, :].detach().cpu()

        base_top1_id = int(torch.argmax(base_next_logits).item())
        ft_top1_id = int(torch.argmax(ft_next_logits).item())

        layer_names = [
            name for name in base_acts.keys() if name == "embedding" or name.startswith("layer_")
        ]
        layer_sweep: list[dict[str, Any]] = []
        for layer_name in layer_names:
            base_hidden = base_acts[layer_name]
            layer_idx = -1 if layer_name == "embedding" else int(layer_name.split("_")[1])
            patch_tensor = base_hidden[:, patch_token_idx : patch_token_idx + 1, :].clone()
            ft_wrapper.set_patch_config(
                {
                    "layer_idx": layer_idx,
                    "mode": "replace",
                    "alpha": 1.0,
                    "batch_idx": slice(0, 1),
                    "token_idx": slice(patch_token_idx, patch_token_idx + 1),
                    "tensor": patch_tensor,
                }
            )
            _, patched_outputs = ft_wrapper.forward_pass(
                input_ids=ft_input_ids,
                attention_mask=ft_attention_mask,
            )
            patched_logits = patched_outputs.logits[0, patch_token_idx, :].detach().cpu()
            patched_top1_id = int(torch.argmax(patched_logits).item())
            layer_sweep.append(
                {
                    "layer_name": layer_name,
                    "layer_idx": layer_idx,
                    "patched_top1_id": patched_top1_id,
                    "patched_top1_token": tokenizer.decode([patched_top1_id]),
                    "patched_base_token_logit": float(patched_logits[base_top1_id].item()),
                    "patched_ft_token_logit": float(patched_logits[ft_top1_id].item()),
                    "patched_base_token_rank": int(
                        (patched_logits > patched_logits[base_top1_id]).sum().item() + 1
                    ),
                    "patched_ft_token_rank": int(
                        (patched_logits > patched_logits[ft_top1_id]).sum().item() + 1
                    ),
                    "patched_top5": _topk_tokens(patched_logits, tokenizer, k=5),
                    "reverted_to_base_top1": bool(patched_top1_id == base_top1_id),
                }
            )

        ft_wrapper.clear_patch_config()

        position_sweeps.append(
            {
                "generated_position": generated_pos,
                "patch_token_idx": patch_token_idx,
                "context_tokens": tokenizer.convert_ids_to_tokens(current_input_ids[0].tolist()),
                "context_text": tokenizer.decode(current_input_ids[0].tolist()),
                "base_top1": {
                    "token_id": base_top1_id,
                    "token_str": tokenizer.decode([base_top1_id]),
                    "top5": _topk_tokens(base_next_logits, tokenizer, k=5),
                },
                "ft_top1": {
                    "token_id": ft_top1_id,
                    "token_str": tokenizer.decode([ft_top1_id]),
                    "top5": _topk_tokens(ft_next_logits, tokenizer, k=5),
                },
                "first_reversion_layer_idx": _first_reversion_layer(layer_sweep),
                "num_reverted_layers": int(sum(1 for row in layer_sweep if row["reverted_to_base_top1"])),
                "layer_sweep": layer_sweep,
            }
        )

        next_token = torch.tensor([[ft_top1_id]], device=current_input_ids.device, dtype=current_input_ids.dtype)
        next_mask = torch.ones((1, 1), device=current_attention_mask.device, dtype=current_attention_mask.dtype)
        current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, next_mask], dim=1)

    result = {
        "prompt": prompt,
        "formatted_prompt": prompt_text,
        "base_model_id": base_model_id,
        "adapter_path": adapter_path,
        "comparison_model_id": comparison_model_id,
        "dtype": dtype_name,
        "force_cpu": force_cpu,
        "base_force_cpu": force_cpu if base_force_cpu is None else base_force_cpu,
        "comparison_force_cpu": force_cpu if comparison_force_cpu is None else comparison_force_cpu,
        "comparison_force_single_gpu": comparison_force_single_gpu,
        "base_quantization_config_name": base_quantization_config_name,
        "base_quantization_config_source": base_quantization_config_source,
        "comparison_quantization_config_name": comparison_quantization_config_name,
        "comparison_quantization_config_source": comparison_quantization_config_source,
        "num_generated_positions": num_generated_positions,
        "position_sweeps": position_sweeps,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an autoregressive multi-position base->FT patch scope")
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--comparison-model-id", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--chat-template-path", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--base-force-cpu", action="store_true")
    parser.add_argument("--comparison-force-cpu", action="store_true")
    parser.add_argument("--comparison-force-single-gpu", action="store_true")
    parser.add_argument("--num-generated-positions", type=int, default=4)
    parser.add_argument("--base-quantization-config-name", default=None)
    parser.add_argument("--base-quantization-config-source", default=None)
    parser.add_argument("--comparison-quantization-config-name", default=None)
    parser.add_argument("--comparison-quantization-config-source", default=None)
    args = parser.parse_args()

    run_autoregressive_patch_scope(
        base_model_id=args.base_model_id,
        adapter_path=args.adapter_path,
        comparison_model_id=args.comparison_model_id,
        prompt=args.prompt,
        output_path=args.output_path,
        tokenizer_id=args.tokenizer_id,
        system_prompt=args.system_prompt,
        use_chat_template=args.use_chat_template,
        chat_template_path=args.chat_template_path,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        force_cpu=args.force_cpu,
        base_force_cpu=True if args.base_force_cpu else None,
        comparison_force_cpu=True if args.comparison_force_cpu else None,
        comparison_force_single_gpu=args.comparison_force_single_gpu,
        num_generated_positions=args.num_generated_positions,
        base_quantization_config_name=args.base_quantization_config_name,
        base_quantization_config_source=args.base_quantization_config_source,
        comparison_quantization_config_name=args.comparison_quantization_config_name,
        comparison_quantization_config_source=args.comparison_quantization_config_source,
    )


if __name__ == "__main__":
    main()

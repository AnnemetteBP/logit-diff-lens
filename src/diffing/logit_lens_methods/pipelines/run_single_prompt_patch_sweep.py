from __future__ import annotations

import argparse
import gc
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


def _load_model(
    *,
    model_id: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    force_cpu: bool,
    adapter_path: str | None = None,
):
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "device_map": "cpu" if force_cpu else "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def _topk_tokens(
    logits: torch.Tensor,
    tokenizer,
    k: int = 5,
) -> list[dict[str, Any]]:
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


def run_patch_sweep(
    *,
    base_model_id: str,
    adapter_path: str,
    prompt: str,
    output_path: Path,
    tokenizer_id: str | None = None,
    system_prompt: str | None = None,
    use_chat_template: bool = False,
    chat_template_path: str | None = None,
    dtype_name: str = "bfloat16",
    trust_remote_code: bool = False,
    force_cpu: bool = False,
) -> dict[str, Any]:
    _clear_cache()
    dtype = _resolve_dtype(dtype_name)
    tokenizer = load_tokenizer(tokenizer_id or base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if chat_template_path:
        tokenizer.chat_template = Path(chat_template_path).read_text(encoding="utf-8")

    base_model = _load_model(
        model_id=base_model_id,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        force_cpu=force_cpu,
    )
    ft_model = _load_model(
        model_id=base_model_id,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        force_cpu=force_cpu,
        adapter_path=adapter_path,
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

    prompt_text = prompt
    if use_chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    tokenized = base_wrapper.tokenize_inputs(prompt_text, add_special_tokens=not use_chat_template)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    patch_token_idx = int(attention_mask[0].sum().item()) - 1

    base_acts, base_outputs = base_wrapper.forward_pass(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    _, ft_outputs = ft_wrapper.forward_pass(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    base_next_logits = base_outputs.logits[0, patch_token_idx, :].detach().cpu()
    ft_next_logits = ft_outputs.logits[0, patch_token_idx, :].detach().cpu()

    base_top1_id = int(torch.argmax(base_next_logits).item())
    ft_top1_id = int(torch.argmax(ft_next_logits).item())
    base_top1_token = tokenizer.decode([base_top1_id])
    ft_top1_token = tokenizer.decode([ft_top1_id])

    sweep_results: list[dict[str, Any]] = []
    layer_names = [name for name in base_acts.keys() if name == "embedding" or name.startswith("layer_")]

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
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        patched_logits = patched_outputs.logits[0, patch_token_idx, :].detach().cpu()
        patched_top1_id = int(torch.argmax(patched_logits).item())

        sweep_results.append(
            {
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "patched_top1_id": patched_top1_id,
                "patched_top1_token": tokenizer.decode([patched_top1_id]),
                "base_top1_logit": float(base_next_logits[base_top1_id].item()),
                "ft_top1_logit": float(ft_next_logits[ft_top1_id].item()),
                "patched_base_token_logit": float(patched_logits[base_top1_id].item()),
                "patched_ft_token_logit": float(patched_logits[ft_top1_id].item()),
                "patched_base_token_rank": int((patched_logits > patched_logits[base_top1_id]).sum().item() + 1),
                "patched_ft_token_rank": int((patched_logits > patched_logits[ft_top1_id]).sum().item() + 1),
                "patched_top5": _topk_tokens(patched_logits, tokenizer, k=5),
                "reverted_to_base_top1": bool(patched_top1_id == base_top1_id),
            }
        )

    ft_wrapper.clear_patch_config()

    result = {
        "prompt": prompt,
        "formatted_prompt": prompt_text,
        "patch_token_idx": patch_token_idx,
        "prompt_tokens": tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
        "base_model_id": base_model_id,
        "adapter_path": adapter_path,
        "dtype": dtype_name,
        "force_cpu": force_cpu,
        "base_top1": {
            "token_id": base_top1_id,
            "token_str": base_top1_token,
            "top5": _topk_tokens(base_next_logits, tokenizer, k=5),
        },
        "ft_top1": {
            "token_id": ft_top1_id,
            "token_str": ft_top1_token,
            "top5": _topk_tokens(ft_next_logits, tokenizer, k=5),
        },
        "layer_sweep": sweep_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-prompt base->FT hidden-state patch sweep")
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--chat-template-path", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    result = run_patch_sweep(
        base_model_id=args.base_model_id,
        adapter_path=args.adapter_path,
        prompt=args.prompt,
        output_path=args.output_path,
        tokenizer_id=args.tokenizer_id,
        system_prompt=args.system_prompt,
        use_chat_template=args.use_chat_template,
        chat_template_path=args.chat_template_path,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        force_cpu=args.force_cpu,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

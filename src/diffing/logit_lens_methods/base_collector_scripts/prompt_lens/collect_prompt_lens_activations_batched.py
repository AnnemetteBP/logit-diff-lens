from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import torch
from tqdm.auto import tqdm

from ...wrapper import (
    LogitLensWrapper,
    normalize_activations,
    lmhead_project,
    resolve_block_component_module,
)
from ...logitdiff_gen.core import _format_generation_prompt


@dataclass
class PromptLensActivationCollectorConfig:
    prompt: str = ""
    use_chat_template: bool = False
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain"
    system_prompt: str | None = None
    add_special_tokens: bool = True
    force_include_input: bool = True
    force_include_output: bool = False
    norm_modes: tuple[str, ...] = ("raw", "model_norm")
    collect_components: bool = False
    project_component_logits: bool = False
    save_logits: bool = True


def _detach_full_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def _build_collection_text_and_kind(
    row: Dict[str, Any],
    *,
    text_field: str,
) -> tuple[str, str]:
    source_kind = str(row.get("source_kind", ""))
    model_role = str(row.get("model_role", ""))
    prompt = str(row.get("prompt_clean") or row.get("prompt") or row.get("source_prompt") or "").strip()
    response_only = row.get("response_only") or row.get("response_text")

    if source_kind == "model_response" and response_only is not None:
        response = str(response_only).strip()
        if prompt and response:
            combined = f"{prompt} {response}"
        else:
            combined = response or prompt
        if model_role == "base" or row.get("variant") == "base_response":
            continuation_kind = "prompt_plus_base_response"
        elif model_role == "finetuned" or row.get("variant") == "finetuned_response":
            continuation_kind = "prompt_plus_finetuned_response"
        else:
            continuation_kind = "prompt_plus_response"
        return combined, continuation_kind

    text_value = str(row.get(text_field, "")).strip()
    if text_value:
        variant = str(row.get("variant", "")).strip()
        if variant == "base_response":
            return text_value, "prompt_plus_base_response"
        if variant == "finetuned_response":
            return text_value, "prompt_plus_finetuned_response"
        return text_value, "prompt_only"
    if prompt:
        return prompt, "prompt_only"
    raise ValueError(
        f"Could not determine collection text from row id={row.get('id')} using text_field='{text_field}'"
    )


def _collect_layer_records(
    wrapper: LogitLensWrapper,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden_states: Sequence[torch.Tensor],
    attention_outputs: Sequence[torch.Tensor] | None,
    mlp_outputs: Sequence[torch.Tensor] | None,
    attention_logits_by_mode: Dict[str, List[torch.Tensor]] | None,
    mlp_logits_by_mode: Dict[str, List[torch.Tensor]] | None,
    config: PromptLensActivationCollectorConfig,
) -> List[Dict[str, Any]]:
    seq_len = int(attention_mask[0].sum().item())
    hidden_seq = [tensor[:, :seq_len, :] for tensor in hidden_states]
    tokens = _detach_full_tensor_to_cpu(input_ids[:, :seq_len])
    records: List[Dict[str, Any]] = []

    if config.force_include_input:
        hidden_full = hidden_seq[0]
        rec: Dict[str, Any] = {
            "layer_index": -1,
            "layer_name": "embedding",
            "tokens": tokens,
            "attention_mask": _detach_full_tensor_to_cpu(attention_mask[:, :seq_len]),
            "hidden": _detach_full_tensor_to_cpu(hidden_full),
        }
        if config.save_logits:
            for mode in config.norm_modes:
                h_norm = normalize_activations(
                    x=hidden_full.clone(),
                    mode=mode,
                    block="embedding",
                    layer_index=-1,
                    model_device=wrapper.model_device,
                    model_dtype=wrapper.model_dtype,
                    final_norm=wrapper.final_norm,
                )
                logits, _ = lmhead_project(
                    x=h_norm,
                    lm_head=wrapper.lm_head,
                    stable=wrapper.stable,
                    model_device=wrapper.model_device,
                )
                rec[f"logits_{mode}"] = _detach_full_tensor_to_cpu(logits)
        records.append(rec)

    num_blocks = len(wrapper.blocks)
    for idx in range(num_blocks):
        hidden_full = hidden_seq[idx + 1]
        rec = {
            "layer_index": idx,
            "layer_name": f"layer_{idx}",
            "tokens": tokens,
            "attention_mask": _detach_full_tensor_to_cpu(attention_mask[:, :seq_len]),
            "hidden": _detach_full_tensor_to_cpu(hidden_full),
        }
        if config.save_logits:
            for mode in config.norm_modes:
                h_norm = normalize_activations(
                    x=hidden_full.clone(),
                    mode=mode,
                    block="block",
                    layer_index=idx,
                    model_device=wrapper.model_device,
                    model_dtype=wrapper.model_dtype,
                    final_norm=wrapper.final_norm,
                )
                logits, _ = lmhead_project(
                    x=h_norm,
                    lm_head=wrapper.lm_head,
                    stable=wrapper.stable,
                    model_device=wrapper.model_device,
                )
                rec[f"logits_{mode}"] = _detach_full_tensor_to_cpu(logits)
        if config.collect_components and attention_outputs is not None and idx < len(attention_outputs):
            attn_full = attention_outputs[idx]
            mlp_full = mlp_outputs[idx] if mlp_outputs is not None and idx < len(mlp_outputs) else None
            if attn_full is not None:
                rec["attention_output"] = _detach_full_tensor_to_cpu(attn_full[:, :seq_len, :])
            if mlp_full is not None:
                rec["mlp_output"] = _detach_full_tensor_to_cpu(mlp_full[:, :seq_len, :])
            if config.project_component_logits and attention_logits_by_mode is not None:
                for mode in config.norm_modes:
                    attn_logits = attention_logits_by_mode.get(mode, [])
                    mlp_logits = mlp_logits_by_mode.get(mode, []) if mlp_logits_by_mode is not None else []
                    if idx < len(attn_logits):
                        rec[f"attention_logits_{mode}"] = _detach_full_tensor_to_cpu(attn_logits[idx][:, :seq_len, :])
                    if idx < len(mlp_logits):
                        rec[f"mlp_logits_{mode}"] = _detach_full_tensor_to_cpu(mlp_logits[idx][:, :seq_len, :])
        records.append(rec)

    if config.force_include_output and num_blocks > 0:
        hidden_full = hidden_seq[-1]
        out_idx = num_blocks
        rec = {
            "layer_index": out_idx,
            "layer_name": "output",
            "tokens": tokens,
            "attention_mask": _detach_full_tensor_to_cpu(attention_mask[:, :seq_len]),
            "hidden": _detach_full_tensor_to_cpu(hidden_full),
        }
        if config.save_logits:
            for mode in config.norm_modes:
                h_norm = normalize_activations(
                    x=hidden_full.clone(),
                    mode=mode,
                    block="output",
                    layer_index=out_idx,
                    model_device=wrapper.model_device,
                    model_dtype=wrapper.model_dtype,
                    final_norm=wrapper.final_norm,
                )
                logits, _ = lmhead_project(
                    x=h_norm,
                    lm_head=wrapper.lm_head,
                    stable=wrapper.stable,
                    model_device=wrapper.model_device,
                )
                rec[f"logits_{mode}"] = _detach_full_tensor_to_cpu(logits)
        records.append(rec)

    return records


@torch.no_grad()
def collect_prompt_lens_activations(
    wrapper: LogitLensWrapper,
    config: PromptLensActivationCollectorConfig,
) -> Dict[str, Any]:
    prompt_formatted = _format_generation_prompt(
        wrapper,
        config.prompt,
        prompt_format=config.prompt_format,
        use_chat_template=config.use_chat_template,
        system_prompt=config.system_prompt,
    )

    inputs = wrapper.tokenize_inputs(
        texts=prompt_formatted,
        device=wrapper.model_device,
        add_special_tokens=config.add_special_tokens and not config.use_chat_template,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    hook_buffers: Dict[str, Dict[int, torch.Tensor]] = {
        "attention_outputs": {},
        "mlp_outputs": {},
    }
    hook_handles: List[Any] = []

    if config.collect_components:
        def _save_component_hook(component_name: str, layer_idx: int):
            def fn(module, inp, out):
                tensor = wrapper._extract_tensor(out)
                if tensor is None:
                    return out
                hook_buffers[component_name][layer_idx] = _detach_full_tensor_to_cpu(tensor)
                return out
            return fn

        component_registry = getattr(wrapper, "component_registry", {}) or {}
        for layer_idx, block in enumerate(wrapper.blocks):
            attn_entry = component_registry.get(f"attention_{layer_idx:02d}")
            mlp_entry = component_registry.get(f"mlp_{layer_idx:02d}")
            attn_module = attn_entry["module"] if attn_entry is not None else resolve_block_component_module(block, "attention")
            mlp_module = mlp_entry["module"] if mlp_entry is not None else resolve_block_component_module(block, "mlp")
            if attn_module is None:
                raise ValueError(f"Could not resolve attention module for layer {layer_idx}")
            if mlp_module is None:
                raise ValueError(f"Could not resolve MLP module for layer {layer_idx}")
            hook_handles.append(attn_module.register_forward_hook(_save_component_hook("attention_outputs", layer_idx)))
            hook_handles.append(mlp_module.register_forward_hook(_save_component_hook("mlp_outputs", layer_idx)))

    try:
        outputs = wrapper.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
    finally:
        for handle in hook_handles:
            handle.remove()

    hidden_states = list(outputs.hidden_states)
    attention_outputs = [hook_buffers["attention_outputs"].get(i) for i in range(len(wrapper.blocks))]
    mlp_outputs = [hook_buffers["mlp_outputs"].get(i) for i in range(len(wrapper.blocks))]
    attention_logits_by_mode = None
    mlp_logits_by_mode = None
    if config.collect_components and config.project_component_logits:
        attention_logits_by_mode = {mode: [] for mode in config.norm_modes}
        mlp_logits_by_mode = {mode: [] for mode in config.norm_modes}
        for idx in range(len(wrapper.blocks)):
            attn_full = hook_buffers["attention_outputs"].get(idx)
            mlp_full = hook_buffers["mlp_outputs"].get(idx)
            if attn_full is None or mlp_full is None:
                attention_logits_by_mode = None
                mlp_logits_by_mode = None
                break
            for mode in config.norm_modes:
                for full, store in (
                    (attn_full, attention_logits_by_mode[mode]),
                    (mlp_full, mlp_logits_by_mode[mode]),
                ):
                    h_norm = normalize_activations(
                        x=full.clone().to(device=wrapper.model_device, dtype=wrapper.model_dtype),
                        mode=mode,
                        block="block",
                        layer_index=idx,
                        model_device=wrapper.model_device,
                        model_dtype=wrapper.model_dtype,
                        final_norm=wrapper.final_norm,
                    )
                    logits, _ = lmhead_project(
                        x=h_norm,
                        lm_head=wrapper.lm_head,
                        stable=wrapper.stable,
                        model_device=wrapper.model_device,
                    )
                    store.append(_detach_full_tensor_to_cpu(logits))

    records = _collect_layer_records(
        wrapper,
        input_ids=input_ids,
        attention_mask=attention_mask,
        hidden_states=hidden_states,
        attention_outputs=attention_outputs,
        mlp_outputs=mlp_outputs,
        attention_logits_by_mode=attention_logits_by_mode,
        mlp_logits_by_mode=mlp_logits_by_mode,
        config=config,
    )
    return {
        "prompt": config.prompt,
        "prompt_formatted": prompt_formatted,
        "layer_records": records,
    }


def collect_activation_dataset_incremental(
    *,
    wrapper: LogitLensWrapper,
    dataset_path: str | Path,
    output_path: str | Path,
    partial_path: str | Path,
    text_field: str,
    label_field: str,
    model_key: str,
    use_chat_template: bool,
    prompt_format: str,
    system_prompt: str | None,
    add_special_tokens: bool,
    force_include_input: bool,
    force_include_output: bool,
    norm_modes: tuple[str, ...],
    collect_components: bool,
    project_component_logits: bool,
    save_logits: bool = True,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload_rows = []
    for row in tqdm(rows, desc=f"collect:{model_key}"):
        text, continuation_kind = _build_collection_text_and_kind(row, text_field=text_field)
        cfg = PromptLensActivationCollectorConfig(
            prompt=text,
            use_chat_template=use_chat_template,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            add_special_tokens=add_special_tokens,
            force_include_input=force_include_input,
            force_include_output=force_include_output,
            norm_modes=norm_modes,
            collect_components=collect_components,
            project_component_logits=project_component_logits,
            save_logits=bool(save_logits),
        )
        item = collect_prompt_lens_activations(wrapper, cfg)
        payload_rows.append(
            {
                "id": row.get("id"),
                "group_id": row.get("group_id"),
                "variant": row.get("variant"),
                "language": row.get("language"),
                "label": row.get(label_field),
                "continuation_kind": continuation_kind,
                "prompt": text,
                "prompt_formatted": item["prompt_formatted"],
                "layer_records": item["layer_records"],
            }
        )
        torch.save(
            {
                "dataset_path": str(dataset_path),
                "model_key": model_key,
                "text_field": text_field,
                "label_field": label_field,
                "rows": payload_rows,
                "num_rows_completed": len(payload_rows),
                "num_examples": len(rows),
                "norm_modes": list(norm_modes),
            },
            partial_path,
        )

    payload = {
        "dataset_path": str(dataset_path),
        "model_key": model_key,
        "text_field": text_field,
        "label_field": label_field,
        "rows": payload_rows,
        "num_rows_completed": len(payload_rows),
        "num_examples": len(rows),
        "norm_modes": list(norm_modes),
    }
    torch.save(payload, output_path)
    return payload

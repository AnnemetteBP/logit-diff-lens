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
class TeacherForcingActivationCollectorConfig:
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
    config: TeacherForcingActivationCollectorConfig,
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
def collect_teacher_forced_activations(
    wrapper: LogitLensWrapper,
    config: TeacherForcingActivationCollectorConfig,
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
            hook_handles.append(
                attn_module.register_forward_hook(_save_component_hook("attention_outputs", layer_idx))
            )
            hook_handles.append(
                mlp_module.register_forward_hook(_save_component_hook("mlp_outputs", layer_idx))
            )

    try:
        outputs = wrapper.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
    finally:
        for hook in hook_handles:
            try:
                hook.remove()
            except Exception:
                pass

    hidden_states = outputs.hidden_states
    logits = outputs.logits
    if hidden_states is None:
        raise ValueError("Model forward pass did not return hidden_states")
    if logits is None:
        raise ValueError("Model forward pass did not return logits")

    expected_num_layers = len(wrapper.blocks)
    if len(hidden_states) != expected_num_layers + 1:
        raise ValueError(
            f"Expected embedding + {expected_num_layers} block hidden states, got {len(hidden_states)}"
        )

    attention_outputs = [hook_buffers["attention_outputs"].get(idx) for idx in range(expected_num_layers)]
    mlp_outputs = [hook_buffers["mlp_outputs"].get(idx) for idx in range(expected_num_layers)]
    if config.collect_components:
        if any(tensor is None for tensor in attention_outputs):
            missing = [idx for idx, tensor in enumerate(attention_outputs) if tensor is None]
            raise ValueError(f"Missing attention outputs for layers {missing}")
        if any(tensor is None for tensor in mlp_outputs):
            missing = [idx for idx, tensor in enumerate(mlp_outputs) if tensor is None]
            raise ValueError(f"Missing MLP outputs for layers {missing}")

    attention_logits_by_mode: Dict[str, List[torch.Tensor]] = {}
    mlp_logits_by_mode: Dict[str, List[torch.Tensor]] = {}
    if config.collect_components and config.project_component_logits:
        for mode in config.norm_modes:
            attention_logits_by_mode[mode] = []
            mlp_logits_by_mode[mode] = []
            for idx in range(expected_num_layers):
                attn_full = attention_outputs[idx]
                mlp_full = mlp_outputs[idx]
                attn_norm = normalize_activations(
                    x=attn_full.clone(),
                    mode=mode,
                    block="block",
                    layer_index=idx,
                    model_device=wrapper.model_device,
                    model_dtype=wrapper.model_dtype,
                    final_norm=wrapper.final_norm,
                )
                attn_logits, _ = lmhead_project(
                    x=attn_norm,
                    lm_head=wrapper.lm_head,
                    stable=wrapper.stable,
                    model_device=wrapper.model_device,
                )
                attention_logits_by_mode[mode].append(_detach_full_tensor_to_cpu(attn_logits))
                mlp_norm = normalize_activations(
                    x=mlp_full.clone(),
                    mode=mode,
                    block="block",
                    layer_index=idx,
                    model_device=wrapper.model_device,
                    model_dtype=wrapper.model_dtype,
                    final_norm=wrapper.final_norm,
                )
                mlp_logits, _ = lmhead_project(
                    x=mlp_norm,
                    lm_head=wrapper.lm_head,
                    stable=wrapper.stable,
                    model_device=wrapper.model_device,
                )
                mlp_logits_by_mode[mode].append(_detach_full_tensor_to_cpu(mlp_logits))

    layer_records = _collect_layer_records(
        wrapper,
        input_ids=input_ids,
        attention_mask=attention_mask,
        hidden_states=hidden_states,
        attention_outputs=attention_outputs if config.collect_components else None,
        mlp_outputs=mlp_outputs if config.collect_components else None,
        attention_logits_by_mode=attention_logits_by_mode if config.collect_components else None,
        mlp_logits_by_mode=mlp_logits_by_mode if config.collect_components else None,
        config=config,
    )

    return {
        "input_ids": _detach_full_tensor_to_cpu(input_ids),
        "attention_mask": _detach_full_tensor_to_cpu(attention_mask),
        "embedding": _detach_full_tensor_to_cpu(hidden_states[0]),
        "hidden_states": [_detach_full_tensor_to_cpu(tensor) for tensor in hidden_states],
        "logits": _detach_full_tensor_to_cpu(logits),
        "attention_outputs": attention_outputs if config.collect_components else [],
        "mlp_outputs": mlp_outputs if config.collect_components else [],
        "attention_logits_by_mode": attention_logits_by_mode,
        "mlp_logits_by_mode": mlp_logits_by_mode,
        "layer_records": layer_records,
        "force_include_input": bool(config.force_include_input),
        "force_include_output": bool(config.force_include_output),
        "norm_modes": list(config.norm_modes),
        "collect_components": bool(config.collect_components),
        "project_component_logits": bool(config.project_component_logits),
    }


def collect_activation_dataset_incremental(
    *,
    wrapper: LogitLensWrapper,
    dataset_path: str | Path,
    output_path: str | Path,
    partial_path: str | Path | None = None,
    text_field: str = "analysis_text",
    label_field: str = "label",
    model_key: str = "model",
    use_chat_template: bool = False,
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain",
    system_prompt: str | None = None,
    add_special_tokens: bool = True,
    force_include_input: bool = True,
    force_include_output: bool = False,
    norm_modes: tuple[str, ...] = ("raw", "model_norm"),
    collect_components: bool = False,
    project_component_logits: bool = False,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    partial_path = Path(partial_path) if partial_path is not None else output_path.with_suffix(".partial.pt")

    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Dataset {dataset_path} is empty")

    completed_ids: set[int] = set()
    result_rows: List[Dict[str, Any]] = []

    if partial_path.exists():
        payload = torch.load(partial_path, map_location="cpu")
        result_rows = list(payload.get("rows", []))
        completed_ids = {int(row["id"]) for row in result_rows if row.get("id") is not None}

    def _get_lm_head_weight() -> torch.Tensor:
        lm_head = getattr(wrapper, "lm_head", None)
        if lm_head is None and getattr(wrapper, "model", None) is not None:
            lm_head = wrapper.model.get_output_embeddings()
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise ValueError("Could not resolve LM head weight for collected activation payload")
        return _detach_full_tensor_to_cpu(lm_head.weight)

    def _save_checkpoint() -> Dict[str, Any]:
        payload = {
            "dataset_path": str(dataset_path),
            "text_field": text_field,
            "label_field": label_field,
            "model_key": model_key,
            "use_chat_template": use_chat_template,
            "prompt_format": prompt_format,
            "system_prompt": system_prompt,
            "force_include_input": force_include_input,
            "force_include_output": force_include_output,
            "norm_modes": list(norm_modes),
            "collect_components": collect_components,
            "project_component_logits": project_component_logits,
            "num_examples": len(rows),
            "num_rows_completed": len(result_rows),
            "completed_ids": sorted(completed_ids),
            "lm_head_weight": _get_lm_head_weight(),
            "rows": result_rows,
        }
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, partial_path)
        return payload

    for row in tqdm(rows, desc=f"Collecting activations ({model_key})"):
        row_id = int(row["id"])
        if row_id in completed_ids:
            continue
        collection_text, continuation_kind = _build_collection_text_and_kind(row, text_field=text_field)
        config = TeacherForcingActivationCollectorConfig(
            prompt=collection_text,
            use_chat_template=use_chat_template,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            add_special_tokens=add_special_tokens,
            force_include_input=force_include_input,
            force_include_output=force_include_output,
            norm_modes=norm_modes,
            collect_components=collect_components,
            project_component_logits=project_component_logits,
        )
        result = collect_teacher_forced_activations(wrapper, config)
        result_rows.append(
            {
                **row,
                "collection_text_field": text_field,
                "collection_prompt_format": prompt_format,
                "collection_system_prompt": system_prompt,
                "collection_use_chat_template": use_chat_template,
                "collection_concat_strategy": "prompt_space_response_for_model_responses",
                "collection_text": collection_text,
                "continuation_kind": continuation_kind,
                "label": row.get(label_field),
                "input_ids": result["input_ids"],
                "attention_mask": result["attention_mask"],
                "embedding": result["embedding"],
                "hidden_states": result["hidden_states"],
                "logits": result["logits"],
                "attention_outputs": result["attention_outputs"],
                "mlp_outputs": result["mlp_outputs"],
                "attention_logits_by_mode": result["attention_logits_by_mode"],
                "mlp_logits_by_mode": result["mlp_logits_by_mode"],
                "layer_records": result["layer_records"],
            }
        )
        completed_ids.add(row_id)
        _save_checkpoint()

    final_payload = _save_checkpoint()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_payload, output_path)
    return final_payload


ActivationCollectorConfig = TeacherForcingActivationCollectorConfig


def collect_teacher_forcing_activations(
    wrapper: LogitLensWrapper,
    config: TeacherForcingActivationCollectorConfig,
) -> Dict[str, Any]:
    return collect_teacher_forced_activations(wrapper, config)

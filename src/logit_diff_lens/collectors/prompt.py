from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import torch
from tqdm.auto import tqdm

from ..schemas import BackendMetadata, PromptDecodeArtifact, PromptLayerRecord
from ..validation import validate_prompt_decode_artifact
from ..wrappers import (
    LogitLensWrapper,
    as_tensor,
    lmhead_project,
    normalize_activations,
    resolve_block_component_module,
)


def _format_generation_prompt(
    wrapper: LogitLensWrapper,
    prompt: str,
    *,
    prompt_format: str,
    use_chat_template: bool,
    system_prompt: str | None,
) -> str:
    if use_chat_template and prompt_format == "plain":
        prompt_format = "chat_template"

    if prompt_format == "plain":
        return prompt

    if prompt_format == "user_assistant_prefix":
        parts: List[str] = []
        if system_prompt:
            parts.append(f"System: {system_prompt}\n")
        parts.append(f"User: {prompt}\nAssistant:")
        return "".join(parts)

    if prompt_format != "chat_template":
        raise ValueError(
            f"Unknown prompt_format '{prompt_format}'. Expected 'plain', 'chat_template', or 'user_assistant_prefix'."
        )

    tokenizer = wrapper.tokenizer
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError(
            "prompt_format='chat_template' requires a tokenizer with an available chat template."
        )

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(formatted, str):
        raise TypeError("Expected string output from tokenizer.apply_chat_template(..., tokenize=False)")
    return formatted


@dataclass
class PromptLensActivationCollectorConfig:
    prompt: str = ""
    use_chat_template: bool = False
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain"
    system_prompt: str | None = None
    add_special_tokens: bool = True
    truncation: bool = False
    max_length: int | None = None
    padding: bool | str | None = None
    force_include_input: bool = True
    force_include_output: bool = False
    norm_modes: tuple[str, ...] = ("raw", "model_norm")
    collect_components: bool = False
    project_component_logits: bool = False
    save_logits: bool = True


def _detach_full_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def _decode_token_ids(tok, token_ids: torch.Tensor) -> List[str]:
    ids = token_ids.detach().cpu().tolist()
    return [tok.decode([tid], clean_up_tokenization_spaces=False) for tid in ids]


def _safe_tokenizer_name(tokenizer: Any) -> str | None:
    for attr in ("name_or_path",):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, str) and value:
            return value
    return tokenizer.__class__.__name__ if tokenizer is not None else None


def _build_backend_metadata(wrapper: LogitLensWrapper) -> BackendMetadata:
    model = wrapper.model
    config = getattr(model, "config", None)
    quantization = "bitsandbytes" if getattr(wrapper, "is_bnb_quantized", False) else "none"
    model_id = getattr(config, "_name_or_path", None) if config is not None else None
    model_revision = getattr(config, "_commit_hash", None) if config is not None else None
    return BackendMetadata(
        model_backend="transformers",
        activation_backend="wrapper",
        decode_backend="wrapper_utils",
        device_policy="follow_lm_head",
        dtype_compute=str(wrapper.model_dtype),
        dtype_storage="torch.float32@cpu",
        quantization=quantization,
        device_map=str(getattr(model, "hf_device_map", "single_device")),
        model_id=model_id,
        tokenizer_id=_safe_tokenizer_name(wrapper.tokenizer),
        model_revision=model_revision,
        wrapper_name=wrapper.__class__.__name__,
        architecture=str(getattr(wrapper, "arch", "unknown")),
        extra={
            "stable_projection": bool(getattr(wrapper, "stable", False)),
            "include_final_norm": bool(getattr(wrapper, "include_final_norm", False)),
        },
    )


def _build_prompt_decode_artifact(
    *,
    wrapper: LogitLensWrapper,
    config: PromptLensActivationCollectorConfig,
    prompt_formatted: str,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_records: List[Dict[str, Any]],
) -> PromptDecodeArtifact:
    layer_objs = [PromptLayerRecord.from_legacy_dict(record) for record in layer_records]
    seq_tokens = token_ids[0]
    artifact = PromptDecodeArtifact(
        prompt_text=config.prompt,
        prompt_formatted=prompt_formatted,
        token_ids=_detach_full_tensor_to_cpu(token_ids),
        token_text=_decode_token_ids(wrapper.tokenizer, seq_tokens),
        attention_mask=_detach_full_tensor_to_cpu(attention_mask),
        layer_records=layer_objs,
        backend_metadata=_build_backend_metadata(wrapper),
        lens_modes=[mode for mode in config.norm_modes if mode in ("raw", "model_norm")],
        metadata={
            "add_special_tokens": bool(config.add_special_tokens),
            "truncation": bool(config.truncation),
            "max_length": config.max_length,
            "padding": config.padding,
            "force_include_input": bool(config.force_include_input),
            "force_include_output": bool(config.force_include_output),
            "collect_components": bool(config.collect_components),
            "project_component_logits": bool(config.project_component_logits),
            "save_logits": bool(config.save_logits),
            "prompt_format": config.prompt_format,
            "use_chat_template": bool(config.use_chat_template),
            "system_prompt": config.system_prompt,
        },
    )
    validate_prompt_decode_artifact(artifact)
    return artifact


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
    token_text = _decode_token_ids(wrapper.tokenizer, input_ids[0, :seq_len])
    records: List[Dict[str, Any]] = []

    if config.force_include_input:
        hidden_full = hidden_seq[0]
        rec: Dict[str, Any] = {
            "layer_index": -1,
            "layer_name": "embedding",
            "tokens": tokens,
            "token_text": token_text,
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
            "token_text": token_text,
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
            "token_text": token_text,
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
        truncation=config.truncation,
        max_length=config.max_length,
        padding=config.padding,
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
        acts, _ = wrapper.forward_pass(
            input_ids=input_ids,
            attention_mask=attention_mask,
            collect_attn=False,
        )
    finally:
        for handle in hook_handles:
            handle.remove()

    hidden_states = []
    embedding_name = next(
        name for name, entry in wrapper.layer_registry.items()
        if entry["type"] == "embedding" and name in acts
    )
    hidden_states.append(acts[embedding_name])
    block_names = [
        name
        for name, entry in sorted(
            wrapper.layer_registry.items(),
            key=lambda item: (item[1].get("idx", -1), item[0]),
        )
        if entry["type"] == "block" and name in acts
    ]
    hidden_states.extend(acts[name] for name in block_names)
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
    artifact = _build_prompt_decode_artifact(
        wrapper=wrapper,
        config=config,
        prompt_formatted=prompt_formatted,
        token_ids=input_ids[:, : int(attention_mask[0].sum().item())],
        attention_mask=attention_mask[:, : int(attention_mask[0].sum().item())],
        layer_records=records,
    )
    return {
        "prompt": config.prompt,
        "prompt_formatted": prompt_formatted,
        "layer_records": records,
        "artifact": artifact,
        "artifact_dict": artifact.to_dict(),
    }


def collect_prompt_activation_dataset_incremental(
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
    truncation: bool = False,
    max_length: int | None = None,
    padding: bool | str | None = None,
    force_include_input: bool = True,
    force_include_output: bool = False,
    norm_modes: tuple[str, ...] = ("raw", "model_norm"),
    collect_components: bool = False,
    project_component_logits: bool = False,
    save_logits: bool = True,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload_rows = []
    artifact_dicts: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc=f"collect:{model_key}"):
        text, continuation_kind = _build_collection_text_and_kind(row, text_field=text_field)
        cfg = PromptLensActivationCollectorConfig(
            prompt=text,
            use_chat_template=use_chat_template,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            force_include_input=force_include_input,
            force_include_output=force_include_output,
            norm_modes=norm_modes,
            collect_components=collect_components,
            project_component_logits=project_component_logits,
            save_logits=bool(save_logits),
        )
        item = collect_prompt_lens_activations(wrapper, cfg)
        artifact_dict = item["artifact_dict"]
        artifact_dict["prompt_id"] = str(row.get("id")) if row.get("id") is not None else None
        artifact_dict.setdefault("metadata", {})
        artifact_dict["metadata"].update(
            {
                "row_id": row.get("id"),
                "group_id": row.get("group_id"),
                "variant": row.get("variant"),
                "language": row.get("language"),
                "label": row.get(label_field),
                "continuation_kind": continuation_kind,
                "model_key": model_key,
            }
        )
        artifact_dicts.append(artifact_dict)
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
                "artifact": artifact_dict,
            }
        )
        torch.save(
            {
                "dataset_path": str(dataset_path),
                "model_key": model_key,
                "text_field": text_field,
                "label_field": label_field,
                "rows": payload_rows,
                "artifacts": artifact_dicts,
                "num_rows_completed": len(payload_rows),
                "num_examples": len(rows),
                "norm_modes": list(norm_modes),
                "artifact_schema": "PromptDecodeArtifact",
            },
            partial_path,
        )

    payload = {
        "dataset_path": str(dataset_path),
        "model_key": model_key,
        "text_field": text_field,
        "label_field": label_field,
        "rows": payload_rows,
        "artifacts": artifact_dicts,
        "num_rows_completed": len(payload_rows),
        "num_examples": len(rows),
        "norm_modes": list(norm_modes),
        "artifact_schema": "PromptDecodeArtifact",
    }
    torch.save(payload, output_path)
    return payload


@torch.no_grad()
def collect_prompt_logits_for_plotter(
    arch_wrapper: LogitLensWrapper,
    prompt: str | List[str] | None = None,
    mode: str = "raw",
    topk: int = 5,
    selected_layers: List[int] | None = None,
    add_special_tokens: bool = False,
    force_include_input: bool = True,
    force_include_output: bool = True,
) -> Dict[str, Any]:
    model = arch_wrapper.model
    model.eval()
    device = arch_wrapper.model_device
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    tok = arch_wrapper.tokenizer
    inputs = arch_wrapper.tokenize_inputs(
        texts=prompt,
        device=device,
        add_special_tokens=add_special_tokens,
    )
    full_input_ids = as_tensor(inputs["input_ids"], device=arch_wrapper.model_device)
    full_attention_mask = as_tensor(inputs["attention_mask"], device=arch_wrapper.model_device)
    if full_input_ids.ndim == 1:
        full_input_ids = full_input_ids.unsqueeze(0)
    if full_attention_mask.ndim == 1:
        full_attention_mask = full_attention_mask.unsqueeze(0)
    if full_input_ids.shape[1] < 2:
        raise ValueError("Prompt must tokenize to at least 2 tokens for next-token analysis.")
    input_ids = full_input_ids[:, :-1]
    target_ids = full_input_ids[:, 1:]
    attention_mask = full_attention_mask[:, :-1]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if target_ids.ndim == 1:
        target_ids = target_ids.unsqueeze(0)
    token_count = input_ids.shape[1]
    acts, _ = arch_wrapper.forward_pass(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collect_attn=False,
    )
    result = {
        "prompt": prompt,
        "full_tokens": _decode_token_ids(tok, full_input_ids[0]),
        "tokens": _decode_token_ids(tok, input_ids[0]),
        "target_ids": target_ids[0].detach().cpu(),
        "target_tokens": _decode_token_ids(tok, target_ids[0]),
        "attention_mask": attention_mask[0].detach().cpu(),
        "layers": [],
        "hidden": {},
        "logits": {},
        "topk_preds": {},
        "mode": mode,
        "tokenizer": tok,
        "quantized": arch_wrapper.is_bnb_quantized,
    }
    if force_include_input and "embedding" in acts:
        h_emb = acts["embedding"]
        h_norm = normalize_activations(
            x=h_emb.clone(),
            mode=mode,
            block="embedding",
            layer_index=-1,
            model_device=arch_wrapper.model_device,
            model_dtype=arch_wrapper.model_dtype,
            final_norm=arch_wrapper.final_norm,
        )
        l_norm, _ = lmhead_project(
            x=h_norm,
            lm_head=arch_wrapper.lm_head,
            stable=arch_wrapper.stable,
            model_device=arch_wrapper.model_device,
        )
        result["hidden"][(-1, mode)] = arch_wrapper.save_to_fp32(h_norm[0]) if arch_wrapper.fp32_save else h_norm[0].cpu()
        result["logits"][(-1, mode)] = arch_wrapper.save_to_fp32(l_norm[0]) if arch_wrapper.fp32_save else l_norm[0].cpu()
        result["layers"].append(("embedding", mode))
    selected_layer_set = set(selected_layers) if selected_layers is not None else None
    blocks = sorted(
        (v["idx"], k)
        for k, v in arch_wrapper.layer_registry.items()
        if v["type"] == "block" and k in acts
    )
    last_raw = None
    for layer_index, key in blocks:
        last_raw = acts[key]
        if selected_layer_set is not None and layer_index not in selected_layer_set:
            continue
        h = normalize_activations(
            x=last_raw.clone(),
            mode=mode,
            block="block",
            layer_index=layer_index,
            model_device=arch_wrapper.model_device,
            model_dtype=arch_wrapper.model_dtype,
            final_norm=arch_wrapper.final_norm,
        )
        logits, _ = lmhead_project(
            x=h,
            lm_head=arch_wrapper.lm_head,
            stable=arch_wrapper.stable,
            model_device=arch_wrapper.model_device,
        )
        result["hidden"][(layer_index, mode)] = arch_wrapper.save_to_fp32(h[0]) if arch_wrapper.fp32_save else h[0].cpu()
        result["logits"][(layer_index, mode)] = arch_wrapper.save_to_fp32(logits[0]) if arch_wrapper.fp32_save else logits[0].cpu()
        result["layers"].append((layer_index, mode))
    if force_include_output and last_raw is not None:
        out_idx = max(i for i, _ in result["layers"] if isinstance(i, int)) + 1
        h = normalize_activations(
            x=last_raw.clone(),
            mode=mode,
            block="output",
            layer_index=out_idx,
            model_device=arch_wrapper.model_device,
            model_dtype=arch_wrapper.model_dtype,
            final_norm=arch_wrapper.final_norm,
        )
        logits, _ = lmhead_project(
            x=h,
            lm_head=arch_wrapper.lm_head,
            stable=arch_wrapper.stable,
            model_device=arch_wrapper.model_device,
        )
        result["hidden"][(out_idx, mode)] = arch_wrapper.save_to_fp32(h[0]) if arch_wrapper.fp32_save else h[0].cpu()
        result["logits"][(out_idx, mode)] = arch_wrapper.save_to_fp32(logits[0]) if arch_wrapper.fp32_save else logits[0].cpu()
        result["layers"].append(("output", mode))
    for (layer_id, layer_mode), logits in result["logits"].items():
        if layer_mode != mode:
            continue
        probs = torch.softmax(logits, dim=-1)
        _, topk_idx = torch.topk(probs, k=topk, dim=-1)
        result["topk_preds"][(layer_id, mode)] = [
            [tok.decode([token_id]) for token_id in row.tolist()]
            for row in topk_idx
        ]
    for key, value in result["hidden"].items():
        assert value.shape[0] == token_count, (key, value.shape)
    for key, value in result["logits"].items():
        assert value.shape[0] == token_count, (key, value.shape)
    return result


run_prompt_collection = None


__all__ = [
    "PromptLensActivationCollectorConfig",
    "collect_prompt_activation_dataset_incremental",
    "collect_prompt_lens_activations",
    "collect_prompt_logits_for_plotter",
    "run_prompt_collection",
]

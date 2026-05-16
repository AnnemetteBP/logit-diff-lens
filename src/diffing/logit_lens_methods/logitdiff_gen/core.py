from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import gc
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Sequence

import torch

from ..wrapper import LogitLensWrapper, normalize_activations, lmhead_project, generate_with_model


@dataclass
class LogitDiffRunConfig:
    prompts: List[str]
    prompt_metadata: List[Dict[str, Any]] | None = None
    layers: Sequence[float | int] = field(default_factory=lambda: (0.5, 0.6, 0.7, 0.8, 0.9))
    top_k: int = 10
    comparison_top_ks: tuple[int, ...] = (1, 5, 10)
    max_new_tokens: int = 0
    add_special_tokens: bool = True
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain"
    use_chat_template: bool = False
    system_prompt: str | None = None
    template_name: str = "plain"
    norm_mode: str = "raw"
    do_sample: bool = False
    temperature: float = 1.0
    use_cache: bool = False
    output_hidden_states: bool = False
    seed: int | None = None
    output_format: Literal["rich", "toolkit_legacy"] = "rich"
    label: str = "logitdiff"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _make_toolkit_legacy_payload(results: Dict[str, Any]) -> Dict[str, Any]:
    legacy: Dict[str, Any] = {}
    for layer_key, entries in results.items():
        legacy[layer_key] = []
        for entry in entries:
            legacy_entry = {
                "prompt": entry["prompt"],
                "layer_relative": entry["layer_relative"],
                "layer_absolute": entry["layer_absolute"],
                "mean_iou": entry["mean_iou"],
                "positions": [],
            }
            for pos in entry["positions"]:
                legacy_entry["positions"].append(
                    {
                        "position": pos["position"],
                        "input_token": pos["input_token"],
                        "is_generated": pos["is_generated"],
                        "iou": pos["iou"],
                        "intersection": pos["intersection"],
                        "only_base": pos["only_base"],
                        "only_finetuned": pos["only_finetuned"],
                        "num_intersection": pos["num_intersection"],
                        "num_only_base": pos["num_only_base"],
                        "num_only_finetuned": pos["num_only_finetuned"],
                    }
                )
            legacy[layer_key].append(legacy_entry)
    return legacy


def _validate_tokenizers(
    wrapper_a: LogitLensWrapper,
    wrapper_b: LogitLensWrapper,
) -> None:
    tok_a = wrapper_a.tokenizer
    tok_b = wrapper_b.tokenizer

    if type(tok_a) is not type(tok_b):
        raise ValueError(
            f"Incompatible tokenizers: {type(tok_a).__name__} vs {type(tok_b).__name__}"
        )
    if tok_a.vocab_size != tok_b.vocab_size:
        raise ValueError(
            f"Tokenizer vocab mismatch: {tok_a.vocab_size} vs {tok_b.vocab_size}"
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


def _extract_effective_system_prompt(formatted_prompt: str) -> str | None:
    start_marker = "<|im_start|>system\n"
    end_marker = "<|im_end|>"

    start_idx = formatted_prompt.find(start_marker)
    if start_idx == -1:
        return None

    content_start = start_idx + len(start_marker)
    end_idx = formatted_prompt.find(end_marker, content_start)
    if end_idx == -1:
        return None

    system_prompt = formatted_prompt[content_start:end_idx].strip()
    return system_prompt or None


def _resolve_layer_indices(
    wrapper: LogitLensWrapper,
    layers: Sequence[float | int],
) -> List[tuple[float, int]]:
    num_blocks = len(wrapper.blocks)
    if num_blocks < 1:
        raise ValueError("Wrapper must expose at least one transformer block")

    resolved: List[tuple[float, int]] = []
    for layer in layers:
        if isinstance(layer, float):
            if not (0.0 <= layer <= 1.0):
                raise ValueError(f"Relative layer must be in [0, 1], got {layer}")
            abs_idx = min(int(round((num_blocks - 1) * layer)), num_blocks - 1)
            rel = float(layer)
        else:
            abs_idx = int(layer)
            if not (0 <= abs_idx < num_blocks):
                raise ValueError(
                    f"Absolute layer index must be in [0, {num_blocks - 1}], got {abs_idx}"
                )
            rel = abs_idx / max(num_blocks - 1, 1)
        resolved.append((rel, abs_idx))

    seen = set()
    deduped: List[tuple[float, int]] = []
    for rel, abs_idx in resolved:
        if abs_idx in seen:
            continue
        seen.add(abs_idx)
        deduped.append((rel, abs_idx))
    return deduped


@torch.no_grad()
def _generate_sequence(
    wrapper: LogitLensWrapper,
    prompt: str,
    *,
    add_special_tokens: bool,
    prompt_format: str,
    use_chat_template: bool,
    system_prompt: str | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    use_cache: bool,
    output_hidden_states: bool,
    seed: int | None,
) -> tuple[torch.Tensor, int, torch.Tensor, str]:
    if output_hidden_states:
        raise ValueError(
            "logitdiff generation uses hook-based tracing assumptions; "
            "set output_hidden_states=False."
        )

    prompt_for_generation = _format_generation_prompt(
        wrapper,
        prompt,
        prompt_format=prompt_format,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )

    inputs = wrapper.tokenize_inputs(
        texts=prompt_for_generation,
        device=wrapper.model_device,
        add_special_tokens=add_special_tokens and not use_chat_template,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_len = input_ids.shape[1]

    if max_new_tokens <= 0:
        return input_ids, prompt_len, attention_mask, prompt_for_generation

    wrapper.model.eval()
    generated, generated_attention_mask = generate_with_model(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        use_cache=use_cache,
        seed=seed,
    )
    return generated, prompt_len, generated_attention_mask, prompt_for_generation


@torch.no_grad()
def _collect_layer_logits(
    wrapper: LogitLensWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    layer_indices: Sequence[int],
    norm_mode: str,
) -> Dict[int, torch.Tensor]:
    acts, _ = wrapper.forward_pass(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collect_attn=False,
    )

    block_names = {
        entry["idx"]: name
        for name, entry in wrapper.layer_registry.items()
        if entry["type"] == "block" and name in acts
    }

    logits_by_layer: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        if layer_idx not in block_names:
            raise KeyError(f"Layer {layer_idx} was not captured by hooks")
        hidden = acts[block_names[layer_idx]]
        h_norm = normalize_activations(
            x=hidden.clone(),
            mode=norm_mode,
            block="block",
            layer_index=layer_idx,
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
        logits_by_layer[layer_idx] = logits.detach().cpu()
    return logits_by_layer


@torch.no_grad()
def _collect_layer_topk_predictions(
    wrapper: LogitLensWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    layer_indices: Sequence[int],
    norm_mode: str,
    max_k: int,
) -> Dict[int, torch.Tensor]:
    acts, _ = wrapper.forward_pass(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collect_attn=False,
    )

    block_names = {
        entry["idx"]: name
        for name, entry in wrapper.layer_registry.items()
        if entry["type"] == "block" and name in acts
    }

    topk_by_layer: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        if layer_idx not in block_names:
            raise KeyError(f"Layer {layer_idx} was not captured by hooks")
        hidden = acts[block_names[layer_idx]]
        h_norm = normalize_activations(
            x=hidden.clone(),
            mode=norm_mode,
            block="block",
            layer_index=layer_idx,
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
        topk_indices = logits.topk(max_k, dim=-1).indices.detach().cpu()
        topk_by_layer[layer_idx] = topk_indices
        del h_norm, logits
    return topk_by_layer


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def _sanitize_topk_values(topk_values: Sequence[int], required_top_k: int) -> List[int]:
    values = sorted({int(value) for value in topk_values if int(value) > 0})
    if not values:
        values = [required_top_k]
    if values[-1] < required_top_k:
        values.append(int(required_top_k))
    return values


def _compute_topk_details(
    *,
    tokenizer: Any,
    topk_ids_a: List[int],
    topk_ids_b: List[int],
    k: int,
) -> Dict[str, Any]:
    ids_a = [int(token_id) for token_id in topk_ids_a[:k]]
    ids_b = [int(token_id) for token_id in topk_ids_b[:k]]
    set_a = set(ids_a)
    set_b = set(ids_b)
    shared = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    union = set_a | set_b
    jaccard = len(shared) / len(union) if union else 1.0
    return {
        "k": int(k),
        "base_token_ids": ids_a,
        "base_tokens": [_decode_token(tokenizer, token_id) for token_id in ids_a],
        "finetuned_token_ids": ids_b,
        "finetuned_tokens": [_decode_token(tokenizer, token_id) for token_id in ids_b],
        "shared_token_ids": sorted(shared),
        "shared_tokens": [_decode_token(tokenizer, token_id) for token_id in sorted(shared)],
        "base_only_token_ids": sorted(only_a),
        "base_only_tokens": [_decode_token(tokenizer, token_id) for token_id in sorted(only_a)],
        "finetuned_only_token_ids": sorted(only_b),
        "finetuned_only_tokens": [_decode_token(tokenizer, token_id) for token_id in sorted(only_b)],
        "jaccard": round(jaccard, 4),
    }


def _compare_topk(
    *,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_len: int,
    top_k: int,
    comparison_top_ks: Sequence[int],
    layer_rel: float,
    layer_abs: int,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    base_generated_ids: torch.Tensor | None = None,
    ft_generated_ids: torch.Tensor | None = None,
) -> Dict[str, Any]:
    seq_ids = input_ids[0].detach().cpu()
    valid_len = (
        int(attention_mask[0].sum().item())
        if attention_mask is not None
        else int(seq_ids.shape[0])
    )
    layer_logits_a = logits_a[0]
    layer_logits_b = logits_b[0]
    base_seq_ids = (
        base_generated_ids[0].detach().cpu() if base_generated_ids is not None else seq_ids
    )
    ft_seq_ids = (
        ft_generated_ids[0].detach().cpu() if ft_generated_ids is not None else seq_ids
    )

    positions: List[Dict[str, Any]] = []
    valid_top_ks = _sanitize_topk_values(comparison_top_ks, top_k)
    max_k = max(valid_top_ks)
    for pos in range(valid_len):
        token_id = int(seq_ids[pos].item())
        topk_out_a = layer_logits_a[pos].topk(max_k)
        topk_out_b = layer_logits_b[pos].topk(max_k)
        topk_ids_a = [int(token_id) for token_id in topk_out_a.indices.tolist()]
        topk_ids_b = [int(token_id) for token_id in topk_out_b.indices.tolist()]
        per_k = {
            str(k): _compute_topk_details(
                tokenizer=tokenizer,
                topk_ids_a=topk_ids_a,
                topk_ids_b=topk_ids_b,
                k=k,
            )
            for k in valid_top_ks
        }
        primary = per_k[str(int(top_k))]
        base_top1_id = int(topk_ids_a[0])
        ft_top1_id = int(topk_ids_b[0])

        positions.append(
            {
                "position": pos,
                "position_kind": "generated" if pos >= prompt_len else "prompt",
                "input_token": _decode_token(tokenizer, token_id),
                "input_token_id": token_id,
                "base_generated_token": _decode_token(tokenizer, int(base_seq_ids[pos].item())),
                "base_generated_token_id": int(base_seq_ids[pos].item()),
                "ft_generated_token": _decode_token(tokenizer, int(ft_seq_ids[pos].item())),
                "ft_generated_token_id": int(ft_seq_ids[pos].item()),
                "base_top1_token": _decode_token(tokenizer, base_top1_id),
                "base_top1_token_id": base_top1_id,
                "ft_top1_token": _decode_token(tokenizer, ft_top1_id),
                "ft_top1_token_id": ft_top1_id,
                "top1_match": base_top1_id == ft_top1_id,
                "base_top5_tokens": per_k.get("5", {}).get("base_tokens", []),
                "base_top5_token_ids": per_k.get("5", {}).get("base_token_ids", []),
                "ft_top5_tokens": per_k.get("5", {}).get("finetuned_tokens", []),
                "ft_top5_token_ids": per_k.get("5", {}).get("finetuned_token_ids", []),
                "base_top10_tokens": per_k.get("10", {}).get("base_tokens", []),
                "base_top10_token_ids": per_k.get("10", {}).get("base_token_ids", []),
                "ft_top10_tokens": per_k.get("10", {}).get("finetuned_tokens", []),
                "ft_top10_token_ids": per_k.get("10", {}).get("finetuned_token_ids", []),
                "topk_predictions": per_k,
                "is_generated": pos >= prompt_len,
                "iou": primary["jaccard"],
                "intersection": primary["shared_tokens"],
                "only_base": primary["base_only_tokens"],
                "only_finetuned": primary["finetuned_only_tokens"],
                "num_intersection": len(primary["shared_token_ids"]),
                "num_only_base": len(primary["base_only_token_ids"]),
                "num_only_finetuned": len(primary["finetuned_only_token_ids"]),
                "top1_jaccard": per_k.get("1", {}).get("jaccard"),
                "top5_jaccard": per_k.get("5", {}).get("jaccard"),
                "top10_jaccard": per_k.get("10", {}).get("jaccard"),
            }
        )

    ious = [pos["iou"] for pos in positions]
    return {
        "layer_relative": round(layer_rel, 4),
        "layer_absolute": layer_abs,
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
        "positions": positions,
    }


def _compare_topk_predictions(
    *,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_len: int,
    top_k: int,
    comparison_top_ks: Sequence[int],
    layer_rel: float,
    layer_abs: int,
    topk_ids_a: torch.Tensor,
    topk_ids_b: torch.Tensor,
    base_generated_ids: torch.Tensor | None = None,
    ft_generated_ids: torch.Tensor | None = None,
) -> Dict[str, Any]:
    seq_ids = input_ids[0].detach().cpu()
    base_valid_len = (
        int(attention_mask[0].sum().item())
        if attention_mask is not None
        else int(seq_ids.shape[0])
    )
    layer_topk_a = topk_ids_a[0]
    layer_topk_b = topk_ids_b[0]
    base_seq_ids = (
        base_generated_ids[0].detach().cpu() if base_generated_ids is not None else seq_ids
    )
    ft_seq_ids = (
        ft_generated_ids[0].detach().cpu() if ft_generated_ids is not None else seq_ids
    )
    ft_valid_len = int(ft_seq_ids.shape[0])
    valid_len = min(
        int(base_valid_len),
        int(layer_topk_a.shape[0]),
        int(layer_topk_b.shape[0]),
        int(base_seq_ids.shape[0]),
        int(ft_seq_ids.shape[0]),
    )

    positions: List[Dict[str, Any]] = []
    valid_top_ks = _sanitize_topk_values(comparison_top_ks, top_k)
    for pos in range(valid_len):
        token_id = int(seq_ids[pos].item())
        topk_list_a = [int(token_id) for token_id in layer_topk_a[pos].tolist()]
        topk_list_b = [int(token_id) for token_id in layer_topk_b[pos].tolist()]
        per_k = {
            str(k): _compute_topk_details(
                tokenizer=tokenizer,
                topk_ids_a=topk_list_a,
                topk_ids_b=topk_list_b,
                k=k,
            )
            for k in valid_top_ks
        }
        primary = per_k[str(int(top_k))]
        base_top1_id = int(topk_list_a[0])
        ft_top1_id = int(topk_list_b[0])

        positions.append(
            {
                "position": pos,
                "position_kind": "generated" if pos >= prompt_len else "prompt",
                "input_token": _decode_token(tokenizer, token_id),
                "input_token_id": token_id,
                "base_generated_token": _decode_token(tokenizer, int(base_seq_ids[pos].item())),
                "base_generated_token_id": int(base_seq_ids[pos].item()),
                "ft_generated_token": _decode_token(tokenizer, int(ft_seq_ids[pos].item())),
                "ft_generated_token_id": int(ft_seq_ids[pos].item()),
                "base_top1_token": _decode_token(tokenizer, base_top1_id),
                "base_top1_token_id": base_top1_id,
                "ft_top1_token": _decode_token(tokenizer, ft_top1_id),
                "ft_top1_token_id": ft_top1_id,
                "top1_match": base_top1_id == ft_top1_id,
                "base_top5_tokens": per_k.get("5", {}).get("base_tokens", []),
                "base_top5_token_ids": per_k.get("5", {}).get("base_token_ids", []),
                "ft_top5_tokens": per_k.get("5", {}).get("finetuned_tokens", []),
                "ft_top5_token_ids": per_k.get("5", {}).get("finetuned_token_ids", []),
                "base_top10_tokens": per_k.get("10", {}).get("base_tokens", []),
                "base_top10_token_ids": per_k.get("10", {}).get("base_token_ids", []),
                "ft_top10_tokens": per_k.get("10", {}).get("finetuned_tokens", []),
                "ft_top10_token_ids": per_k.get("10", {}).get("finetuned_token_ids", []),
                "topk_predictions": per_k,
                "is_generated": pos >= prompt_len,
                "iou": primary["jaccard"],
                "intersection": primary["shared_tokens"],
                "only_base": primary["base_only_tokens"],
                "only_finetuned": primary["finetuned_only_tokens"],
                "num_intersection": len(primary["shared_token_ids"]),
                "num_only_base": len(primary["base_only_token_ids"]),
                "num_only_finetuned": len(primary["finetuned_only_token_ids"]),
                "top1_jaccard": per_k.get("1", {}).get("jaccard"),
                "top5_jaccard": per_k.get("5", {}).get("jaccard"),
                "top10_jaccard": per_k.get("10", {}).get("jaccard"),
            }
        )

    ious = [pos["iou"] for pos in positions]
    return {
        "layer_relative": round(layer_rel, 4),
        "layer_absolute": layer_abs,
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
        "compared_sequence_length": int(valid_len),
        "base_sequence_length": int(base_valid_len),
        "ft_sequence_length": int(ft_valid_len),
        "positions": positions,
    }


def _build_rich_payload(
    *,
    config: LogitDiffRunConfig,
    tokenizer_chat_template: str | None,
    base_model_name: str,
    finetuned_model_name: str,
    adapter_path: str | None,
    base_quantized: bool,
    finetuned_quantized: bool,
    effective_system_prompt: str | None,
    prompt_records: List[Dict[str, Any]],
    analysis_rows: List[Dict[str, Any]],
    results: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "metadata": {
            "label": config.label,
            "top_k": config.top_k,
            "comparison_top_ks": list(config.comparison_top_ks),
            "max_new_tokens": config.max_new_tokens,
            "norm_mode": config.norm_mode,
            "template_name": config.template_name,
            "use_chat_template": config.use_chat_template,
            "prompt_format": config.prompt_format,
            "system_prompt": config.system_prompt,
            "effective_system_prompt": effective_system_prompt,
            "chat_template": tokenizer_chat_template,
            "generation": {
                "do_sample": config.do_sample,
                "temperature": config.temperature,
                "use_cache": config.use_cache,
                "output_hidden_states": config.output_hidden_states,
                "seed": config.seed,
            },
            "layers": list(config.layers),
            "base_model_name": base_model_name,
            "finetuned_model_name": finetuned_model_name,
            "adapter_path": adapter_path,
            "do_sample": config.do_sample,
            "temperature": config.temperature,
            "use_cache": config.use_cache,
            "base_quantized": base_quantized,
            "finetuned_quantized": finetuned_quantized,
            "config": _to_serializable(asdict(config)),
        },
        "prompts": prompt_records,
        "analysis_rows": analysis_rows,
        "results": results,
    }


@torch.no_grad()
def run_logitdiff(
    arch_wrappers: tuple[LogitLensWrapper, LogitLensWrapper],
    config: LogitDiffRunConfig,
    *,
    output_path: str | Path | None = None,
    responses_output_path: str | Path | None = None,
) -> Dict[str, Any]:
    wrapper_a, wrapper_b = arch_wrappers
    _validate_tokenizers(wrapper_a, wrapper_b)
    if config.prompt_metadata is not None and len(config.prompt_metadata) != len(config.prompts):
        raise ValueError("prompt_metadata must have the same length as prompts")

    resolved_layers = _resolve_layer_indices(wrapper_a, config.layers)
    layer_indices = [abs_idx for _, abs_idx in resolved_layers]

    results: Dict[str, Any] = {}
    analysis_rows: List[Dict[str, Any]] = []
    effective_system_prompt: str | None = None
    prompt_records: List[Dict[str, Any]] = []
    response_rows: List[Dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(config.prompts):
        prompt_meta = (
            dict(config.prompt_metadata[prompt_idx])
            if config.prompt_metadata is not None
            else {}
        )
        generated_ids, prompt_len, attention_mask, prompt_formatted = _generate_sequence(
            wrapper=wrapper_a,
            prompt=prompt,
            add_special_tokens=config.add_special_tokens,
            prompt_format=config.prompt_format,
            use_chat_template=config.use_chat_template,
            system_prompt=config.system_prompt,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            use_cache=config.use_cache,
            output_hidden_states=config.output_hidden_states,
            seed=config.seed,
        )
        ft_generated_ids, _, _, _ = _generate_sequence(
            wrapper=wrapper_b,
            prompt=prompt,
            add_special_tokens=config.add_special_tokens,
            prompt_format=config.prompt_format,
            use_chat_template=config.use_chat_template,
            system_prompt=config.system_prompt,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            use_cache=config.use_cache,
            output_hidden_states=config.output_hidden_states,
            seed=None if config.seed is None else int(config.seed) + 1,
        )

        if effective_system_prompt is None and config.prompt_format == "chat_template":
            effective_system_prompt = _extract_effective_system_prompt(prompt_formatted)

        prompt_record = {
            "prompt_index": prompt_idx,
            "prompt_id": prompt_meta.get("prompt_id", prompt_meta.get("id", prompt_idx)),
            "prompt": prompt,
            "prompt_rendered": prompt_formatted,
            "prompt_formatted": prompt_formatted,
            "prompt_length": prompt_len,
            "sequence_length": int(attention_mask[0].sum().item()),
            "template_name": config.template_name,
            "prompt_format": config.prompt_format,
            "use_chat_template": config.use_chat_template,
            "system_prompt": config.system_prompt,
            "effective_system_prompt": effective_system_prompt,
            "metadata": prompt_meta,
        }
        prompt_records.append(prompt_record)
        base_response = wrapper_a.tokenizer.decode(
            generated_ids[0, prompt_len:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        comparison_response = wrapper_b.tokenizer.decode(
            ft_generated_ids[0, prompt_len:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        response_rows.append(
            {
                "prompt_index": prompt_idx,
                "prompt_id": prompt_record["prompt_id"],
                "prompt": prompt,
                "template_name": config.template_name,
                "max_new_tokens": config.max_new_tokens,
                "base_response": base_response,
                "comparison_response": comparison_response,
            }
        )

        logits_a = _collect_layer_logits(
            wrapper=wrapper_a,
            input_ids=generated_ids,
            attention_mask=attention_mask,
            layer_indices=layer_indices,
            norm_mode=config.norm_mode,
        )
        logits_b = _collect_layer_logits(
            wrapper=wrapper_b,
            input_ids=generated_ids.to(wrapper_b.model_device),
            attention_mask=attention_mask.to(wrapper_b.model_device),
            layer_indices=layer_indices,
            norm_mode=config.norm_mode,
        )

        for layer_rel, layer_abs in resolved_layers:
            key = str(layer_rel)
            entry = _compare_topk(
                tokenizer=wrapper_a.tokenizer,
                input_ids=generated_ids,
                attention_mask=attention_mask,
                prompt_len=prompt_len,
                top_k=config.top_k,
                comparison_top_ks=config.comparison_top_ks,
                layer_rel=layer_rel,
                layer_abs=layer_abs,
                logits_a=logits_a[layer_abs],
                logits_b=logits_b[layer_abs],
                base_generated_ids=generated_ids,
                ft_generated_ids=ft_generated_ids,
            )
            entry["prompt"] = prompt
            entry["prompt_index"] = prompt_idx
            entry["prompt_id"] = prompt_record["prompt_id"]
            entry["prompt_rendered"] = prompt_formatted
            entry["prompt_formatted"] = prompt_formatted
            entry["prompt_length"] = prompt_len
            entry["sequence_length"] = int(attention_mask[0].sum().item())
            entry["norm_mode"] = config.norm_mode
            entry["template_name"] = config.template_name
            entry["prompt_metadata"] = prompt_meta
            results.setdefault(key, []).append(entry)
            for position in entry["positions"]:
                analysis_rows.append(
                    {
                        "prompt_index": prompt_idx,
                        "prompt_id": prompt_record["prompt_id"],
                        "prompt": prompt,
                        "template_name": config.template_name,
                        "prompt_format": config.prompt_format,
                        "use_chat_template": config.use_chat_template,
                        "system_prompt": config.system_prompt,
                        "effective_system_prompt": effective_system_prompt,
                        "norm_mode": config.norm_mode,
                        "layer_relative": entry["layer_relative"],
                        "layer_absolute": entry["layer_absolute"],
                        "position": position["position"],
                        "position_kind": position["position_kind"],
                        "is_generated": position["is_generated"],
                        "input_token": position["input_token"],
                        "input_token_id": position["input_token_id"],
                        "base_generated_token": position["base_generated_token"],
                        "base_generated_token_id": position["base_generated_token_id"],
                        "ft_generated_token": position["ft_generated_token"],
                        "ft_generated_token_id": position["ft_generated_token_id"],
                        "base_top1_token": position["base_top1_token"],
                        "base_top1_token_id": position["base_top1_token_id"],
                        "ft_top1_token": position["ft_top1_token"],
                        "ft_top1_token_id": position["ft_top1_token_id"],
                        "top1_match": position["top1_match"],
                        "top1_jaccard": position["top1_jaccard"],
                        "top5_jaccard": position["top5_jaccard"],
                        "top10_jaccard": position["top10_jaccard"],
                        "iou": position["iou"],
                        "topk_predictions": position["topk_predictions"],
                    }
                )

    payload_rich = _build_rich_payload(
        config=config,
        tokenizer_chat_template=getattr(wrapper_a.tokenizer, "chat_template", None),
        base_model_name=getattr(wrapper_a.model, "name_or_path", "unknown"),
        finetuned_model_name=getattr(wrapper_b.model, "name_or_path", "unknown"),
        adapter_path=None,
        base_quantized=wrapper_a.is_bnb_quantized,
        finetuned_quantized=wrapper_b.is_bnb_quantized,
        effective_system_prompt=effective_system_prompt,
        prompt_records=prompt_records,
        analysis_rows=analysis_rows,
        results=results,
    )

    payload: Dict[str, Any]
    if config.output_format == "toolkit_legacy":
        payload = _make_toolkit_legacy_payload(results)
    else:
        payload = payload_rich

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(payload), f, indent=2, ensure_ascii=False)
    if responses_output_path is not None:
        responses_path = Path(responses_output_path)
        responses_path.parent.mkdir(parents=True, exist_ok=True)
        with responses_path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(response_rows), f, indent=2, ensure_ascii=False)

    return payload


@torch.no_grad()
def run_logitdiff_sequential(
    wrapper_loader_a: Callable[[], LogitLensWrapper],
    wrapper_loader_b: Callable[[], LogitLensWrapper],
    config: LogitDiffRunConfig,
    *,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    wrapper_a_probe = wrapper_loader_a()
    wrapper_b_probe = wrapper_loader_b()
    try:
        _validate_tokenizers(wrapper_a_probe, wrapper_b_probe)
        resolved_layers = _resolve_layer_indices(wrapper_a_probe, config.layers)
        tokenizer = wrapper_a_probe.tokenizer
        tokenizer_chat_template = getattr(tokenizer, "chat_template", None)
        base_model_name = getattr(wrapper_a_probe.model, "name_or_path", "unknown")
        finetuned_model_name = getattr(wrapper_b_probe.model, "name_or_path", "unknown")
        base_quantized = wrapper_a_probe.is_bnb_quantized
        finetuned_quantized = wrapper_b_probe.is_bnb_quantized
    finally:
        del wrapper_a_probe, wrapper_b_probe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_indices = [abs_idx for _, abs_idx in resolved_layers]
    max_k = max(_sanitize_topk_values(config.comparison_top_ks, config.top_k))
    results: Dict[str, Any] = {}
    analysis_rows: List[Dict[str, Any]] = []
    prompt_records: List[Dict[str, Any]] = []
    effective_system_prompt: str | None = None

    if config.prompt_metadata is not None and len(config.prompt_metadata) != len(config.prompts):
        raise ValueError("prompt_metadata must have the same length as prompts")

    for prompt_idx, prompt in enumerate(config.prompts):
        prompt_meta = (
            dict(config.prompt_metadata[prompt_idx])
            if config.prompt_metadata is not None
            else {}
        )

        wrapper_a = wrapper_loader_a()
        try:
            generated_ids, prompt_len, attention_mask, prompt_formatted = _generate_sequence(
                wrapper=wrapper_a,
                prompt=prompt,
                add_special_tokens=config.add_special_tokens,
                prompt_format=config.prompt_format,
                use_chat_template=config.use_chat_template,
                system_prompt=config.system_prompt,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                use_cache=config.use_cache,
                output_hidden_states=config.output_hidden_states,
                seed=config.seed,
            )
            base_topk = _collect_layer_topk_predictions(
                wrapper=wrapper_a,
                input_ids=generated_ids,
                attention_mask=attention_mask,
                layer_indices=layer_indices,
                norm_mode=config.norm_mode,
                max_k=max_k,
            )
            if effective_system_prompt is None and config.prompt_format == "chat_template":
                effective_system_prompt = _extract_effective_system_prompt(prompt_formatted)
        finally:
            del wrapper_a
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        wrapper_b = wrapper_loader_b()
        try:
            ft_generated_ids, _, _, _ = _generate_sequence(
                wrapper=wrapper_b,
                prompt=prompt,
                add_special_tokens=config.add_special_tokens,
                prompt_format=config.prompt_format,
                use_chat_template=config.use_chat_template,
                system_prompt=config.system_prompt,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                use_cache=config.use_cache,
                output_hidden_states=config.output_hidden_states,
                seed=None if config.seed is None else int(config.seed) + 1,
            )
            ft_topk = _collect_layer_topk_predictions(
                wrapper=wrapper_b,
                input_ids=generated_ids.to(wrapper_b.model_device),
                attention_mask=attention_mask.to(wrapper_b.model_device),
                layer_indices=layer_indices,
                norm_mode=config.norm_mode,
                max_k=max_k,
            )
        finally:
            del wrapper_b
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        prompt_record = {
            "prompt_index": prompt_idx,
            "prompt_id": prompt_meta.get("prompt_id", prompt_meta.get("id", prompt_idx)),
            "prompt": prompt,
            "prompt_rendered": prompt_formatted,
            "prompt_formatted": prompt_formatted,
            "prompt_length": prompt_len,
            "sequence_length": int(attention_mask[0].sum().item()),
            "template_name": config.template_name,
            "prompt_format": config.prompt_format,
            "use_chat_template": config.use_chat_template,
            "system_prompt": config.system_prompt,
            "effective_system_prompt": effective_system_prompt,
            "metadata": prompt_meta,
        }
        prompt_records.append(prompt_record)

        for layer_rel, layer_abs in resolved_layers:
            key = str(layer_rel)
            entry = _compare_topk_predictions(
                tokenizer=tokenizer,
                input_ids=generated_ids,
                attention_mask=attention_mask,
                prompt_len=prompt_len,
                top_k=config.top_k,
                comparison_top_ks=config.comparison_top_ks,
                layer_rel=layer_rel,
                layer_abs=layer_abs,
                topk_ids_a=base_topk[layer_abs],
                topk_ids_b=ft_topk[layer_abs],
                base_generated_ids=generated_ids,
                ft_generated_ids=ft_generated_ids,
            )
            entry["prompt"] = prompt
            entry["prompt_index"] = prompt_idx
            entry["prompt_id"] = prompt_record["prompt_id"]
            entry["prompt_rendered"] = prompt_formatted
            entry["prompt_formatted"] = prompt_formatted
            entry["prompt_length"] = prompt_len
            entry["sequence_length"] = int(attention_mask[0].sum().item())
            entry["norm_mode"] = config.norm_mode
            entry["template_name"] = config.template_name
            entry["prompt_metadata"] = prompt_meta
            results.setdefault(key, []).append(entry)
            for position in entry["positions"]:
                analysis_rows.append(
                    {
                        "prompt_index": prompt_idx,
                        "prompt_id": prompt_record["prompt_id"],
                        "prompt": prompt,
                        "template_name": config.template_name,
                        "prompt_format": config.prompt_format,
                        "use_chat_template": config.use_chat_template,
                        "system_prompt": config.system_prompt,
                        "effective_system_prompt": effective_system_prompt,
                        "norm_mode": config.norm_mode,
                        "layer_relative": entry["layer_relative"],
                        "layer_absolute": entry["layer_absolute"],
                        "position": position["position"],
                        "position_kind": position["position_kind"],
                        "is_generated": position["is_generated"],
                        "input_token": position["input_token"],
                        "input_token_id": position["input_token_id"],
                        "base_generated_token": position["base_generated_token"],
                        "base_generated_token_id": position["base_generated_token_id"],
                        "ft_generated_token": position["ft_generated_token"],
                        "ft_generated_token_id": position["ft_generated_token_id"],
                        "base_top1_token": position["base_top1_token"],
                        "base_top1_token_id": position["base_top1_token_id"],
                        "ft_top1_token": position["ft_top1_token"],
                        "ft_top1_token_id": position["ft_top1_token_id"],
                        "top1_match": position["top1_match"],
                        "top1_jaccard": position["top1_jaccard"],
                        "top5_jaccard": position["top5_jaccard"],
                        "top10_jaccard": position["top10_jaccard"],
                        "iou": position["iou"],
                        "topk_predictions": position["topk_predictions"],
                    }
                )

        del generated_ids, ft_generated_ids, attention_mask, base_topk, ft_topk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload_rich = _build_rich_payload(
        config=config,
        tokenizer_chat_template=tokenizer_chat_template,
        base_model_name=base_model_name,
        finetuned_model_name=finetuned_model_name,
        adapter_path=None,
        base_quantized=base_quantized,
        finetuned_quantized=finetuned_quantized,
        effective_system_prompt=effective_system_prompt,
        prompt_records=prompt_records,
        analysis_rows=analysis_rows,
        results=results,
    )

    payload: Dict[str, Any]
    if config.output_format == "toolkit_legacy":
        payload = _make_toolkit_legacy_payload(results)
    else:
        payload = payload_rich

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(payload), f, indent=2, ensure_ascii=False)

    return payload


def run_logitdiff_repeated(
    arch_wrappers: tuple[LogitLensWrapper, LogitLensWrapper],
    config: LogitDiffRunConfig,
    *,
    num_runs: int,
    base_seed: int = 0,
    seed_stride: int = 1,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")
    if seed_stride < 1:
        raise ValueError(f"seed_stride must be >= 1, got {seed_stride}")

    runs: List[Dict[str, Any]] = []
    for run_idx in range(num_runs):
        run_seed = int(base_seed) + run_idx * int(seed_stride)
        run_config = replace(
            config,
            seed=run_seed,
            metadata={
                **config.metadata,
                "run_idx": run_idx,
                "run_seed": run_seed,
                "num_runs_total": num_runs,
            },
        )
        run_payload = run_logitdiff(arch_wrappers, run_config, output_path=None)
        if isinstance(run_payload, dict) and "metadata" in run_payload:
            run_payload["metadata"]["run_idx"] = run_idx
            run_payload["metadata"]["run_seed"] = run_seed
        runs.append(run_payload)

    payload = {
        "metadata": {
            "label": config.label,
            "collection_type": "repeated_logitdiff_runs",
            "num_runs": num_runs,
            "base_seed": int(base_seed),
            "seed_stride": int(seed_stride),
            "config": _to_serializable(asdict(config)),
        },
        "runs": runs,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(payload), f, indent=2, ensure_ascii=False)

    return payload


def build_model_organisms_logitdiff_path(
    root_dir: str | Path,
    base_model_name: str,
    organism_name: str,
    *,
    top_k: int,
    prompt_examples_subdir: str | None = None,
) -> Path:
    root = Path(root_dir)
    if prompt_examples_subdir:
        return (
            root
            / prompt_examples_subdir
            / base_model_name
            / organism_name
            / "logitdiff"
            / f"logitdiff_results_k{top_k}.json"
        )
    return (
        root
        / "diffing_results"
        / base_model_name
        / organism_name
        / "logitdiff"
        / f"logitdiff_results_k{top_k}.json"
    )


def save_model_organisms_logitdiff(
    arch_wrappers: tuple[LogitLensWrapper, LogitLensWrapper],
    config: LogitDiffRunConfig,
    *,
    root_dir: str | Path,
    base_model_name: str,
    organism_name: str,
    prompt_examples_subdir: str | None = None,
) -> Path:
    path = build_model_organisms_logitdiff_path(
        root_dir=root_dir,
        base_model_name=base_model_name,
        organism_name=organism_name,
        top_k=config.top_k,
        prompt_examples_subdir=prompt_examples_subdir,
    )
    run_logitdiff(
        arch_wrappers,
        LogitDiffRunConfig(
            **{
                **asdict(config),
                "output_format": "toolkit_legacy",
            }
        ),
        output_path=path,
    )
    return path

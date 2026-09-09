from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from .collectors.collect_prompt_lens_logits import collect_logits_for_plotter
from .wrappers import LogitLensWrapper, generate_with_model, lmhead_project, normalize_activations


def _validate_matching_tokenizers(
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


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def _decode_ids(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    return [_decode_token(tokenizer, token_id) for token_id in token_ids]


def _sanitize_topk_values(topk_values: Sequence[int], required_top_k: int) -> list[int]:
    values = sorted({int(value) for value in topk_values if int(value) > 0})
    if not values:
        values = [required_top_k]
    if values[-1] < required_top_k:
        values.append(int(required_top_k))
    return values


def _compute_topk_details(
    *,
    tokenizer: Any,
    topk_ids_a: list[int],
    topk_ids_b: list[int],
    k: int,
) -> dict[str, Any]:
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
        "base_tokens": _decode_ids(tokenizer, ids_a),
        "finetuned_token_ids": ids_b,
        "finetuned_tokens": _decode_ids(tokenizer, ids_b),
        "shared_token_ids": sorted(shared),
        "shared_tokens": _decode_ids(tokenizer, sorted(shared)),
        "base_only_token_ids": sorted(only_a),
        "base_only_tokens": _decode_ids(tokenizer, sorted(only_a)),
        "finetuned_only_token_ids": sorted(only_b),
        "finetuned_only_tokens": _decode_ids(tokenizer, sorted(only_b)),
        "jaccard": round(jaccard, 4),
    }


def _collect_layer_logits(
    wrapper: LogitLensWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    layer_indices: Sequence[int],
    norm_mode: str,
) -> dict[int, torch.Tensor]:
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

    logits_by_layer: dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        if layer_idx not in block_names:
            raise KeyError(f"Layer {layer_idx} was not captured by hooks.")
        hidden = acts[block_names[layer_idx]]
        normalized = normalize_activations(
            x=hidden.clone(),
            mode=norm_mode,
            block="block",
            layer_index=layer_idx,
            model_device=wrapper.model_device,
            model_dtype=wrapper.model_dtype,
            final_norm=wrapper.final_norm,
        )
        logits, _ = lmhead_project(
            x=normalized,
            lm_head=wrapper.lm_head,
            stable=wrapper.stable,
            model_device=wrapper.model_device,
        )
        logits_by_layer[layer_idx] = logits.detach().cpu()
    return logits_by_layer


def _build_prompt_payload_from_results(
    result_base: dict[str, Any],
    result_compare: dict[str, Any],
    *,
    top_k: int,
    readout_mode: str,
    base_model_name: str,
    finetuned_model_name: str,
) -> dict[str, Any]:
    target_a = result_base["target_ids"]
    target_b = result_compare["target_ids"]
    if len(target_a) != len(target_b) or not torch.equal(target_a, target_b):
        raise ValueError("Prompt LogitDiff requires identical tokenization between the two models.")

    layer_indices = sorted(layer for layer, mode in result_base["logits"] if mode == readout_mode)
    results: dict[str, list[dict[str, Any]]] = {}
    total_layers = len(layer_indices)
    tokenizer = result_base["tokenizer"]

    for order_idx, layer_idx in enumerate(layer_indices):
        logits_base = result_base["logits"][(layer_idx, readout_mode)]
        logits_compare = result_compare["logits"][(layer_idx, readout_mode)]
        seq_len = min(logits_base.shape[0], logits_compare.shape[0], len(result_base["tokens"]))
        positions: list[dict[str, Any]] = []
        ious: list[float] = []
        for pos_idx in range(seq_len):
            ids_base = torch.topk(
                logits_base[pos_idx],
                k=min(top_k, logits_base.shape[-1]),
                dim=-1,
            ).indices.tolist()
            ids_compare = torch.topk(
                logits_compare[pos_idx],
                k=min(top_k, logits_compare.shape[-1]),
                dim=-1,
            ).indices.tolist()
            set_base = set(int(v) for v in ids_base)
            set_compare = set(int(v) for v in ids_compare)
            union = set_base | set_compare
            inter = set_base & set_compare
            only_base = sorted(set_base - set_compare)
            only_compare = sorted(set_compare - set_base)
            iou = 0.0 if not union else len(inter) / len(union)
            ious.append(float(iou))
            positions.append(
                {
                    "position": pos_idx,
                    "input_token": result_base["tokens"][pos_idx],
                    "is_generated": False,
                    "iou": float(iou),
                    "intersection": _decode_ids(tokenizer, sorted(inter)),
                    "only_base": _decode_ids(tokenizer, only_base),
                    "only_finetuned": _decode_ids(tokenizer, only_compare),
                    "num_intersection": len(inter),
                    "num_only_base": len(only_base),
                    "num_only_finetuned": len(only_compare),
                }
            )

        results[str(layer_idx)] = [
            {
                "prompt": result_base["prompt"],
                "layer_relative": (order_idx / max(total_layers - 1, 1)),
                "layer_absolute": int(layer_idx),
                "mean_iou": float(sum(ious) / len(ious)) if ious else 0.0,
                "positions": positions,
            }
        ]

    return {
        "metadata": {
            "base_model_name": base_model_name,
            "finetuned_model_name": finetuned_model_name,
            "top_k": int(top_k),
            "readout_mode": readout_mode,
        },
        "results": results,
    }


@torch.no_grad()
def build_live_prompt_logitdiff_payload(
    base_wrapper: LogitLensWrapper,
    compare_wrapper: LogitLensWrapper,
    *,
    prompt: str,
    readout_mode: str = "model_norm",
    top_k: int = 5,
    add_special_tokens: bool = True,
    force_include_input: bool = True,
    force_include_output: bool = True,
) -> dict[str, Any]:
    _validate_matching_tokenizers(base_wrapper, compare_wrapper)

    result_base = collect_logits_for_plotter(
        arch_wrapper=base_wrapper,
        prompt=prompt,
        mode=readout_mode,
        topk=top_k,
        selected_layers=None,
        add_special_tokens=add_special_tokens,
        force_include_input=force_include_input,
        force_include_output=force_include_output,
    )
    result_compare = collect_logits_for_plotter(
        arch_wrapper=compare_wrapper,
        prompt=prompt,
        mode=readout_mode,
        topk=top_k,
        selected_layers=None,
        add_special_tokens=add_special_tokens,
        force_include_input=force_include_input,
        force_include_output=force_include_output,
    )

    return _build_prompt_payload_from_results(
        result_base,
        result_compare,
        top_k=top_k,
        readout_mode=readout_mode,
        base_model_name=getattr(base_wrapper.model, "name_or_path", "Base"),
        finetuned_model_name=getattr(compare_wrapper.model, "name_or_path", "Comparison"),
    )


def _tokenize_prompt(
    wrapper: LogitLensWrapper,
    prompt: str,
    *,
    add_special_tokens: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = wrapper.tokenize_inputs(
        texts=prompt,
        device=wrapper.model_device,
        add_special_tokens=add_special_tokens,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask


@torch.no_grad()
def _generate_sequence(
    wrapper: LogitLensWrapper,
    prompt: str,
    *,
    add_special_tokens: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    seed: int | None,
) -> tuple[torch.Tensor, int]:
    input_ids, attention_mask = _tokenize_prompt(
        wrapper,
        prompt,
        add_special_tokens=add_special_tokens,
    )
    prompt_len = int(attention_mask[0].sum().item())
    if max_new_tokens <= 0:
        return input_ids, prompt_len
    tokens, _ = generate_with_model(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        use_cache=False,
        seed=seed,
    )
    return tokens, prompt_len


def _build_generation_entry(
    *,
    tokenizer: Any,
    input_ids: torch.Tensor,
    prompt_len: int,
    top_k: int,
    comparison_top_ks: Sequence[int],
    layer_rel: float,
    layer_abs: int,
    logits_base: torch.Tensor,
    logits_compare: torch.Tensor,
    base_generated_ids: torch.Tensor,
    compare_generated_ids: torch.Tensor,
) -> dict[str, Any]:
    seq_ids = input_ids[0].detach().cpu()
    base_seq_ids = base_generated_ids[0].detach().cpu()
    compare_seq_ids = compare_generated_ids[0].detach().cpu()
    valid_len = min(
        int(seq_ids.shape[0]),
        int(logits_base.shape[1]),
        int(logits_compare.shape[1]),
        int(base_seq_ids.shape[0]),
        int(compare_seq_ids.shape[0]),
    )
    per_k_values = _sanitize_topk_values(comparison_top_ks, top_k)
    max_k = max(per_k_values)
    positions: list[dict[str, Any]] = []
    ious: list[float] = []

    for pos in range(valid_len):
        ids_base = [int(v) for v in logits_base[0, pos].topk(max_k).indices.tolist()]
        ids_compare = [int(v) for v in logits_compare[0, pos].topk(max_k).indices.tolist()]
        per_k = {
            str(k): _compute_topk_details(
                tokenizer=tokenizer,
                topk_ids_a=ids_base,
                topk_ids_b=ids_compare,
                k=k,
            )
            for k in per_k_values
        }
        primary = per_k[str(int(top_k))]
        base_top1_id = ids_base[0]
        compare_top1_id = ids_compare[0]
        ious.append(float(primary["jaccard"]))
        positions.append(
            {
                "position": pos,
                "position_kind": "generated" if pos >= prompt_len else "prompt",
                "input_token": _decode_token(tokenizer, int(seq_ids[pos].item())),
                "input_token_id": int(seq_ids[pos].item()),
                "base_generated_token": _decode_token(tokenizer, int(base_seq_ids[pos].item())),
                "base_generated_token_id": int(base_seq_ids[pos].item()),
                "ft_generated_token": _decode_token(tokenizer, int(compare_seq_ids[pos].item())),
                "ft_generated_token_id": int(compare_seq_ids[pos].item()),
                "base_top1_token": _decode_token(tokenizer, base_top1_id),
                "base_top1_token_id": base_top1_id,
                "ft_top1_token": _decode_token(tokenizer, compare_top1_id),
                "ft_top1_token_id": compare_top1_id,
                "top1_match": base_top1_id == compare_top1_id,
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

    return {
        "layer_relative": round(layer_rel, 4),
        "layer_absolute": int(layer_abs),
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
        "positions": positions,
    }


@torch.no_grad()
def build_live_generation_logitdiff_payload(
    base_wrapper: LogitLensWrapper,
    compare_wrapper: LogitLensWrapper,
    *,
    prompt: str,
    readout_mode: str = "model_norm",
    top_k: int = 5,
    max_new_tokens: int = 32,
    add_special_tokens: bool = True,
    do_sample: bool = False,
    temperature: float = 1.0,
    seed: int | None = None,
) -> dict[str, Any]:
    _validate_matching_tokenizers(base_wrapper, compare_wrapper)

    base_generated_ids, prompt_len = _generate_sequence(
        base_wrapper,
        prompt,
        add_special_tokens=add_special_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        seed=seed,
    )
    compare_generated_ids, _ = _generate_sequence(
        compare_wrapper,
        prompt,
        add_special_tokens=add_special_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        seed=None if seed is None else int(seed) + 1,
    )

    attention_mask = torch.ones_like(base_generated_ids, device=base_generated_ids.device)
    layer_indices = sorted(
        entry["idx"]
        for entry in base_wrapper.layer_registry.values()
        if entry["type"] == "block"
    )
    logits_base = _collect_layer_logits(
        base_wrapper,
        base_generated_ids,
        attention_mask,
        layer_indices=layer_indices,
        norm_mode=readout_mode,
    )
    logits_compare = _collect_layer_logits(
        compare_wrapper,
        base_generated_ids.to(compare_wrapper.model_device),
        attention_mask.to(compare_wrapper.model_device),
        layer_indices=layer_indices,
        norm_mode=readout_mode,
    )

    results: dict[str, list[dict[str, Any]]] = {}
    total_layers = len(layer_indices)
    tokenizer = base_wrapper.tokenizer
    for order_idx, layer_idx in enumerate(layer_indices):
        entry = _build_generation_entry(
            tokenizer=tokenizer,
            input_ids=base_generated_ids,
            prompt_len=prompt_len,
            top_k=top_k,
            comparison_top_ks=(1, 5, 10),
            layer_rel=(order_idx / max(total_layers - 1, 1)),
            layer_abs=layer_idx,
            logits_base=logits_base[layer_idx],
            logits_compare=logits_compare[layer_idx],
            base_generated_ids=base_generated_ids,
            compare_generated_ids=compare_generated_ids,
        )
        entry["prompt"] = prompt
        results[str(order_idx / max(total_layers - 1, 1))] = [entry]

    return {
        "metadata": {
            "base_model_name": getattr(base_wrapper.model, "name_or_path", "Base"),
            "finetuned_model_name": getattr(compare_wrapper.model, "name_or_path", "Comparison"),
            "top_k": int(top_k),
            "readout_mode": readout_mode,
            "max_new_tokens": int(max_new_tokens),
        },
        "results": results,
    }

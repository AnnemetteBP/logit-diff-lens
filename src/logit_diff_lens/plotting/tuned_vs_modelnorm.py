from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM, AutoTokenizer

from tuned_lens.nn.lenses import TunedLens

from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps import (
    logitdiff_gen_plotter as _GEN_PLOTTER,
)


def _find_final_norm(model) -> torch.nn.Module:
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise RuntimeError("Could not find final norm module for this model.")


def _clean_token(token: str) -> str:
    token = str(token)
    token = token.replace("Ġ", " ").replace("▁", " ")
    token = token.replace("\n", "\\n")
    return token.strip() or " "


def _decode_token(tokenizer, token_id: int) -> str:
    return _clean_token(
        tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    )


def _axis_token_label(token: str) -> str:
    cleaned = _clean_token(token)
    if cleaned == " ":
        return "[space]"
    return cleaned


def _display_prompt_text(prompt: str) -> str:
    lines = [line.strip() for line in str(prompt).splitlines() if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("assistant:"):
            break
        if ":" in line and line.split(":", 1)[0].strip().lower() in {
            "user",
            "assistant",
            "system",
        }:
            cleaned.append(line.split(":", 1)[1].strip())
        else:
            cleaned.append(line)
    return " ".join(cleaned).strip() if cleaned else str(prompt).strip()


def _topk_payload(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    tokenizer,
    *,
    top_k: int,
) -> dict[str, Any]:
    idx_a = torch.topk(logits_a, k=top_k, dim=-1).indices.tolist()
    idx_b = torch.topk(logits_b, k=top_k, dim=-1).indices.tolist()

    set_a = set(int(v) for v in idx_a)
    set_b = set(int(v) for v in idx_b)
    shared = [tok for tok in idx_a if tok in set_b]
    only_a = [tok for tok in idx_a if tok not in set_b]
    only_b = [tok for tok in idx_b if tok not in set_a]
    union = set_a | set_b
    jaccard = float(len(set_a & set_b) / len(union)) if union else 0.0

    return {
        "jaccard": jaccard,
        "shared_tokens": [_decode_token(tokenizer, tok) for tok in shared],
        "base_only_tokens": [_decode_token(tokenizer, tok) for tok in only_a],
        "finetuned_only_tokens": [_decode_token(tokenizer, tok) for tok in only_b],
        "shared_ids": shared,
        "base_only_ids": only_a,
        "finetuned_only_ids": only_b,
    }


def _distribution_metrics(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> dict[str, float]:
    log_probs_a = F.log_softmax(logits_a, dim=-1)
    log_probs_b = F.log_softmax(logits_b, dim=-1)
    probs_a = log_probs_a.exp()
    probs_b = log_probs_b.exp()
    mixture = 0.5 * (probs_a + probs_b)
    log_mixture = mixture.clamp_min(1e-12).log()

    kl_a_to_b = torch.sum(probs_a * (log_probs_a - log_probs_b)).item()
    kl_b_to_a = torch.sum(probs_b * (log_probs_b - log_probs_a)).item()
    kl_a_to_m = torch.sum(probs_a * (log_probs_a - log_mixture)).item()
    kl_b_to_m = torch.sum(probs_b * (log_probs_b - log_mixture)).item()
    jsd = 0.5 * (kl_a_to_m + kl_b_to_m)
    return {
        "kl_modelnorm_to_tuned": float(kl_a_to_b),
        "kl_tuned_to_modelnorm": float(kl_b_to_a),
        "jsd": float(jsd),
    }


def build_single_prompt_payload(
    *,
    model,
    tokenizer,
    tuned_lens: TunedLens,
    prompt: str,
    top_k: int,
    add_special_tokens: bool,
) -> dict[str, Any]:
    final_norm = _find_final_norm(model)
    lm_head = model.get_output_embeddings()

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    full_input_ids = encoded["input_ids"].to(device)
    full_attention_mask = encoded["attention_mask"].to(device)
    if full_input_ids.shape[1] < 2:
        raise ValueError("Prompt must tokenize to at least 2 tokens for next-token analysis.")

    input_ids = full_input_ids[:, :-1]
    attention_mask = full_attention_mask[:, :-1]
    target_ids = full_input_ids[:, 1:]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    num_layers = model.config.num_hidden_layers
    results: dict[str, list[dict[str, Any]]] = {}

    norm_param = next(final_norm.parameters())
    lm_head_param = next(lm_head.parameters())

    for layer_idx in range(num_layers):
        hidden = hidden_states[layer_idx + 1][0]
        model_norm_hidden = final_norm(
            hidden.to(device=norm_param.device, dtype=norm_param.dtype)
        )
        model_norm_logits = lm_head(
            model_norm_hidden.to(device=lm_head_param.device, dtype=lm_head_param.dtype)
        ).detach().float().cpu()
        tuned_logits = tuned_lens(
            hidden.to(device=device, dtype=model_dtype), layer_idx
        ).detach().float().cpu()

        positions: list[dict[str, Any]] = []
        for pos_idx in range(target_ids.shape[1]):
            target_token_id = int(target_ids[0, pos_idx])
            base_top1_id = int(model_norm_logits[pos_idx].argmax().item())
            tuned_top1_id = int(tuned_logits[pos_idx].argmax().item())
            input_token = _decode_token(tokenizer, int(input_ids[0, pos_idx]))
            target_token = _decode_token(tokenizer, target_token_id)
            topk = _topk_payload(
                model_norm_logits[pos_idx],
                tuned_logits[pos_idx],
                tokenizer,
                top_k=top_k,
            )
            distribution_metrics = _distribution_metrics(
                model_norm_logits[pos_idx],
                tuned_logits[pos_idx],
            )
            positions.append(
                {
                    "position": pos_idx,
                    "is_generated": False,
                    "input_token": input_token,
                    "target_token": target_token,
                    "target_token_id": target_token_id,
                    "base_generated_token": target_token,
                    "ft_generated_token": target_token,
                    "base_top1_token": _decode_token(tokenizer, base_top1_id),
                    "ft_top1_token": _decode_token(tokenizer, tuned_top1_id),
                    "iou": topk["jaccard"],
                    "jsd": distribution_metrics["jsd"],
                    "kl_modelnorm_to_tuned": distribution_metrics["kl_modelnorm_to_tuned"],
                    "kl_tuned_to_modelnorm": distribution_metrics["kl_tuned_to_modelnorm"],
                    "intersection": topk["shared_tokens"],
                    "only_base": topk["base_only_tokens"],
                    "only_finetuned": topk["finetuned_only_tokens"],
                    "correct_token_in_modelnorm_topk": (
                        target_token_id in topk["shared_ids"]
                        or target_token_id in topk["base_only_ids"]
                    ),
                    "correct_token_in_tuned_topk": (
                        target_token_id in topk["shared_ids"]
                        or target_token_id in topk["finetuned_only_ids"]
                    ),
                    "correct_token_in_both_topk": target_token_id in topk["shared_ids"],
                    "topk_predictions": {str(top_k): topk},
                }
            )

        results[str(layer_idx)] = [
            {
                "prompt": prompt,
                "layer_relative": layer_idx,
                "layer_absolute": layer_idx,
                "positions": positions,
            }
        ]

    return {
        "metadata": {
            "base_model_name": "ModelNorm Lens",
            "finetuned_model_name": "Tuned Lens",
            "top_k": top_k,
            "prompt_text": prompt,
        },
        "results": results,
    }


def _pair_positions(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(positions) < 1:
        raise ValueError("Need at least one position to build predictor/target token pairs.")
    return [
        {
            "predictor_position": int(position["position"]),
            "target_position": int(position["position"]) + 1,
            "predictor_token": _clean_token(position.get("input_token")),
            "target_token": _clean_token(position.get("target_token")),
        }
        for position in positions
    ]


def _format_token_block(tokens: list[str], max_items: int, max_chars: int) -> str:
    cleaned = [
        token[:max_chars] if len(token) <= max_chars else token[: max_chars - 1] + "…"
        for token in tokens[:max_items]
    ]
    return ", ".join(cleaned) if cleaned else "—"


def _build_hover_text(
    layer_result: dict[str, Any],
    predictor_position: dict[str, Any],
    target_token: str,
    metric_key: str,
    top_k: int,
    max_token_chars: int,
) -> str:
    shared = _format_token_block(
        predictor_position.get("intersection", []),
        top_k,
        max_token_chars,
    )
    only_base = _format_token_block(
        predictor_position.get("only_base", []),
        top_k,
        max_token_chars,
    )
    only_tuned = _format_token_block(
        predictor_position.get("only_finetuned", []),
        top_k,
        max_token_chars,
    )
    metric_labels = {
        "iou": "IoU",
        "jsd": "JSD",
        "kl_modelnorm_to_tuned": "KL(ModelNorm || Tuned)",
        "kl_tuned_to_modelnorm": "KL(Tuned || ModelNorm)",
    }
    correct_modelnorm = bool(
        predictor_position.get("correct_token_in_modelnorm_topk", False)
    )
    correct_tuned = bool(predictor_position.get("correct_token_in_tuned_topk", False))
    return (
        f"<b>Layer</b>: {layer_result['layer_relative']} (abs {layer_result['layer_absolute']})<br>"
        f"<b>Predictor token</b>: {_clean_token(predictor_position.get('input_token'))}<br>"
        f"<b>Target token</b>: {target_token}<br>"
        f"<b>Position</b>: {predictor_position['position']} → {predictor_position['position'] + 1}<br>"
        f"<b>{metric_labels[metric_key]}</b>: {float(predictor_position[metric_key]):.4f}<br>"
        f"<b>IoU</b>: {float(predictor_position['iou']):.4f}<br>"
        f"<b>JSD</b>: {float(predictor_position['jsd']):.4f}<br>"
        f"<b>KL(ModelNorm || Tuned)</b>: {float(predictor_position['kl_modelnorm_to_tuned']):.4f}<br>"
        f"<b>KL(Tuned || ModelNorm)</b>: {float(predictor_position['kl_tuned_to_modelnorm']):.4f}<br>"
        f"<b>Correct next token in ModelNorm top-{top_k}</b>: {'yes' if correct_modelnorm else 'no'}<br>"
        f"<b>Correct next token in Tuned top-{top_k}</b>: {'yes' if correct_tuned else 'no'}<br>"
        f"<b>Shared</b>: {shared}<br>"
        f"{only_base} <> {only_tuned}"
    )


def _choose_divergent_layers(
    per_layer_prompt_results: list[dict[str, Any]],
    usable_positions: list[dict[str, Any]],
    keep_last_fraction: float,
    max_layers: int,
    metric_key: str,
) -> list[int]:
    num_layers = len(per_layer_prompt_results)
    start_idx = int(np.floor(num_layers * (1.0 - keep_last_fraction)))
    candidate_indices = list(range(max(0, start_idx), num_layers))
    if not candidate_indices:
        candidate_indices = list(range(num_layers))

    position_keys = {int(position["position"]) for position in usable_positions}
    scored: list[tuple[float, int]] = []
    for layer_idx in candidate_indices:
        positions = [
            position
            for position in per_layer_prompt_results[layer_idx]["positions"]
            if int(position["position"]) in position_keys
        ]
        if not positions:
            continue
        mean_metric = float(np.mean([float(position[metric_key]) for position in positions]))
        scored.append((mean_metric, layer_idx))

    reverse = metric_key != "iou"
    scored.sort(key=lambda item: item[0], reverse=reverse)
    selected = [layer_idx for _, layer_idx in scored[:max_layers]]
    return sorted(
        selected,
        key=lambda idx: float(per_layer_prompt_results[idx]["layer_relative"]),
    )


def _prepare_next_token_heatmap_data(
    payload: dict[str, Any],
    *,
    metric_key: str,
    display_top_tokens: int,
    max_token_chars: int,
    selected_layer_indices: list[int] | None = None,
) -> dict[str, Any]:
    results = payload["results"]
    layer_keys = sorted(results.keys(), key=float)
    per_layer_prompt_results = [results[layer_key][0] for layer_key in layer_keys]
    if not per_layer_prompt_results:
        raise ValueError("No layer results found in payload.")

    all_positions = per_layer_prompt_results[0]["positions"]
    reference_positions = [
        position for position in all_positions if not bool(position.get("is_generated", False))
    ]
    if len(reference_positions) < 1:
        raise ValueError("No prompt positions available for plotting.")

    paired_positions = _pair_positions(reference_positions)
    selected_predictor_positions = [pair["predictor_position"] for pair in paired_positions]
    position_to_column = {
        position: idx for idx, position in enumerate(selected_predictor_positions)
    }

    if selected_layer_indices is None:
        selected_layer_indices = list(range(len(per_layer_prompt_results)))
    if not selected_layer_indices:
        raise ValueError("No layer indices selected for plotting.")

    selected_layer_results = [per_layer_prompt_results[idx] for idx in selected_layer_indices]
    num_layers = len(selected_layer_results)
    num_positions = len(paired_positions)
    z = np.full((num_layers, num_positions), np.nan, dtype=float)
    hover_text = np.empty((num_layers, num_positions), dtype=object)
    cell_parts = np.empty((num_layers, num_positions), dtype=object)
    cell_markers = np.empty((num_layers, num_positions), dtype=object)
    y_labels: list[str] = []
    total_model_layers = (
        max(int(layer_result["layer_absolute"]) for layer_result in per_layer_prompt_results) + 1
    )

    for row_idx, layer_result in enumerate(selected_layer_results):
        y_labels.append(
            f"Layer {int(layer_result['layer_absolute']) + 1}/{total_model_layers}"
        )
        layer_positions = {
            int(position["position"]): position for position in layer_result["positions"]
        }
        for pair in paired_positions:
            predictor_position = layer_positions.get(pair["predictor_position"])
            if predictor_position is None:
                continue
            col_idx = position_to_column[pair["predictor_position"]]
            z[row_idx, col_idx] = float(predictor_position[metric_key])
            hover_text[row_idx, col_idx] = _build_hover_text(
                layer_result,
                predictor_position,
                pair["target_token"],
                metric_key=metric_key,
                top_k=display_top_tokens,
                max_token_chars=max_token_chars,
            )
            cell_parts[row_idx, col_idx] = _GEN_PLOTTER._build_cell_parts(
                predictor_position,
                display_top_tokens=display_top_tokens,
                max_token_chars=max_token_chars,
            )
            correct_modelnorm = bool(
                predictor_position.get("correct_token_in_modelnorm_topk", False)
            )
            correct_tuned = bool(
                predictor_position.get("correct_token_in_tuned_topk", False)
            )
            if correct_modelnorm and correct_tuned:
                cell_markers[row_idx, col_idx] = "both"
            elif correct_modelnorm:
                cell_markers[row_idx, col_idx] = "modelnorm_only"
            elif correct_tuned:
                cell_markers[row_idx, col_idx] = "tuned_only"
            else:
                cell_markers[row_idx, col_idx] = "neither"

    return {
        "prompt": per_layer_prompt_results[0]["prompt"],
        "display_prompt": _display_prompt_text(per_layer_prompt_results[0]["prompt"]),
        "x_labels": [_axis_token_label(pair["predictor_token"]) for pair in paired_positions],
        "x_labels_secondary": [
            _axis_token_label(pair["target_token"]) for pair in paired_positions
        ],
        "x_positions": selected_predictor_positions,
        "y_labels": y_labels,
        "z": z,
        "hover_text": hover_text,
        "cell_parts": cell_parts,
        "cell_markers": cell_markers,
        "payload": payload,
    }


def plot_prompt_style_verification_heatmap(
    payload: dict[str, Any],
    *,
    metric: str,
    top_k: int,
    max_divergent_layers: int,
    keep_last_layer_fraction: float,
    max_token_chars: int,
    layer_mode: str = "all",
    title: str | None = None,
    colorscale: str = "RdBu",
) -> go.Figure:
    metric_key = metric
    display_top_tokens = min(top_k, 10)
    effective_max_token_chars = (
        min(max_token_chars, 10) if display_top_tokens <= 5 else max_token_chars
    )
    results = payload["results"]
    layer_keys = sorted(results.keys(), key=float)
    per_layer_prompt_results = [results[layer_key][0] for layer_key in layer_keys]
    first_positions = per_layer_prompt_results[0]["positions"]
    filtered_positions = [
        position for position in first_positions if not bool(position.get("is_generated", False))
    ]
    if layer_mode == "all":
        selected_layer_indices = list(range(len(per_layer_prompt_results)))
    elif layer_mode == "most_divergent":
        selected_layer_indices = _choose_divergent_layers(
            per_layer_prompt_results,
            usable_positions=filtered_positions[:-1],
            keep_last_fraction=keep_last_layer_fraction,
            max_layers=max_divergent_layers,
            metric_key=metric_key,
        )
    else:
        raise ValueError(f"Unsupported layer_mode: {layer_mode}")
    data = _prepare_next_token_heatmap_data(
        payload=payload,
        metric_key=metric_key,
        display_top_tokens=display_top_tokens,
        max_token_chars=effective_max_token_chars,
        selected_layer_indices=selected_layer_indices,
    )

    num_layers, num_positions = data["z"].shape
    max_x_label_len = max((len(label) for label in data["x_labels"]), default=1)
    max_y_label_len = max((len(label) for label in data["y_labels"]), default=1)
    shared_line_count = max(1, (display_top_tokens + 1) // 2)
    nonshared_line_count = max(1, display_top_tokens)
    line_count = shared_line_count + nonshared_line_count
    if display_top_tokens <= 1:
        layout_scale = "top1"
        annotation_font_size = max(22, min(26, int(192 / max(1, line_count))))
    elif display_top_tokens <= 5:
        layout_scale = "top5"
        annotation_font_size = max(18, min(22, int(214 / max(1, line_count))))
    else:
        layout_scale = "top10"
        annotation_font_size = max(17, min(21, int(228 / max(1, line_count))))

    longest_visible_token = 1
    longest_visible_line = 1
    for row_idx in range(num_layers):
        for col_idx in range(num_positions):
            parts = data["cell_parts"][row_idx, col_idx]
            if parts is None:
                continue
            visible_tokens = (
                parts["shared"][:display_top_tokens]
                + parts["base_only"][:display_top_tokens]
                + parts["finetuned_only"][:display_top_tokens]
            )
            if visible_tokens:
                longest_visible_token = max(
                    longest_visible_token,
                    max(len(token) for token in visible_tokens),
                )
            for idx in range(display_top_tokens):
                left = parts["base_only"][idx] if idx < len(parts["base_only"]) else "—"
                right = (
                    parts["finetuned_only"][idx]
                    if idx < len(parts["finetuned_only"])
                    else "—"
                )
                longest_visible_line = max(
                    longest_visible_line,
                    len(f"'{left}' <> '{right}'"),
                )

    base_cell_w = 48 + max(
        max_x_label_len * 5,
        longest_visible_token * 10,
        longest_visible_line * 7,
    )
    if layout_scale == "top1":
        cell_w = max(184, min(360, base_cell_w + 16))
        cell_h = max(96, int(line_count * (annotation_font_size * 1.18) + 12))
        bottom_margin = max(130, min(180, 82 + max_x_label_len * 3)) + 72
        top_margin = 190
    elif layout_scale == "top5":
        cell_w = max(170, min(348, base_cell_w + 8))
        cell_h = max(88, int(line_count * (annotation_font_size * 1.11) + 11))
        bottom_margin = max(130, min(180, 82 + max_x_label_len * 3)) + 60
        top_margin = 200
    else:
        cell_w = max(160, min(340, base_cell_w))
        cell_h = max(82, int(line_count * (annotation_font_size * 1.08) + 10))
        bottom_margin = max(130, min(180, 82 + max_x_label_len * 3)) + 60
        top_margin = 210

    left_margin = max(130, min(220, 85 + max_y_label_len * 4))
    right_margin = 120
    width = max(960, left_margin + right_margin + num_positions * cell_w)
    extra_height = 46 if layout_scale == "top1" else (38 if layout_scale == "top5" else 30)
    height = max(420, top_margin + bottom_margin + num_layers * cell_h + extra_height)
    top_label_max_len = max(
        (len(label) for label in (data["x_labels_secondary"] or [])),
        default=0,
    )
    top_tick_font_size = 34 if top_label_max_len <= 16 else 32 if top_label_max_len <= 24 else 30
    top_axis_title_font_size = (
        32 if top_label_max_len <= 16 else 30 if top_label_max_len <= 24 else 28
    )

    metric_labels = {
        "iou": "Top-k IoU",
        "jsd": "JSD",
        "kl_modelnorm_to_tuned": "KL(ModelNorm || Tuned)",
        "kl_tuned_to_modelnorm": "KL(Tuned || ModelNorm)",
    }
    plot_title = (
        title
        or f"ModelNorm Lens <> Tuned Lens | {data['display_prompt']} | {metric_labels[metric_key]}"
    )
    if width >= 12000:
        title_font_size = 42 if len(plot_title) <= 160 else 40 if len(plot_title) <= 220 else 38
    elif width >= 9000:
        title_font_size = 40 if len(plot_title) <= 160 else 38 if len(plot_title) <= 220 else 36
    else:
        title_font_size = 38 if len(plot_title) <= 140 else 36 if len(plot_title) <= 200 else 34
    adaptive_top_margin = 58 + top_tick_font_size + top_axis_title_font_size + title_font_size
    top_margin = max(top_margin, adaptive_top_margin)

    if metric_key == "iou":
        zmin = 0.0
        zmax = 1.0
        colorbar_title = "IoU"
        colorscale = "RdBu"
    else:
        zmin = 0.0
        zmax = max(float(np.nanmax(data["z"])), 1e-6)
        colorbar_title = metric_labels[metric_key]
        colorscale = "Viridis"

    heatmap_text = np.empty_like(data["cell_parts"], dtype=object)
    for row_idx in range(num_layers):
        for col_idx in range(num_positions):
            parts = data["cell_parts"][row_idx, col_idx]
            if parts is None:
                heatmap_text[row_idx, col_idx] = ""
                continue
            color = _GEN_PLOTTER._text_color_for_value(
                float(data["z"][row_idx, col_idx]),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
            )
            heatmap_text[row_idx, col_idx] = (
                f"<span style='color:{color}'>"
                f"{_GEN_PLOTTER._cell_annotation_html(parts, visible_rows=display_top_tokens)}"
                f"</span>"
            )

    cell_shapes: list[dict[str, Any]] = []
    for row_idx in range(num_layers):
        for col_idx in range(num_positions):
            marker = data["cell_markers"][row_idx, col_idx]
            if marker == "neither" or marker is None:
                continue
            if marker == "both":
                line_color = "#111111"
                line_dash = "solid"
                line_width = 3
            elif marker == "modelnorm_only":
                line_color = "#1f77b4"
                line_dash = "dash"
                line_width = 3
            else:
                line_color = "#d62728"
                line_dash = "dash"
                line_width = 3
            cell_shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": col_idx - 0.48,
                    "x1": col_idx + 0.48,
                    "y0": row_idx - 0.48,
                    "y1": row_idx + 0.48,
                    "line": {
                        "color": line_color,
                        "width": line_width,
                        "dash": line_dash,
                    },
                    "fillcolor": "rgba(0,0,0,0)",
                    "layer": "above",
                }
            )

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Heatmap(
            z=data["z"],
            x=list(range(num_positions)),
            y=list(range(num_layers)),
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            xgap=1,
            ygap=1,
            text=heatmap_text,
            texttemplate="%{text}",
            textfont={
                "family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                "size": annotation_font_size,
            },
            hovertext=data["hover_text"],
            hoverinfo="text",
            colorbar={
                "title": {
                    "text": colorbar_title,
                    "font": {
                        "size": 32,
                        "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                    },
                },
                "tickformat": ".2f",
                "tickfont": {
                    "size": 26,
                    "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                },
                "orientation": "v",
                "thickness": 18,
                "len": 0.82,
                "x": 1.02,
                "xanchor": "left",
                "y": 0.5,
                "yanchor": "middle",
            },
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(num_positions)),
            y=[None] * num_positions,
            mode="markers",
            marker_opacity=0,
            showlegend=False,
            hoverinfo="skip",
            xaxis="x2",
            yaxis="y",
        )
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(num_positions)),
        ticktext=data["x_labels"],
        tickangle=32,
        side="bottom",
        title={
            "text": "Input tokens",
            "font": {
                "size": 34,
                "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
            },
            "standoff": 10,
        },
        showticklabels=True,
        tickfont={
            "size": 32,
            "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
        },
        range=[-0.5, num_positions - 0.5],
        showgrid=False,
        zeroline=False,
        automargin=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=data["y_labels"],
        range=[-0.5, num_layers - 0.5],
        title={
            "text": "Layer",
            "font": {
                "size": 30,
                "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
            },
            "standoff": 10,
        },
        tickfont={
            "size": 34,
            "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
        },
        showgrid=False,
        zeroline=False,
        automargin=True,
        row=1,
        col=1,
    )
    fig.update_layout(
        title={
            "text": f"<span style='font-weight:600'>{plot_title}</span>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.99,
            "yanchor": "top",
            "font": {
                "size": title_font_size,
                "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                "color": "black",
            },
        },
        width=width,
        height=height,
        autosize=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": left_margin, "r": 70, "t": top_margin, "b": bottom_margin},
        font={
            "family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
            "size": 22,
            "color": "black",
        },
        hoverlabel={
            "font": {"color": "black", "size": 12},
            "bgcolor": "white",
            "bordercolor": "black",
            "align": "left",
        },
        hovermode="closest",
        hoverdistance=5,
        annotations=[],
        shapes=cell_shapes,
        xaxis2={
            "anchor": "y",
            "overlaying": "x",
            "side": "top",
            "tickmode": "array",
            "tickvals": list(range(num_positions)),
            "ticktext": data["x_labels_secondary"],
            "tickangle": 32,
            "showticklabels": True,
            "title": {
                "text": "Next tokens",
                "font": {
                    "size": top_axis_title_font_size,
                    "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                },
                "standoff": 0,
            },
            "tickfont": {
                "size": top_tick_font_size,
                "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
            },
            "range": [-0.5, num_positions - 0.5],
            "automargin": True,
            "showgrid": False,
            "zeroline": False,
        },
    )
    return fig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Plotly PDF heatmap comparing ModelNorm Lens and Tuned Lens on one prompt."
    )
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--tuned-lens-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--metric",
        choices=("iou", "jsd", "kl_modelnorm_to_tuned", "kl_tuned_to_modelnorm"),
        default="jsd",
    )
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument(
        "--layer-mode",
        choices=("all", "most_divergent"),
        default="all",
    )
    parser.add_argument("--keep-last-layer-fraction", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--add-special-tokens", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if args.dtype.lower() not in dtype_map:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype_map[args.dtype.lower()],
    ).to(device)
    model.eval()

    tuned_lens = TunedLens.from_model_and_pretrained(
        model,
        lens_resource_id=args.tuned_lens_dir,
        map_location=device,
    ).to(device)
    tuned_lens.eval()

    payload = build_single_prompt_payload(
        model=model,
        tokenizer=tokenizer,
        tuned_lens=tuned_lens,
        prompt=args.prompt,
        top_k=args.top_k,
        add_special_tokens=args.add_special_tokens,
    )

    max_layers = (
        model.config.num_hidden_layers if args.max_layers is None else args.max_layers
    )
    if max_layers <= 0:
        raise ValueError("--max-layers must be positive when provided.")
    if not (0.0 < args.keep_last_layer_fraction <= 1.0):
        raise ValueError("--keep-last-layer-fraction must be in (0, 1].")

    fig = plot_prompt_style_verification_heatmap(
        payload,
        metric=args.metric,
        top_k=min(args.top_k, 10),
        max_divergent_layers=max_layers,
        keep_last_layer_fraction=args.keep_last_layer_fraction,
        max_token_chars=12,
        layer_mode=args.layer_mode,
        title=None,
    )
    output_path = Path(args.output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")
    pio.write_image(
        fig,
        str(output_path),
        format="pdf",
        engine="kaleido",
        width=max(1800, int(fig.layout.width)),
        height=max(1500, int(fig.layout.height)),
    )
    print(f"saved plotly pdf to {output_path}")


__all__ = [
    "build_arg_parser",
    "build_single_prompt_payload",
    "main",
    "plot_prompt_style_verification_heatmap",
]

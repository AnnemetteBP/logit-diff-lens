from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots


def _clean_token(token: str | None) -> str:
    if token is None:
        return ""

    token = str(token)
    replacements = {
        "<|begin_text|>": "BOS",
        "<|begin_of_text|>": "BOS",
        "<begin_text>": "BOS",
        "<begin_of_text>": "BOS",
        "<s>": "BOS",
        "<|end_text|>": "EOS",
        "<|end_of_text|>": "EOS",
        "<end_text>": "EOS",
        "<end_of_text>": "EOS",
        "</s>": "EOS",
        "<pad>": "PAD",
        "<|pad|>": "PAD",
        "<unk>": "UNK",
        "<|unk|>": "UNK",
    }
    token = replacements.get(token, token)
    token = token.replace("Ġ", " ").replace("▁", " ")
    token = token.replace("\n", "\\n")
    return token.strip() or " "


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1] + "…"


def _strip_chat_role_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:User|Assistant|System)\s*:\s*", "", text, flags=re.IGNORECASE)


def _display_prompt_text(prompt: str) -> str:
    lines = [line.strip() for line in str(prompt).splitlines() if line.strip()]
    cleaned: List[str] = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("assistant:"):
            break
        cleaned.append(_strip_chat_role_prefix(line))
    if cleaned:
        return " ".join(cleaned).strip()
    return _strip_chat_role_prefix(str(prompt)).strip()


def _compact_model_label(label: str | None, fallback: str) -> str:
    if not label:
        return fallback
    text = str(label).strip()
    if not text:
        return fallback
    return Path(text).name or text


def _finetuned_display_label(model_meta: Dict[str, Any], fallback: str) -> str:
    adapter_path = (
        model_meta.get("config", {})
        .get("metadata", {})
        .get("adapter_path")
    )
    if adapter_path:
        return "FT Risky Financial Advice"
    return fallback


def _sorted_layer_keys(results: Dict[str, Any]) -> List[str]:
    return sorted(results.keys(), key=float)


def _load_payload(payload_or_path: Dict[str, Any] | str | Path) -> Dict[str, Any]:
    if isinstance(payload_or_path, dict):
        return payload_or_path
    path = Path(payload_or_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_results(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "results" in payload and isinstance(payload["results"], dict):
        return payload["results"]
    return payload


def list_available_prompts(payload_or_path: Dict[str, Any] | str | Path) -> List[str]:
    payload = _load_payload(payload_or_path)
    results = _extract_results(payload)
    layer_keys = _sorted_layer_keys(results)
    if not layer_keys:
        return []
    return [entry["prompt"] for entry in results[layer_keys[0]]]


def _select_prompt(
    results: Dict[str, Any],
    prompt_index: int | None,
    prompt_text: str | None,
) -> List[Dict[str, Any]]:
    layer_keys = _sorted_layer_keys(results)
    if not layer_keys:
        raise ValueError("No layers found in LogitDiff results.")

    prompts = [entry["prompt"] for entry in results[layer_keys[0]]]
    if not prompts:
        raise ValueError("No prompt entries found in LogitDiff results.")

    if prompt_text is not None:
        if prompt_text not in prompts:
            raise ValueError(f"Prompt not found. Available prompts are: {prompts}")
        prompt_index = prompts.index(prompt_text)
    elif prompt_index is None:
        prompt_index = 0

    if prompt_index < 0 or prompt_index >= len(prompts):
        raise IndexError(
            f"prompt_index={prompt_index} is out of range for {len(prompts)} prompts."
        )

    return [results[layer_key][prompt_index] for layer_key in layer_keys]


def _filter_positions(
    positions: Sequence[Dict[str, Any]],
    include_prompt_tokens: bool,
    include_generated_tokens: bool,
) -> List[Dict[str, Any]]:
    filtered = []
    for position in positions:
        is_generated = bool(position.get("is_generated", False))
        if is_generated and not include_generated_tokens:
            continue
        if not is_generated and not include_prompt_tokens:
            continue
        filtered.append(position)
    return filtered


def _slice_positions(
    positions: Sequence[Dict[str, Any]],
    start_idx: int | None,
    end_idx: int | None,
) -> List[Dict[str, Any]]:
    start_idx = 0 if start_idx is None else start_idx
    end_idx = len(positions) if end_idx is None else end_idx
    return list(positions[start_idx:end_idx])


def _build_cell_parts(
    position: Dict[str, Any],
    display_top_tokens: int,
    max_token_chars: int,
    comparison_k: int | None = None,
) -> Dict[str, List[str]]:
    topk_payload = None
    if comparison_k is not None:
        topk_payload = position.get("topk_predictions", {}).get(str(comparison_k))

    if topk_payload is not None:
        shared_source = topk_payload.get("shared_tokens", [])
        base_only_source = topk_payload.get("base_only_tokens", [])
        ft_only_source = topk_payload.get("finetuned_only_tokens", [])
    else:
        shared_source = position.get("intersection", [])
        base_only_source = position.get("only_base", [])
        ft_only_source = position.get("only_finetuned", [])

    shared = [
        _truncate(_clean_token(token), max_token_chars)
        for token in shared_source[:display_top_tokens]
    ]
    only_base = [
        _truncate(_clean_token(token), max_token_chars)
        for token in base_only_source[:display_top_tokens]
    ]
    only_ft = [
        _truncate(_clean_token(token), max_token_chars)
        for token in ft_only_source[:display_top_tokens]
    ]
    return {
        "shared": shared,
        "base_only": only_base,
        "finetuned_only": only_ft,
    }


def _layer_tick_label(label: str) -> str:
    return label


def _build_hover_text(
    layer_result: Dict[str, Any],
    position: Dict[str, Any],
    *,
    comparison_k: int | None = None,
) -> str:
    token_kind = "generated" if position.get("is_generated", False) else "prompt"
    input_token = _clean_token(position.get("input_token"))
    topk_payload = None
    if comparison_k is not None:
        topk_payload = position.get("topk_predictions", {}).get(str(comparison_k))
    if topk_payload is not None:
        shared_source = topk_payload.get("shared_tokens", [])
        base_only_source = topk_payload.get("base_only_tokens", [])
        ft_only_source = topk_payload.get("finetuned_only_tokens", [])
        iou_value = float(topk_payload.get("jaccard", position.get("iou", 0.0)))
    else:
        shared_source = position.get("intersection", [])
        base_only_source = position.get("only_base", [])
        ft_only_source = position.get("only_finetuned", [])
        iou_value = float(position.get("iou", 0.0))
    shared = ", ".join(_clean_token(token) for token in shared_source)
    only_base = ", ".join(_clean_token(token) for token in base_only_source)
    only_ft = ", ".join(_clean_token(token) for token in ft_only_source)
    base_generated = _clean_token(position.get("base_generated_token", ""))
    ft_generated = _clean_token(position.get("ft_generated_token", ""))
    base_top1 = _clean_token(position.get("base_top1_token", ""))
    ft_top1 = _clean_token(position.get("ft_top1_token", ""))
    metric_label = f"Top-{comparison_k} IoU" if comparison_k is not None else "IoU"
    return (
        f"<b>Layer</b>: {layer_result['layer_relative']} (abs {layer_result['layer_absolute']})<br>"
        f"<b>Position</b>: {position['position']}<br>"
        f"<b>Input token</b>: {input_token}<br>"
        f"<b>FT generated token</b>: {ft_generated or '—'}<br>"
        f"<b>Base generated token</b>: {base_generated or '—'}<br>"
        f"<b>FT top-1 prediction</b>: {ft_top1 or '—'}<br>"
        f"<b>Base top-1 prediction</b>: {base_top1 or '—'}<br>"
        f"<b>Type</b>: {token_kind}<br>"
        f"<b>{metric_label}</b>: {iou_value:.4f}<br>"
        f"<b>Shared</b>: {shared or '—'}<br>"
        f"<b>Base only</b>: {only_base or '—'}<br>"
        f"<b>Finetuned only</b>: {only_ft or '—'}"
    )


def _tokens_for_tick_mode(
    positions: Sequence[Dict[str, Any]],
    tick_mode: str,
) -> List[str]:
    if tick_mode == "input_tokens":
        key = "input_token"
    elif tick_mode == "base_generated":
        key = "base_generated_token"
    elif tick_mode == "ft_generated":
        key = "ft_generated_token"
    elif tick_mode == "base_top1":
        key = "base_top1_token"
    elif tick_mode == "ft_top1":
        key = "ft_top1_token"
    elif tick_mode == "position":
        return [str(position["position"]) for position in positions]
    else:
        raise ValueError(
            f"Unknown tick_mode '{tick_mode}'. Expected 'input_tokens', 'base_generated', 'ft_generated', 'base_top1', 'ft_top1', or 'position'."
        )

    tokens = [_clean_token(position.get(key)) for position in positions]
    if len(tokens) >= 2 and tokens[0].lower() in {"user", "assistant", "system"} and tokens[1] == ":":
        tokens = tokens[2:]
    return tokens


def _prepare_heatmap_data(
    payload_or_path: Dict[str, Any] | str | Path,
    prompt_index: int | None = None,
    prompt_text: str | None = None,
    include_prompt_tokens: bool = False,
    include_generated_tokens: bool = True,
    start_idx: int | None = None,
    end_idx: int | None = None,
    display_top_tokens: int = 2,
    max_token_chars: int = 12,
    comparison_k: int | None = None,
    max_layers: int | None = None,
    layer_selection: str = "most_divergent",
    x_tick_mode: str = "ft_generated",
    x_tick_mode_secondary: str | None = "base_generated",
) -> Dict[str, Any]:
    payload = _load_payload(payload_or_path)
    results = _extract_results(payload)
    per_layer_prompt_results = _select_prompt(results, prompt_index, prompt_text)
    all_positions = per_layer_prompt_results[0]["positions"]

    reference_positions = _slice_positions(
        _filter_positions(
            all_positions,
            include_prompt_tokens=include_prompt_tokens,
            include_generated_tokens=include_generated_tokens,
        ),
        start_idx,
        end_idx,
    )
    if not reference_positions:
        raise ValueError(
            "No positions left after filtering. Adjust include_prompt_tokens, "
            "include_generated_tokens, start_idx, or end_idx."
        )

    selected_positions = [position["position"] for position in reference_positions]
    position_to_column = {position: idx for idx, position in enumerate(selected_positions)}

    num_layers = len(per_layer_prompt_results)
    num_positions = len(selected_positions)
    if comparison_k is None:
        comparison_k = display_top_tokens if display_top_tokens in (1, 5, 10) else None
    z = np.full((num_layers, num_positions), np.nan, dtype=float)
    hover_text = np.empty((num_layers, num_positions), dtype=object)
    cell_parts = np.empty((num_layers, num_positions), dtype=object)
    y_labels = []
    total_model_layers = max(
        int(layer_result["layer_absolute"]) for layer_result in per_layer_prompt_results
    ) + 1

    for layer_idx, layer_result in enumerate(per_layer_prompt_results):
        y_labels.append(
            f"Layer {int(layer_result['layer_absolute']) + 1}/{total_model_layers}"
        )
        for position in layer_result["positions"]:
            column_idx = position_to_column.get(position["position"])
            if column_idx is None:
                continue
            topk_payload = (
                position.get("topk_predictions", {}).get(str(comparison_k))
                if comparison_k is not None
                else None
            )
            z[layer_idx, column_idx] = float(
                topk_payload.get("jaccard", position["iou"]) if topk_payload is not None else position["iou"]
            )
            hover_text[layer_idx, column_idx] = _build_hover_text(
                layer_result,
                position,
                comparison_k=comparison_k,
            )
            cell_parts[layer_idx, column_idx] = _build_cell_parts(
                position,
                display_top_tokens=display_top_tokens,
                max_token_chars=max_token_chars,
                comparison_k=comparison_k,
            )

    mean_per_layer = np.nanmean(z, axis=1)
    selected_layer_indices = list(range(num_layers))
    if max_layers is not None and max_layers < num_layers:
        if layer_selection == "most_divergent":
            selected_layer_indices = np.argsort(mean_per_layer)[:max_layers].tolist()
        elif layer_selection == "least_divergent":
            selected_layer_indices = np.argsort(mean_per_layer)[-max_layers:].tolist()
            selected_layer_indices = list(reversed(selected_layer_indices))
        else:
            selected_layer_indices = list(range(max_layers))

        selected_layer_indices = sorted(
            selected_layer_indices,
            key=lambda idx: float(per_layer_prompt_results[idx]["layer_relative"]),
        )

        z = z[selected_layer_indices, :]
        hover_text = hover_text[selected_layer_indices, :]
        cell_parts = cell_parts[selected_layer_indices, :]
        y_labels = [y_labels[idx] for idx in selected_layer_indices]
        mean_per_layer = mean_per_layer[selected_layer_indices]

    mean_per_position = np.nanmean(z, axis=0)
    token_kinds = [
        "gen" if position.get("is_generated", False) else "prompt"
        for position in reference_positions
    ]

    x_labels = _tokens_for_tick_mode(reference_positions, x_tick_mode)
    x_labels_secondary = (
        _tokens_for_tick_mode(reference_positions, x_tick_mode_secondary)
        if x_tick_mode_secondary is not None
        else None
    )

    return {
        "prompt": per_layer_prompt_results[0]["prompt"],
        "display_prompt": _display_prompt_text(per_layer_prompt_results[0]["prompt"]),
        "x_labels": x_labels,
        "x_labels_secondary": x_labels_secondary,
        "x_positions": [position["position"] for position in reference_positions],
        "y_labels": y_labels,
        "token_kinds": token_kinds,
        "z": z,
        "hover_text": hover_text,
        "cell_parts": cell_parts,
        "mean_per_position": mean_per_position,
        "mean_per_layer": mean_per_layer,
        "x_tick_mode": x_tick_mode,
        "x_tick_mode_secondary": x_tick_mode_secondary,
        "comparison_k": comparison_k,
        "payload": payload,
    }


def _prepare_next_token_verification_heatmap_data(
    payload_or_path: Dict[str, Any] | str | Path,
    prompt_index: int | None = None,
    prompt_text: str | None = None,
    include_prompt_tokens: bool = True,
    include_generated_tokens: bool = True,
    start_idx: int | None = None,
    end_idx: int | None = None,
    display_top_tokens: int = 10,
    max_token_chars: int = 12,
    max_layers: int = 5,
    keep_last_layer_fraction: float = 0.5,
) -> Dict[str, Any]:
    payload = _load_payload(payload_or_path)
    results = _extract_results(payload)
    per_layer_prompt_results = _select_prompt(results, prompt_index, prompt_text)
    all_positions = per_layer_prompt_results[0]["positions"]

    reference_positions = _slice_positions(
        _filter_positions(
            all_positions,
            include_prompt_tokens=include_prompt_tokens,
            include_generated_tokens=include_generated_tokens,
        ),
        start_idx,
        end_idx,
    )
    if len(reference_positions) < 2:
        raise ValueError("Need at least two positions after filtering to build token-pair verification view.")

    paired_positions = []
    for idx in range(len(reference_positions) - 1):
        predictor = reference_positions[idx]
        target = reference_positions[idx + 1]
        paired_positions.append(
            {
                "predictor_position": int(predictor["position"]),
                "target_position": int(target["position"]),
                "predictor_token": _clean_token(predictor.get("input_token")),
                "target_token": _clean_token(target.get("input_token")),
                "is_generated": bool(predictor.get("is_generated", False)),
            }
        )

    selected_predictor_positions = [pair["predictor_position"] for pair in paired_positions]
    position_to_column = {
        position: idx for idx, position in enumerate(selected_predictor_positions)
    }

    num_layers_total = len(per_layer_prompt_results)
    candidate_start = int(np.floor(num_layers_total * (1.0 - keep_last_layer_fraction)))
    candidate_indices = list(range(max(0, candidate_start), num_layers_total))
    if not candidate_indices:
        candidate_indices = list(range(num_layers_total))

    mean_iou_candidates: List[tuple[float, int]] = []
    for layer_idx in candidate_indices:
        layer_positions = [
            position
            for position in per_layer_prompt_results[layer_idx]["positions"]
            if int(position["position"]) in position_to_column
        ]
        if not layer_positions:
            continue
        mean_iou_candidates.append(
            (float(np.mean([float(position["iou"]) for position in layer_positions])), layer_idx)
        )

    mean_iou_candidates.sort(key=lambda item: item[0])
    selected_layer_indices = [layer_idx for _, layer_idx in mean_iou_candidates[:max_layers]]
    selected_layer_indices = sorted(
        selected_layer_indices,
        key=lambda idx: float(per_layer_prompt_results[idx]["layer_relative"]),
    )
    if not selected_layer_indices:
        raise ValueError("No layers selected for next-token verification heatmap.")

    num_layers = len(selected_layer_indices)
    num_positions = len(paired_positions)
    z = np.full((num_layers, num_positions), np.nan, dtype=float)
    hover_text = np.empty((num_layers, num_positions), dtype=object)
    cell_parts = np.empty((num_layers, num_positions), dtype=object)
    y_labels = []
    total_model_layers = max(
        int(layer_result["layer_absolute"]) for layer_result in per_layer_prompt_results
    ) + 1

    for row_idx, original_layer_idx in enumerate(selected_layer_indices):
        layer_result = per_layer_prompt_results[original_layer_idx]
        y_labels.append(f"Layer {int(layer_result['layer_absolute']) + 1}/{total_model_layers}")
        layer_positions = {
            int(position["position"]): position for position in layer_result["positions"]
        }
        for pair in paired_positions:
            predictor_position = layer_positions.get(pair["predictor_position"])
            if predictor_position is None:
                continue
            col_idx = position_to_column[pair["predictor_position"]]
            z[row_idx, col_idx] = float(predictor_position["iou"])
            hover_text[row_idx, col_idx] = (
                f"<b>Layer</b>: {layer_result['layer_relative']} (abs {layer_result['layer_absolute']})<br>"
                f"<b>Predictor token</b>: {pair['predictor_token']}<br>"
                f"<b>Target token</b>: {pair['target_token']}<br>"
                f"<b>Position</b>: {pair['predictor_position']} → {pair['target_position']}<br>"
                f"<b>IoU</b>: {float(predictor_position['iou']):.4f}<br>"
                f"<b>Shared</b>: "
                + (", ".join(_clean_token(token) for token in predictor_position.get("intersection", [])[:display_top_tokens]) or "—")
                + "<br><b>Base only</b>: "
                + (", ".join(_clean_token(token) for token in predictor_position.get("only_base", [])[:display_top_tokens]) or "—")
                + "<br><b>Finetuned only</b>: "
                + (", ".join(_clean_token(token) for token in predictor_position.get("only_finetuned", [])[:display_top_tokens]) or "—")
            )
            cell_parts[row_idx, col_idx] = _build_cell_parts(
                predictor_position,
                display_top_tokens=display_top_tokens,
                max_token_chars=max_token_chars,
            )

    mean_per_layer = np.nanmean(z, axis=1)
    mean_per_position = np.nanmean(z, axis=0)

    return {
        "prompt": per_layer_prompt_results[0]["prompt"],
        "display_prompt": _display_prompt_text(per_layer_prompt_results[0]["prompt"]),
        "x_labels": [pair["predictor_token"] for pair in paired_positions],
        "x_labels_secondary": [pair["target_token"] for pair in paired_positions],
        "x_positions": selected_predictor_positions,
        "y_labels": y_labels,
        "token_kinds": ["gen" if pair["is_generated"] else "prompt" for pair in paired_positions],
        "z": z,
        "hover_text": hover_text,
        "cell_parts": cell_parts,
        "mean_per_position": mean_per_position,
        "mean_per_layer": mean_per_layer,
        "x_tick_mode": "predictor_tokens",
        "x_tick_mode_secondary": "target_tokens",
        "payload": payload,
    }


def _cell_annotation_html(parts: Dict[str, List[str]], visible_rows: int) -> str:
    def _quote(token: str) -> str:
        return f"'{token}'"

    def _wrap_shared_tokens(tokens: List[str], *, max_per_line: int = 2, max_chars: int = 24) -> str:
        if not tokens:
            return "—"
        lines: List[str] = []
        current: List[str] = []
        current_len = 0
        for token in tokens:
            token_len = len(token)
            if current and (len(current) >= max_per_line or current_len + 2 + token_len > max_chars):
                lines.append(", ".join(current))
                current = [token]
                current_len = token_len
            else:
                if current:
                    current_len += 2 + token_len
                else:
                    current_len = token_len
                current.append(token)
        if current:
            lines.append(", ".join(current))
        return "<br>".join(lines)

    shared_tokens = parts["shared"][:visible_rows]
    if not shared_tokens:
        shared = "—"
        shared_line_count = 1
    else:
        quoted = [_quote(token) for token in shared_tokens]
        shared = _wrap_shared_tokens(quoted)
        shared_line_count = len(shared.split("<br>"))
    max_total_lines = max(2, visible_rows)
    available_bottom_rows = max(1, max_total_lines - shared_line_count)
    bottom_pairs = []
    partial_used = False
    for idx in range(visible_rows):
        left = parts["base_only"][idx] if idx < len(parts["base_only"]) else "—"
        right = parts["finetuned_only"][idx] if idx < len(parts["finetuned_only"]) else "—"
        if left == "—" and right == "—":
            continue
        if len(bottom_pairs) >= available_bottom_rows:
            break
        is_partial = (left == "—") != (right == "—")
        if is_partial and partial_used:
            continue
        left_text = _quote(left) if left != "—" else left
        right_text = _quote(right) if right != "—" else right
        bottom_pairs.append(f"{left_text} <> {right_text}")
        if is_partial:
            partial_used = True
    bottom = "<br>".join(bottom_pairs) if bottom_pairs else "—"
    return (
        "<span style='font-weight:700; font-size:1.08em'>"
        f"{shared}"
        "</span><br>"
        f"{bottom}"
    )


def _cell_annotation_plaintext(parts: Dict[str, List[str]], visible_rows: int) -> str:
    def _quote(token: str) -> str:
        return f"'{token}'"

    shared_tokens = parts["shared"][:visible_rows]
    shared = ", ".join(_quote(token) for token in shared_tokens) if shared_tokens else "—"
    bottom_lines = []
    for idx in range(visible_rows):
        left = parts["base_only"][idx] if idx < len(parts["base_only"]) else "—"
        right = parts["finetuned_only"][idx] if idx < len(parts["finetuned_only"]) else "—"
        left_text = _quote(left) if left != "—" else left
        right_text = _quote(right) if right != "—" else right
        bottom_lines.append(f"{left_text} <> {right_text}")
    bottom = "\n".join(bottom_lines) if bottom_lines else "—"
    return f"{shared}\n{bottom}"


def _rgb_components(color: str) -> tuple[float, float, float]:
    color = color.strip()
    if color.startswith("rgb("):
        values = color[4:-1].split(",")
    elif color.startswith("rgba("):
        values = color[5:-1].split(",")[:3]
    else:
        return (0.0, 0.0, 0.0)
    r, g, b = [float(v.strip()) for v in values[:3]]
    return r / 255.0, g / 255.0, b / 255.0


def _text_color_for_value(value: float, colorscale: str, zmin: float, zmax: float) -> str:
    if zmax <= zmin:
        norm = 0.5
    else:
        norm = max(0.0, min(1.0, (value - zmin) / (zmax - zmin)))
    sampled = sample_colorscale(colorscale, [norm])[0]
    r, g, b = _rgb_components(sampled)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.45 else "black"


def plot_logitdiff_jaccard_heatmap(
    payload_or_path: Dict[str, Any] | str | Path,
    prompt_index: int | None = None,
    prompt_text: str | None = None,
    include_prompt_tokens: bool = False,
    include_generated_tokens: bool = True,
    start_idx: int | None = None,
    end_idx: int | None = None,
    title: str | None = None,
    colorscale: str = "RdBu",
    display_top_tokens: int = 10,
    visible_cell_tokens: int | None = None,
    max_token_chars: int = 12,
    show_marginals: bool = False,
    max_layers: int | None = 5,
    visible_layers: int | None = None,
    layer_selection: str = "most_divergent",
    x_tick_mode: str = "ft_generated",
    x_tick_mode_secondary: str | None = "base_generated",
    model_a_label: str | None = None,
    model_b_label: str | None = None,
) -> go.Figure:
    display_top_tokens = visible_cell_tokens if visible_cell_tokens is not None else display_top_tokens
    max_layers = visible_layers if visible_layers is not None else max_layers
    if display_top_tokens <= 1:
        layout_scale = "top1"
    elif display_top_tokens <= 5:
        layout_scale = "top5"
    else:
        layout_scale = "top10"
    compact_top1_layout = layout_scale == "top1"
    medium_topk_layout = layout_scale == "top5"
    if layout_scale == "top1":
        effective_max_token_chars = min(max_token_chars, 8)
    elif layout_scale == "top5":
        effective_max_token_chars = min(max_token_chars, 10)
    else:
        effective_max_token_chars = max_token_chars
    data = _prepare_heatmap_data(
        payload_or_path=payload_or_path,
        prompt_index=prompt_index,
        prompt_text=prompt_text,
        include_prompt_tokens=include_prompt_tokens,
        include_generated_tokens=include_generated_tokens,
        start_idx=start_idx,
        end_idx=end_idx,
        display_top_tokens=display_top_tokens,
        max_token_chars=effective_max_token_chars,
        comparison_k=display_top_tokens if display_top_tokens in (1, 5, 10) else None,
        max_layers=max_layers,
        layer_selection=layer_selection,
        x_tick_mode=x_tick_mode,
        x_tick_mode_secondary=x_tick_mode_secondary,
    )

    num_layers, num_positions = data["z"].shape
    max_x_label_len = max((len(label) for label in data["x_labels"]), default=1)
    max_y_label_len = max((len(label) for label in data["y_labels"]), default=1)
    shared_line_count = max(1, (display_top_tokens + 1) // 2)
    nonshared_line_count = max(1, display_top_tokens)
    line_count = shared_line_count + nonshared_line_count
    if layout_scale == "top1":
        annotation_font_size = max(22, min(26, int(192 / max(1, line_count))))
    elif layout_scale == "top5":
        annotation_font_size = max(18, min(22, int(214 / max(1, line_count))))
    else:
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
                longest_visible_token = max(longest_visible_token, max(len(token) for token in visible_tokens))
            for idx in range(display_top_tokens):
                left = parts["base_only"][idx] if idx < len(parts["base_only"]) else "—"
                right = parts["finetuned_only"][idx] if idx < len(parts["finetuned_only"]) else "—"
                longest_visible_line = max(longest_visible_line, len(f"'{left}' <> '{right}'"))
    base_cell_w = 48 + max(max_x_label_len * 5, longest_visible_token * 10, longest_visible_line * 7)
    if layout_scale == "top1":
        cell_w = max(184, min(360, base_cell_w + 16))
    elif layout_scale == "top5":
        cell_w = max(170, min(348, base_cell_w + 8))
    else:
        cell_w = max(160, min(340, base_cell_w))
    if layout_scale == "top1":
        cell_h = max(96, int(line_count * (annotation_font_size * 1.18) + 12))
    elif layout_scale == "top5":
        cell_h = max(88, int(line_count * (annotation_font_size * 1.11) + 11))
    else:
        cell_h = max(82, int(line_count * (annotation_font_size * 1.08) + 10))
    left_margin = max(130, min(220, 85 + max_y_label_len * 4))
    right_margin = 110 if show_marginals else 120
    base_bottom_margin = max(130, min(180, 82 + max_x_label_len * 3))
    if layout_scale == "top1":
        bottom_margin = base_bottom_margin + 72
        top_margin = 210 if data["x_labels_secondary"] is not None else 156
    elif layout_scale == "top5":
        bottom_margin = base_bottom_margin + 60
        top_margin = 216 if data["x_labels_secondary"] is not None else 162
    else:
        bottom_margin = base_bottom_margin + 60
        top_margin = 220 if data["x_labels_secondary"] is not None else 166
    width = max(960, left_margin + right_margin + num_positions * cell_w + (170 if show_marginals else 0))
    extra_height = 36 if layout_scale == "top1" else (28 if layout_scale == "top5" else 20)
    height = max(420, top_margin + bottom_margin + num_layers * cell_h + (90 if show_marginals else 0) + extra_height)

    zmin = 0.0
    zmax = 1.0
    heatmap_text = np.empty_like(data["cell_parts"], dtype=object)
    for row_idx in range(num_layers):
        for col_idx in range(num_positions):
            parts = data["cell_parts"][row_idx, col_idx]
            if parts is None:
                heatmap_text[row_idx, col_idx] = ""
                continue
            color = _text_color_for_value(
                float(data["z"][row_idx, col_idx]),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
            )
            heatmap_text[row_idx, col_idx] = (
                f"<span style='color:{color}'>"
                f"{_cell_annotation_html(parts, visible_rows=display_top_tokens)}"
                f"</span>"
            )

    if show_marginals:
        fig = make_subplots(
            rows=2,
            cols=2,
            row_heights=[0.08, 0.92],
            column_widths=[0.86, 0.14],
            specs=[[{"type": "xy"}, None], [{"type": "heatmap"}, {"type": "xy"}]],
            horizontal_spacing=0.015,
            vertical_spacing=0.02,
        )
        main_row, main_col = 2, 1
    else:
        fig = make_subplots(rows=1, cols=1)
        main_row, main_col = 1, 1

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
            showscale=True,
            colorbar={
                "title": {
                    "text": "IoU",
                    "font": {
                        "size": 32,
                        "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                    },
                },
                "orientation": "v",
                "thickness": 18,
                "len": 0.82,
                "x": 1.02,
                "xanchor": "left",
                "y": 0.5,
                "yanchor": "middle",
                "tickfont": {
                    "size": 26,
                    "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                },
            },
        ),
        row=main_row,
        col=main_col,
    )

    if show_marginals:
        fig.add_trace(
            go.Bar(
                x=list(range(num_positions)),
                y=data["mean_per_position"],
                marker_color="#c7c7c7",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=data["mean_per_layer"],
                y=list(range(num_layers)),
                orientation="h",
                marker_color="#c7c7c7",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(num_positions)),
        ticktext=data["x_labels"],
        tickangle=32,
        side="bottom",
        automargin=True,
        tickfont={
            "size": 32,
            "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
        },
        range=[-0.5, num_positions - 0.5],
        showgrid=False,
        zeroline=False,
        row=main_row,
        col=main_col,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=data["y_labels"],
        tickfont={
            "size": 34,
            "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
        },
        automargin=True,
        range=[-0.5, num_layers - 0.5],
        showgrid=False,
        zeroline=False,
        row=main_row,
        col=main_col,
    )

    if data["x_labels_secondary"] is not None:
        main_yaxis_ref = fig.data[0].yaxis if getattr(fig.data[0], "yaxis", None) else "y"
        fig.add_trace(
            go.Scatter(
                x=list(range(num_positions)),
                y=[None] * num_positions,
                mode="markers",
                marker_opacity=0,
                showlegend=False,
                hoverinfo="skip",
                xaxis="x2",
                yaxis=main_yaxis_ref,
            )
        )
        fig.update_layout(
            xaxis2={
                "anchor": "y",
                "overlaying": "x",
                "side": "top",
                "tickmode": "array",
                "tickvals": list(range(num_positions)),
                "ticktext": data["x_labels_secondary"],
                "tickangle": 32,
                "tickfont": {
                    "size": 32,
                    "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                },
                "range": [-0.5, num_positions - 0.5],
                "automargin": True,
                "showgrid": False,
                "zeroline": False,
            }
        )

    if show_marginals:
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=2)
        fig.update_yaxes(
            range=[-0.5, num_layers - 0.5],
            row=2,
            col=2,
        )

    model_meta = data["payload"].get("metadata", {})
    model_a_label = _compact_model_label(model_a_label or model_meta.get("base_model_name"), "Base")
    model_b_fallback = _compact_model_label(model_b_label or model_meta.get("finetuned_model_name"), "Finetuned")
    model_b_label = _finetuned_display_label(model_meta, model_b_fallback)
    prompt_title = _display_prompt_text(data["prompt"])
    display_title = title or f"{model_a_label} <> {model_b_label}"
    subtitle_k = data.get("comparison_k") or model_meta.get("top_k", "?")
    display_subtitle = f"{prompt_title} | Top-{subtitle_k} Jaccard (IoU)"

    fig.update_layout(
        title={
            "text": f"<b>{display_title} | {display_subtitle}</b>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.992,
            "yanchor": "top",
            "font": {
                "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                "size": 36,
                "color": "black",
            },
        },
        width=width,
        height=height,
        autosize=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif", "size": 22, "color": "black"},
        margin={"l": 120, "r": 70, "t": top_margin, "b": bottom_margin},
        hoverlabel={
            "font": {"color": "black", "size": 12},
            "bgcolor": "white",
            "bordercolor": "black",
            "align": "left",
        },
        hovermode="closest",
        hoverdistance=5,
        annotations=[],
    )

    main_x_title = {
        "input_tokens": "Input prompt tokens",
        "base_generated": "Base model generated tokens",
        "ft_generated": "FT generated tokens",
        "base_top1": "Base model top-1 tokens",
        "ft_top1": "FT top-1 tokens",
        "position": "Token position",
    }[x_tick_mode]
    fig.update_xaxes(
        title={
            "text": main_x_title,
            "font": {
                "size": 34,
                "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
            },
            "standoff": 10,
        },
        row=main_row,
        col=main_col,
    )
    fig.update_yaxes(title={"text": "", "font": {"size": 30}, "standoff": 22}, row=main_row, col=main_col)

    if data["x_labels_secondary"] is not None:
        secondary_x_title = {
            "input_tokens": "Input prompt tokens",
            "base_generated": "Base model generated tokens",
            "ft_generated": "FT generated tokens",
            "base_top1": "Base model top-1 tokens",
            "ft_top1": "FT top-1 tokens",
            "position": "Token position",
        }[data["x_tick_mode_secondary"]]
        fig.update_layout(
            xaxis2={
                **fig.layout.xaxis2.to_plotly_json(),
                "title": {
                    "text": secondary_x_title,
                    "font": {
                        "size": 34,
                        "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                    },
                    "standoff": 12,
                },
            }
        )

    return fig


def plot_logitdiff_next_token_verification_heatmap(
    payload_or_path: Dict[str, Any] | str | Path,
    prompt_index: int | None = None,
    prompt_text: str | None = None,
    include_prompt_tokens: bool = True,
    include_generated_tokens: bool = True,
    start_idx: int | None = None,
    end_idx: int | None = None,
    title: str | None = None,
    colorscale: str = "RdBu",
    display_top_tokens: int = 10,
    visible_cell_tokens: int | None = None,
    max_token_chars: int = 12,
    show_marginals: bool = False,
    max_layers: int = 5,
    keep_last_layer_fraction: float = 0.5,
    model_a_label: str | None = None,
    model_b_label: str | None = None,
) -> go.Figure:
    display_top_tokens = visible_cell_tokens if visible_cell_tokens is not None else display_top_tokens
    data = _prepare_next_token_verification_heatmap_data(
        payload_or_path=payload_or_path,
        prompt_index=prompt_index,
        prompt_text=prompt_text,
        include_prompt_tokens=include_prompt_tokens,
        include_generated_tokens=include_generated_tokens,
        start_idx=start_idx,
        end_idx=end_idx,
        display_top_tokens=display_top_tokens,
        max_token_chars=max_token_chars,
        max_layers=max_layers,
        keep_last_layer_fraction=keep_last_layer_fraction,
    )

    num_layers, num_positions = data["z"].shape
    max_x_label_len = max((len(label) for label in data["x_labels"]), default=1)
    max_y_label_len = max((len(label) for label in data["y_labels"]), default=1)
    line_count = 1 + display_top_tokens
    annotation_font_size = max(9, min(18, int(104 / max(1, line_count))))
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
                longest_visible_token = max(longest_visible_token, max(len(token) for token in visible_tokens))
            for idx in range(display_top_tokens):
                left = parts["base_only"][idx] if idx < len(parts["base_only"]) else "—"
                right = parts["finetuned_only"][idx] if idx < len(parts["finetuned_only"]) else "—"
                longest_visible_line = max(longest_visible_line, len(f"'{left}' <> '{right}'"))
    cell_w = max(155, min(360, 44 + max(max_x_label_len * 5, longest_visible_token * 10, longest_visible_line * 7)))
    cell_h = max(84, int(line_count * (annotation_font_size * 1.55) + 18))
    left_margin = max(170, min(250, 95 + max_y_label_len * 5))
    right_margin = 110 if show_marginals else 120
    bottom_margin = max(140, min(195, 90 + max_x_label_len * 4))
    top_margin = 220
    width = max(960, left_margin + right_margin + num_positions * cell_w + (170 if show_marginals else 0))
    height = max(420, top_margin + bottom_margin + num_layers * cell_h + (90 if show_marginals else 0))

    zmin = 0.0
    zmax = 1.0
    heatmap_text = np.empty_like(data["cell_parts"], dtype=object)
    for row_idx in range(num_layers):
        for col_idx in range(num_positions):
            parts = data["cell_parts"][row_idx, col_idx]
            if parts is None:
                heatmap_text[row_idx, col_idx] = ""
                continue
            color = _text_color_for_value(
                float(data["z"][row_idx, col_idx]),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
            )
            heatmap_text[row_idx, col_idx] = (
                f"<span style='color:{color}'>"
                f"{_cell_annotation_html(parts, visible_rows=display_top_tokens)}"
                f"</span>"
            )

    if show_marginals:
        fig = make_subplots(
            rows=2,
            cols=2,
            row_heights=[0.08, 0.92],
            column_widths=[0.86, 0.14],
            specs=[[{"type": "xy"}, None], [{"type": "heatmap"}, {"type": "xy"}]],
            horizontal_spacing=0.015,
            vertical_spacing=0.02,
        )
        main_row, main_col = 2, 1
    else:
        fig = make_subplots(rows=1, cols=1)
        main_row, main_col = 1, 1

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
                "color": "black",
            },
            hovertext=data["hover_text"],
            hoverinfo="text",
            showscale=True,
            colorbar={
                "title": {"text": "IoU", "font": {"size": 24}},
                "orientation": "v",
                "thickness": 18,
                "len": 0.82,
                "x": 1.02,
                "xanchor": "left",
                "y": 0.5,
                "yanchor": "middle",
                "tickfont": {"size": 22},
            },
        ),
        row=main_row,
        col=main_col,
    )

    if show_marginals:
        fig.add_trace(go.Bar(x=list(range(num_positions)), y=data["mean_per_position"], marker_color="#c7c7c7", hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=data["mean_per_layer"], y=list(range(num_layers)), orientation="h", marker_color="#c7c7c7", hoverinfo="skip", showlegend=False), row=2, col=2)

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(num_positions)),
        ticktext=data["x_labels"],
        tickangle=45 if num_positions > 8 or max_x_label_len > 10 else 0,
        side="bottom",
        automargin=True,
        tickfont={"size": 24},
        range=[-0.5, num_positions - 0.5],
        showgrid=False,
        zeroline=False,
        row=main_row,
        col=main_col,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=data["y_labels"],
        tickfont={"size": 24},
        automargin=True,
        range=[-0.5, num_layers - 0.5],
        showgrid=False,
        zeroline=False,
        row=main_row,
        col=main_col,
    )

    main_yaxis_ref = fig.data[0].yaxis if getattr(fig.data[0], "yaxis", None) else "y"
    fig.add_trace(
        go.Scatter(
            x=list(range(num_positions)),
            y=[None] * num_positions,
            mode="markers",
            marker_opacity=0,
            showlegend=False,
            hoverinfo="skip",
            xaxis="x2",
            yaxis=main_yaxis_ref,
        )
    )
    fig.update_layout(
        xaxis2={
            "anchor": "y",
            "overlaying": "x",
            "side": "top",
            "tickmode": "array",
            "tickvals": list(range(num_positions)),
            "ticktext": data["x_labels_secondary"],
            "tickangle": 45 if num_positions > 8 else 0,
            "tickfont": {"size": 24},
            "range": [-0.5, num_positions - 0.5],
            "automargin": True,
            "showgrid": False,
            "zeroline": False,
        }
    )

    if show_marginals:
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=2)
        fig.update_yaxes(range=[-0.5, num_layers - 0.5], row=2, col=2)

    model_meta = data["payload"].get("metadata", {})
    model_a_label = model_a_label or model_meta.get("base_model_name", "Base")
    model_b_label = model_b_label or model_meta.get("finetuned_model_name", "Finetuned")
    prompt_title = _display_prompt_text(data["prompt"])
    if title is None:
        title = (
            f"{model_a_label} <> {model_b_label}"
            f"<br><sup>{prompt_title} | Top-{model_meta.get('top_k', '?')} next-token verification IoU</sup>"
        )
    else:
        title = f"{model_a_label} <> {model_b_label}<br><sup>{prompt_title} | {title}</sup>"

    fig.update_layout(
        title={"text": title, "font": {"size": 28}, "y": 0.985, "yanchor": "top", "x": 0.5, "xanchor": "center"},
        width=width,
        height=height,
        autosize=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif", "size": 24, "color": "black"},
        margin={"l": left_margin, "r": right_margin, "t": top_margin, "b": bottom_margin},
        hoverlabel={"font": {"color": "black", "size": 12}, "bgcolor": "white", "bordercolor": "black", "align": "left"},
        hovermode="closest",
        hoverdistance=5,
    )

    fig.update_xaxes(title={"text": "Predictor token (input tokens minus last token)", "font": {"size": 25}, "standoff": 14}, row=main_row, col=main_col)
    fig.update_yaxes(title={"text": "Layer", "font": {"size": 25}, "standoff": 18}, row=main_row, col=main_col)
    fig.update_layout(
        xaxis2={
            **fig.layout.xaxis2.to_plotly_json(),
            "title": {"text": "Target token (input tokens minus first token)", "font": {"size": 22}, "standoff": 4},
        }
    )
    return fig


def save_logitdiff_heatmap_html(
    payload_or_path: Dict[str, Any] | str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    fig = plot_logitdiff_jaccard_heatmap(payload_or_path, **kwargs)
    path = Path(output_path).with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(
        fig,
        file=str(path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displayModeBar": False},
        default_width=f"{fig.layout.width}px",
        default_height=f"{fig.layout.height}px",
    )
    return path


def save_logitdiff_heatmap_pdf(
    payload_or_path: Dict[str, Any] | str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    fig = plot_logitdiff_jaccard_heatmap(payload_or_path, **kwargs)
    path = Path(output_path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_width = max(1800, int(fig.layout.width))
        export_height = max(1500, int(fig.layout.height))
        pio.write_image(
            fig,
            str(path),
            format="pdf",
            engine="kaleido",
            width=export_width,
            height=export_height,
        )
    except Exception as exc:
        raise RuntimeError(
            "Direct Plotly-to-PDF export failed. No non-Plotly PDF fallback was used."
        ) from exc
    return path


def save_logitdiff_next_token_verification_html(
    payload_or_path: Dict[str, Any] | str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    fig = plot_logitdiff_next_token_verification_heatmap(payload_or_path, **kwargs)
    path = Path(output_path).with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(
        fig,
        file=str(path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displayModeBar": False},
        default_width=f"{fig.layout.width}px",
        default_height=f"{fig.layout.height}px",
    )
    return path


def save_logitdiff_next_token_verification_pdf(
    payload_or_path: Dict[str, Any] | str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    fig = plot_logitdiff_next_token_verification_heatmap(payload_or_path, **kwargs)
    path = Path(output_path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_width = max(1800, int(fig.layout.width))
        export_height = max(1500, int(fig.layout.height))
        pio.write_image(
            fig,
            str(path),
            format="pdf",
            engine="kaleido",
            width=export_width,
            height=export_height,
        )
    except Exception as exc:
        raise RuntimeError(
            "Direct Plotly-to-PDF export failed. No non-Plotly PDF fallback was used."
        ) from exc
    return path

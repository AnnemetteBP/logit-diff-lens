from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from .logitdiff_gen_plotter import (
    _clean_token,
    _display_prompt_text,
    _extract_results,
    _load_payload,
    _select_prompt,
)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1] + "…"


def _format_token_block(tokens: Sequence[str], max_items: int, max_chars: int) -> str:
    cleaned = [_truncate(_clean_token(token), max_chars) for token in tokens[:max_items]]
    return ", ".join(cleaned) if cleaned else "—"


def _choose_divergent_layers(
    per_layer_prompt_results: Sequence[Dict[str, Any]],
    usable_positions: Sequence[Dict[str, Any]],
    keep_last_fraction: float,
    max_layers: int,
) -> List[int]:
    num_layers = len(per_layer_prompt_results)
    start_idx = int(np.floor(num_layers * (1.0 - keep_last_fraction)))
    candidate_indices = list(range(max(0, start_idx), num_layers))
    if not candidate_indices:
        candidate_indices = list(range(num_layers))

    position_keys = {int(position["position"]) for position in usable_positions}
    scored: List[tuple[float, int]] = []
    for layer_idx in candidate_indices:
        positions = [
            position
            for position in per_layer_prompt_results[layer_idx]["positions"]
            if int(position["position"]) in position_keys
        ]
        if not positions:
            continue
        mean_iou = float(np.mean([float(position["iou"]) for position in positions]))
        scored.append((mean_iou, layer_idx))

    scored.sort(key=lambda item: item[0])
    selected = [layer_idx for _, layer_idx in scored[:max_layers]]
    return sorted(
        selected,
        key=lambda idx: float(per_layer_prompt_results[idx]["layer_relative"]),
    )


def _pair_positions(
    positions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if len(positions) < 2:
        raise ValueError("Need at least two positions to build predictor/target token pairs.")

    pairs: List[Dict[str, Any]] = []
    for idx in range(len(positions) - 1):
        predictor = positions[idx]
        target = positions[idx + 1]
        pairs.append(
            {
                "predictor_position": int(predictor["position"]),
                "target_position": int(target["position"]),
                "predictor_token": _clean_token(predictor.get("input_token")),
                "target_token": _clean_token(target.get("input_token")),
            }
        )
    return pairs


def _build_hover_text(
    layer_result: Dict[str, Any],
    predictor_position: Dict[str, Any],
    target_token: str,
    top_k: int,
    max_token_chars: int,
) -> str:
    shared = _format_token_block(
        predictor_position.get("intersection", []),
        max_items=top_k,
        max_chars=max_token_chars,
    )
    only_base = _format_token_block(
        predictor_position.get("only_base", []),
        max_items=top_k,
        max_chars=max_token_chars,
    )
    only_ft = _format_token_block(
        predictor_position.get("only_finetuned", []),
        max_items=top_k,
        max_chars=max_token_chars,
    )
    return (
        f"<b>Layer</b>: {layer_result['layer_relative']} (abs {layer_result['layer_absolute']})<br>"
        f"<b>Predictor token</b>: {_clean_token(predictor_position.get('input_token'))}<br>"
        f"<b>Target token</b>: {target_token}<br>"
        f"<b>Position</b>: {predictor_position['position']} → {predictor_position['position'] + 1}<br>"
        f"<b>IoU</b>: {float(predictor_position['iou']):.4f}<br>"
        f"<b>Shared top-{top_k}</b>: {shared}<br>"
        f"<b>Base-only</b>: {only_base}<br>"
        f"<b>Finetuned-only</b>: {only_ft}"
    )


def _build_cell_text(
    predictor_position: Dict[str, Any],
    top_k: int,
    max_token_chars: int,
) -> str:
    shared = _format_token_block(
        predictor_position.get("intersection", []),
        max_items=top_k,
        max_chars=max_token_chars,
    )
    base_only = _format_token_block(
        predictor_position.get("only_base", []),
        max_items=top_k,
        max_chars=max_token_chars,
    )
    ft_only = _format_token_block(
        predictor_position.get("only_finetuned", []),
        max_items=top_k,
        max_chars=max_token_chars,
    )
    return (
        f"IoU {float(predictor_position['iou']):.2f}<br>"
        f"S: {shared}<br>"
        f"B: {base_only}<br>"
        f"F: {ft_only}"
    )


def plot_logitdiff_next_token_verification_heatmap(
    payload_or_path: Dict[str, Any] | str | Path,
    *,
    prompt_index: int | None = None,
    prompt_text: str | None = None,
    top_k: int = 10,
    max_divergent_layers: int = 5,
    keep_last_layer_fraction: float = 0.5,
    include_prompt_tokens: bool = True,
    include_generated_tokens: bool = True,
    start_idx: int | None = None,
    end_idx: int | None = None,
    max_token_chars: int = 18,
    title: str | None = None,
    colorscale: str = "RdBu_r",
) -> go.Figure:
    payload = _load_payload(payload_or_path)
    results = _extract_results(payload)
    per_layer_prompt_results = _select_prompt(results, prompt_index, prompt_text)

    first_positions = per_layer_prompt_results[0]["positions"]
    filtered_positions: List[Dict[str, Any]] = []
    for position in first_positions:
        is_generated = bool(position.get("is_generated", False))
        if is_generated and not include_generated_tokens:
            continue
        if not is_generated and not include_prompt_tokens:
            continue
        filtered_positions.append(position)

    start = 0 if start_idx is None else start_idx
    stop = len(filtered_positions) if end_idx is None else end_idx
    filtered_positions = filtered_positions[start:stop]
    usable_pairs = _pair_positions(filtered_positions)

    selected_layer_indices = _choose_divergent_layers(
        per_layer_prompt_results,
        usable_positions=filtered_positions[:-1],
        keep_last_fraction=keep_last_layer_fraction,
        max_layers=max_divergent_layers,
    )
    if not selected_layer_indices:
        raise ValueError("No divergent layers selected for plotting.")

    layer_results = [per_layer_prompt_results[idx] for idx in selected_layer_indices]
    num_layers = len(layer_results)
    num_positions = len(usable_pairs)

    z = np.full((num_layers, num_positions), np.nan, dtype=float)
    text = np.empty((num_layers, num_positions), dtype=object)
    hover_text = np.empty((num_layers, num_positions), dtype=object)
    y_labels: List[str] = []

    position_to_payload = {
        int(position["position"]): position for position in first_positions
    }

    total_model_layers = max(
        int(layer_result["layer_absolute"]) for layer_result in per_layer_prompt_results
    ) + 1

    for row_idx, layer_result in enumerate(layer_results):
        y_labels.append(
            f"Layer {int(layer_result['layer_absolute']) + 1}/{total_model_layers}"
        )
        layer_positions = {
            int(position["position"]): position for position in layer_result["positions"]
        }
        for col_idx, pair in enumerate(usable_pairs):
            predictor_position = layer_positions[pair["predictor_position"]]
            z[row_idx, col_idx] = float(predictor_position["iou"])
            text[row_idx, col_idx] = _build_cell_text(
                predictor_position,
                top_k=top_k,
                max_token_chars=max_token_chars,
            )
            hover_text[row_idx, col_idx] = _build_hover_text(
                layer_result,
                predictor_position,
                pair["target_token"],
                top_k=top_k,
                max_token_chars=max_token_chars,
            )

    display_prompt = _display_prompt_text(per_layer_prompt_results[0]["prompt"])
    plot_title = title or (
        "Next-token verification heatmap"
        f"<br><sup>{display_prompt}</sup>"
    )

    x_vals = list(range(num_positions))
    predictor_labels = [pair["predictor_token"] for pair in usable_pairs]
    target_labels = [pair["target_token"] for pair in usable_pairs]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x_vals,
            y=list(range(num_layers)),
            zmin=0.0,
            zmax=1.0,
            colorscale=colorscale,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 10, "family": "Noto Sans, DejaVu Sans, Arial, sans-serif"},
            hovertext=hover_text,
            hoverinfo="text",
            xgap=1,
            ygap=1,
            colorbar={
                "title": {"text": f"Top-{top_k} IoU"},
                "tickformat": ".2f",
            },
        )
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_vals,
        ticktext=predictor_labels,
        tickangle=45,
        side="bottom",
        title_text="Predictor token (input tokens minus last token)",
        automargin=True,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=y_labels,
        autorange="reversed",
        title_text="Most divergent layers from last 50% of model",
        automargin=True,
    )

    fig.update_layout(
        title=plot_title,
        template="plotly_white",
        width=max(1200, 130 * num_positions),
        height=max(550, 120 * num_layers),
        margin={"l": 140, "r": 60, "t": 170, "b": 170},
        xaxis2={
            "anchor": "y",
            "overlaying": "x",
            "side": "top",
            "tickmode": "array",
            "tickvals": x_vals,
            "ticktext": target_labels,
            "tickangle": 45,
            "title": {"text": "Target token (input tokens minus first token)"},
            "automargin": True,
            "showgrid": False,
            "zeroline": False,
        },
    )

    return fig


def save_logitdiff_next_token_verification_html(
    payload_or_path: Dict[str, Any] | str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    fig = plot_logitdiff_next_token_verification_heatmap(payload_or_path, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".html":
        output_path = output_path.with_suffix(".html")
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return output_path


def save_logitdiff_next_token_verification_pdf(
    payload_or_path: Dict[str, Any] | str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")
    payload = _load_payload(payload_or_path)
    results = _extract_results(payload)
    prompt_index = kwargs.pop("prompt_index", None)
    prompt_text = kwargs.pop("prompt_text", None)
    top_k = kwargs.pop("top_k", 10)
    max_divergent_layers = kwargs.pop("max_divergent_layers", 5)
    keep_last_layer_fraction = kwargs.pop("keep_last_layer_fraction", 0.5)
    include_prompt_tokens = kwargs.pop("include_prompt_tokens", True)
    include_generated_tokens = kwargs.pop("include_generated_tokens", True)
    start_idx = kwargs.pop("start_idx", None)
    end_idx = kwargs.pop("end_idx", None)
    max_token_chars = kwargs.pop("max_token_chars", 18)
    title = kwargs.pop("title", None)

    per_layer_prompt_results = _select_prompt(results, prompt_index, prompt_text)
    first_positions = per_layer_prompt_results[0]["positions"]
    filtered_positions: List[Dict[str, Any]] = []
    for position in first_positions:
        is_generated = bool(position.get("is_generated", False))
        if is_generated and not include_generated_tokens:
            continue
        if not is_generated and not include_prompt_tokens:
            continue
        filtered_positions.append(position)

    start = 0 if start_idx is None else start_idx
    stop = len(filtered_positions) if end_idx is None else end_idx
    filtered_positions = filtered_positions[start:stop]
    usable_pairs = _pair_positions(filtered_positions)
    selected_layer_indices = _choose_divergent_layers(
        per_layer_prompt_results,
        usable_positions=filtered_positions[:-1],
        keep_last_fraction=keep_last_layer_fraction,
        max_layers=max_divergent_layers,
    )
    if not selected_layer_indices:
        raise ValueError("No divergent layers selected for plotting.")

    layer_results = [per_layer_prompt_results[idx] for idx in selected_layer_indices]
    num_layers = len(layer_results)
    num_positions = len(usable_pairs)
    z = np.full((num_layers, num_positions), np.nan, dtype=float)
    cell_text = np.empty((num_layers, num_positions), dtype=object)
    y_labels: List[str] = []
    total_model_layers = max(
        int(layer_result["layer_absolute"]) for layer_result in per_layer_prompt_results
    ) + 1

    for row_idx, layer_result in enumerate(layer_results):
        y_labels.append(
            f"L{int(layer_result['layer_absolute']) + 1}/{total_model_layers}"
        )
        layer_positions = {
            int(position["position"]): position for position in layer_result["positions"]
        }
        for col_idx, pair in enumerate(usable_pairs):
            predictor_position = layer_positions[pair["predictor_position"]]
            z[row_idx, col_idx] = float(predictor_position["iou"])
            shared = _format_token_block(
                predictor_position.get("intersection", []), top_k, max_token_chars
            )
            base_only = _format_token_block(
                predictor_position.get("only_base", []), top_k, max_token_chars
            )
            ft_only = _format_token_block(
                predictor_position.get("only_finetuned", []), top_k, max_token_chars
            )
            cell_text[row_idx, col_idx] = (
                f"{z[row_idx, col_idx]:.2f}\nS:{shared}\nB:{base_only}\nF:{ft_only}"
            )

    predictor_labels = [pair["predictor_token"] for pair in usable_pairs]
    target_labels = [pair["target_token"] for pair in usable_pairs]
    display_prompt = _display_prompt_text(per_layer_prompt_results[0]["prompt"])
    plot_title = title or f"Next-token verification heatmap\n{display_prompt}"

    fig_w = max(12, num_positions * 1.8)
    fig_h = max(5.5, num_layers * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(num_positions))
    ax.set_xticklabels(predictor_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Predictor token (input tokens minus last token)")
    ax.set_ylabel("Most divergent layers from last 50% of model")
    ax.set_title(plot_title, fontsize=12)

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(range(num_positions))
    top_ax.set_xticklabels(target_labels, rotation=45, ha="left", fontsize=9)
    top_ax.set_xlabel("Target token (input tokens minus first token)")

    for row_idx in range(num_layers):
        for col_idx in range(num_positions):
            value = z[row_idx, col_idx]
            color = "white" if value < 0.45 else "black"
            ax.text(
                col_idx,
                row_idx,
                cell_text[row_idx, col_idx],
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(f"Top-{top_k} IoU")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = [
    "plot_logitdiff_next_token_verification_heatmap",
    "save_logitdiff_next_token_verification_html",
    "save_logitdiff_next_token_verification_pdf",
]

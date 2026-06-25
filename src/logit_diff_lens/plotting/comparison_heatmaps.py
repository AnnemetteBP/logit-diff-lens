from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go


def _clean_token(token: str) -> str:
    token = str(token).replace("Ġ", " ").replace("▁", " ")
    token = token.replace("\n", "\\n")
    token = token.strip()
    return token or "[space]"


def _shifted_target_labels(tokens: list[str]) -> list[str]:
    cleaned = [_clean_token(token) for token in tokens]
    if not cleaned:
        return []
    return cleaned[1:] + ["[end]"]


def _to_position_values(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 2 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 1:
        return array
    raise ValueError(f"Expected scalar-like per-position metric, got shape {array.shape}")


def plot_comparison_metric_heatmap(
    comparison: dict[str, Any],
    *,
    metric_key: str = "jsd_ft_base",
    title: str | None = None,
    colorscale: str = "Viridis",
) -> go.Figure:
    layer_results = comparison["layer_results"]
    if not layer_results:
        raise ValueError("comparison['layer_results'] must not be empty")

    token_text = [_clean_token(token) for token in comparison["token_text"]]
    target_text = _shifted_target_labels(comparison["token_text"])
    num_layers = len(layer_results)
    num_positions = len(token_text)
    z = np.full((num_layers, num_positions), np.nan, dtype=float)
    hover = np.empty((num_layers, num_positions), dtype=object)
    y_labels: list[str] = []

    max_layer_index = max(
        int(layer["layer_index"] if isinstance(layer, dict) else layer.layer_index)
        for layer in layer_results
        if int(layer["layer_index"] if isinstance(layer, dict) else layer.layer_index) >= 0
    )
    total_layers = max_layer_index + 1 if max_layer_index >= 0 else len(layer_results)

    for row_idx, layer in enumerate(layer_results):
        layer_index = int(layer["layer_index"] if isinstance(layer, dict) else layer.layer_index)
        layer_name = layer["layer_name"] if isinstance(layer, dict) else layer.layer_name
        layer_metrics = layer["metrics"] if isinstance(layer, dict) else layer.metrics
        layer_num = layer_index
        if layer_num < 0:
            y_labels.append("Embedding")
        elif layer_name == "output":
            y_labels.append("Output")
        else:
            y_labels.append(f"L{layer_num + 1}/{total_layers}")
        values = _to_position_values(layer_metrics[metric_key])
        if values.shape[0] != num_positions:
            raise ValueError(
                f"Metric {metric_key} for layer {layer_name} has {values.shape[0]} positions, expected {num_positions}"
            )
        z[row_idx, :] = values.astype(float)
        for col_idx, metric_value in enumerate(values.tolist()):
            hover[row_idx, col_idx] = (
                f"<b>Layer</b>: {y_labels[row_idx]}<br>"
                f"<b>Input token</b>: {token_text[col_idx]}<br>"
                f"<b>Target token</b>: {target_text[col_idx]}<br>"
                f"<b>{metric_key}</b>: {float(metric_value):.6f}<br>"
                f"<b>Operand order</b>: {comparison.get('operand_order', 'unknown')}<br>"
                f"<b>Readout</b>: {comparison.get('readout_mode', 'unknown')}"
            )

    finite_values = z[np.isfinite(z)]
    zmin = float(finite_values.min()) if finite_values.size else 0.0
    zmax = float(finite_values.max()) if finite_values.size else 1.0
    if abs(zmax - zmin) < 1e-12:
        zmax = zmin + 1e-6

    width = max(980, 140 + 120 * num_positions)
    height = max(460, 180 + 72 * num_layers)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=list(range(num_positions)),
            y=list(range(num_layers)),
            text=hover,
            hoverinfo="text",
            colorscale=colorscale,
            colorbar=dict(title=metric_key),
            zmin=zmin,
            zmax=zmax,
        )
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=y_labels,
        autorange="reversed",
        title_text="Layers",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(num_positions)),
        ticktext=token_text,
        tickangle=-35,
        title_text="Input Tokens (t)",
        side="bottom",
    )
    fig.update_layout(
        width=width,
        height=height,
        title=dict(
            text=title or f"{metric_key} | {comparison.get('comparison_label', 'ft')} - {comparison.get('reference_label', 'base')}",
            x=0.5,
            xanchor="center",
        ),
        margin=dict(l=120, r=80, t=120, b=150),
        xaxis2=dict(
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=list(range(num_positions)),
            ticktext=target_text,
            tickangle=-35,
            title="Target Tokens (t+1)",
        ),
        template="plotly_white",
    )
    return fig


__all__ = ["plot_comparison_metric_heatmap"]

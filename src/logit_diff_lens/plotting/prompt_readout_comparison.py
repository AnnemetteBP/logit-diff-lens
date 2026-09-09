from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..diffing import compare_prompt_artifacts_ft_minus_base
from ..schemas import PromptDecodeArtifact
from .plotly_export import save_plotly_figure


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


def _build_heatmap_payload(comparison: dict[str, Any], metric_key: str) -> dict[str, Any]:
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
        if layer_index < 0:
            y_labels.append("Embedding")
        elif layer_name == "output":
            y_labels.append("Output")
        else:
            y_labels.append(f"L{layer_index + 1}/{total_layers}")
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

    return {
        "z": z,
        "hover": hover,
        "token_text": token_text,
        "target_text": target_text,
        "y_labels": y_labels,
    }


def plot_prompt_readout_comparison(
    ft_artifact: PromptDecodeArtifact,
    base_artifact: PromptDecodeArtifact,
    *,
    readout_modes: Sequence[str] = ("raw", "model_norm"),
    metric_key: str = "jsd_ft_base",
    top_k: int = 10,
    title: str | None = None,
    colorscale: str = "RdBu",
) -> go.Figure:
    if not readout_modes:
        raise ValueError("readout_modes must not be empty")

    comparisons = [
        compare_prompt_artifacts_ft_minus_base(
            ft_artifact,
            base_artifact,
            readout_mode=mode,
            topk=top_k,
            reference_token_ids=ft_artifact.token_ids,
        )
        for mode in readout_modes
    ]
    payloads = [_build_heatmap_payload(comparison, metric_key) for comparison in comparisons]

    finite_values = np.concatenate(
        [payload["z"][np.isfinite(payload["z"])] for payload in payloads if np.isfinite(payload["z"]).any()]
    )
    zmin = float(finite_values.min()) if finite_values.size else 0.0
    zmax = float(finite_values.max()) if finite_values.size else 1.0
    if abs(zmax - zmin) < 1e-12:
        zmax = zmin + 1e-6

    subplot_titles = [f"{mode}" for mode in readout_modes]
    fig = make_subplots(
        rows=1,
        cols=len(readout_modes),
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    top_axis_annotations: list[dict[str, Any]] = []

    for col_idx, payload in enumerate(payloads, start=1):
        showscale = col_idx == len(payloads)
        fig.add_trace(
            go.Heatmap(
                z=payload["z"],
                x=list(range(len(payload["token_text"]))),
                y=list(range(len(payload["y_labels"]))),
                text=payload["hover"],
                hoverinfo="text",
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title=metric_key) if showscale else None,
                showscale=showscale,
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(payload["token_text"]))),
            ticktext=payload["token_text"],
            tickangle=-35,
            title_text="Input Tokens (t)",
            side="bottom",
            row=1,
            col=col_idx,
        )
        xaxis_domain = fig.layout[f"xaxis{'' if col_idx == 1 else col_idx}"].domain
        top_axis_annotations.append(
            dict(
                x=sum(xaxis_domain) / 2.0,
                y=1.08,
                xref="paper",
                yref="paper",
                text="Target Tokens (t+1): " + " | ".join(payload["target_text"]),
                showarrow=False,
                font=dict(size=11),
                align="center",
            )
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(payloads[0]["y_labels"]))),
        ticktext=payloads[0]["y_labels"],
        autorange="reversed",
        title_text="Layers",
        row=1,
        col=1,
    )

    num_positions = len(payloads[0]["token_text"])
    num_layers = len(payloads[0]["y_labels"])
    fig.update_layout(
        title=dict(
            text=title
            or f"Prompt LogitDiff Side-by-Side | {metric_key} | {comparisons[0].get('comparison_label', 'ft')} <> {comparisons[0].get('reference_label', 'base')}",
            x=0.5,
            xanchor="center",
        ),
        annotations=list(fig.layout.annotations) + top_axis_annotations,
        template="plotly_white",
        width=max(1100, 520 * len(readout_modes) + 120 * num_positions),
        height=max(520, 180 + 72 * num_layers),
        margin=dict(l=120, r=80, t=170, b=150),
    )
    return fig


def save_prompt_readout_comparison_figure(
    fig: go.Figure,
    path: str,
    *,
    format: str | None = None,
) -> None:
    save_plotly_figure(fig, path, format=format)


__all__ = [
    "plot_prompt_readout_comparison",
    "save_prompt_readout_comparison_figure",
]

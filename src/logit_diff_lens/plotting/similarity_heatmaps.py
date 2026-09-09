from __future__ import annotations

from pathlib import Path
from typing import Literal

import plotly.graph_objects as go

from .plotly_export import save_plotly_figure
from ..schemas.similarity_outputs import LayerPairSimilarityArtifact, SimilarityRunArtifact
from ..similarity.io import load_similarity_run_artifact


SimilarityHeatmapKind = Literal["raw", "calibrated", "p_value", "adjusted_p_value", "null_percentile"]


def _load_similarity_artifact(source: SimilarityRunArtifact | str | Path) -> SimilarityRunArtifact:
    if isinstance(source, SimilarityRunArtifact):
        return source
    return load_similarity_run_artifact(source)


def _select_matrix_result(
    artifact: SimilarityRunArtifact,
    *,
    matrix_index: int = 0,
) -> LayerPairSimilarityArtifact:
    if not artifact.matrix_results:
        raise ValueError("Similarity artifact contains no matrix_results to plot")
    try:
        return artifact.matrix_results[matrix_index]
    except IndexError as exc:
        raise ValueError(
            f"Requested matrix_index={matrix_index} but artifact only has {len(artifact.matrix_results)} matrix result(s)"
        ) from exc


def _layer_labels(layer_indices: list[int]) -> list[str]:
    labels: list[str] = []
    positive = [idx for idx in layer_indices if idx >= 0]
    total_layers = (max(positive) + 1) if positive else len(layer_indices)
    for idx in layer_indices:
        if idx < 0:
            labels.append("Embedding")
        else:
            labels.append(f"L{idx + 1}/{total_layers}")
    return labels


def plot_similarity_matrix_heatmap(
    source: SimilarityRunArtifact | str | Path,
    *,
    matrix_index: int = 0,
    plot_kind: SimilarityHeatmapKind = "calibrated",
    title: str | None = None,
    colorscale: str = "Viridis",
) -> go.Figure:
    artifact = _load_similarity_artifact(source)
    matrix = _select_matrix_result(artifact, matrix_index=matrix_index)

    if plot_kind == "raw":
        z = matrix.raw_matrix.detach().cpu().numpy()
        colorbar_title = f"Raw {matrix.metric_name}"
    elif plot_kind == "calibrated":
        z = matrix.calibrated_matrix.detach().cpu().numpy()
        colorbar_title = f"Calibrated {matrix.metric_name}"
    elif plot_kind == "p_value":
        z = matrix.p_value_matrix.detach().cpu().numpy()
        colorbar_title = "p-value"
    elif plot_kind == "adjusted_p_value":
        if matrix.adjusted_p_value_matrix is None:
            raise ValueError("Similarity artifact does not contain adjusted p-values to plot")
        z = matrix.adjusted_p_value_matrix.detach().cpu().numpy()
        colorbar_title = "adjusted p-value"
    elif plot_kind == "null_percentile":
        if matrix.null_percentile_matrix is None:
            raise ValueError("Similarity artifact does not contain null percentiles to plot")
        z = matrix.null_percentile_matrix.detach().cpu().numpy()
        colorbar_title = "null percentile"
    else:
        raise ValueError(f"Unsupported plot_kind: {plot_kind!r}")

    y_labels = _layer_labels(matrix.layer_indices_a)
    x_labels = _layer_labels(matrix.layer_indices_b)
    default_title = (
        f"{artifact.side_a_label} vs {artifact.side_b_label} | "
        f"{matrix.metric_name} ({plot_kind})"
    )

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=list(range(len(x_labels))),
                y=list(range(len(y_labels))),
                colorscale=colorscale,
                colorbar=dict(title=colorbar_title),
                hovertemplate=(
                    "A layer: %{y}<br>"
                    "B layer: %{x}<br>"
                    "value: %{z:.6f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(x_labels))),
        ticktext=x_labels,
        title_text=f"{artifact.side_b_label} layers",
        side="bottom",
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(y_labels))),
        ticktext=y_labels,
        title_text=f"{artifact.side_a_label} layers",
        autorange="reversed",
    )
    fig.update_layout(
        title=dict(text=title or default_title, x=0.5, xanchor="center"),
        template="plotly_white",
        width=max(760, 120 + 70 * len(x_labels)),
        height=max(520, 140 + 60 * len(y_labels)),
        margin=dict(l=120, r=60, t=120, b=120),
    )
    return fig


def plot_similarity_aggregate_summary(
    source: SimilarityRunArtifact | str | Path,
    *,
    matrix_index: int = 0,
    title: str | None = None,
) -> go.Figure:
    artifact = _load_similarity_artifact(source)
    matrix = _select_matrix_result(artifact, matrix_index=matrix_index)
    null_scores = matrix.aggregate_null_scores
    default_title = (
        f"{artifact.side_a_label} vs {artifact.side_b_label} | "
        f"aggregate null summary ({matrix.metric_name})"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=null_scores,
            name="Null scores",
            nbinsx=min(40, max(10, len(null_scores) // 5 or 10)),
            marker=dict(color="#8db3e2"),
            opacity=0.85,
        )
    )
    fig.add_vline(
        x=matrix.aggregate_raw_value,
        line_color="#c62828",
        line_width=3,
        annotation_text="Observed",
        annotation_position="top right",
    )
    fig.add_vline(
        x=matrix.aggregate_critical_value,
        line_color="#2e7d32",
        line_dash="dash",
        line_width=3,
        annotation_text="Critical",
        annotation_position="top left",
    )
    fig.update_layout(
        title=dict(text=title or default_title, x=0.5, xanchor="center"),
        template="plotly_white",
        width=900,
        height=520,
        margin=dict(l=90, r=40, t=110, b=90),
        barmode="overlay",
        xaxis_title=f"Aggregate {matrix.aggregate_statistic_name}",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def save_similarity_figure(
    fig: go.Figure,
    path: str | Path,
    *,
    format: str | None = None,
) -> Path:
    return save_plotly_figure(fig, path, format=format)


__all__ = [
    "plot_similarity_aggregate_summary",
    "plot_similarity_matrix_heatmap",
    "save_similarity_figure",
]

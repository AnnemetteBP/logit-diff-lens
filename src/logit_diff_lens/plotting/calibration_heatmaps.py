from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from .plotly_export import save_plotly_figure
from ..calibration.io import load_calibration_run_artifact
from ..schemas.calibration_outputs import CalibrationMatrixResult, CalibrationRunArtifact, CalibrationSummaryResult


def _load_calibration_artifact(source: CalibrationRunArtifact | str | Path) -> CalibrationRunArtifact:
    if isinstance(source, CalibrationRunArtifact):
        return source
    return load_calibration_run_artifact(source)


def _select_matrix_result(artifact: CalibrationRunArtifact, *, metric_name: str | None, matrix_index: int) -> CalibrationMatrixResult:
    if metric_name is not None:
        for item in artifact.matrix_results:
            if item.metric_name == metric_name:
                return item
        raise ValueError(f"Calibration artifact has no matrix metric named {metric_name!r}")
    if not artifact.matrix_results:
        raise ValueError("Calibration artifact contains no matrix_results")
    return artifact.matrix_results[matrix_index]


def _select_summary_result(
    artifact: CalibrationRunArtifact,
    *,
    metric_name: str,
    axis_kind: str,
) -> CalibrationSummaryResult:
    for item in artifact.summary_results:
        if item.metric_name == metric_name and item.axis_kind == axis_kind:
            return item
    raise ValueError(f"Calibration artifact has no summary for metric={metric_name!r} axis_kind={axis_kind!r}")


def _layer_labels(layer_indices: list[int]) -> list[str]:
    positive = [idx for idx in layer_indices if idx >= 0]
    total_layers = (max(positive) + 1) if positive else len(layer_indices)
    labels: list[str] = []
    for idx in layer_indices:
        labels.append("Embedding" if idx < 0 else f"L{idx + 1}/{total_layers}")
    return labels


def plot_calibration_matrix_heatmap(
    source: CalibrationRunArtifact | str | Path,
    *,
    metric_name: str | None = None,
    matrix_index: int = 0,
    title: str | None = None,
    colorscale: str = "Viridis",
) -> go.Figure:
    artifact = _load_calibration_artifact(source)
    matrix = _select_matrix_result(artifact, metric_name=metric_name, matrix_index=matrix_index)
    z = matrix.value_matrix.detach().cpu().numpy()
    hover = []
    for row_idx, layer_index in enumerate(artifact.layer_indices):
        hover_row: list[str] = []
        for col_idx, position_index in enumerate(artifact.position_indices):
            count = float(matrix.sample_count_matrix[row_idx, col_idx].item())
            value = float(matrix.value_matrix[row_idx, col_idx].item())
            extra = [f"count={count:.0f}"]
            if matrix.confidence_mean_matrix is not None:
                extra.append(f"conf={float(matrix.confidence_mean_matrix[row_idx, col_idx].item()):.4f}")
            if matrix.accuracy_mean_matrix is not None:
                extra.append(f"acc={float(matrix.accuracy_mean_matrix[row_idx, col_idx].item()):.4f}")
            hover_row.append(
                f"layer={layer_index}<br>position={position_index}<br>value={value:.6f}<br>" + "<br>".join(extra)
            )
        hover.append(hover_row)
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=z,
                x=artifact.position_indices,
                y=list(range(len(artifact.layer_indices))),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                colorscale=colorscale,
                colorbar=dict(title=matrix.metric_name),
            )
        ]
    )
    fig.update_xaxes(title_text="Token position")
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(artifact.layer_indices))),
        ticktext=_layer_labels(artifact.layer_indices),
        title_text=f"{artifact.side_a_label} layers",
        autorange="reversed",
    )
    fig.update_layout(
        title=dict(
            text=title or f"{artifact.side_a_label} vs {artifact.side_b_label} | {matrix.metric_name}",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=max(840, 120 + 38 * len(artifact.position_indices)),
        height=max(520, 140 + 44 * len(artifact.layer_indices)),
        margin=dict(l=120, r=60, t=110, b=100),
    )
    return fig


def plot_calibration_summary(
    source: CalibrationRunArtifact | str | Path,
    *,
    metric_name: str = "top1_ece",
    axis_kind: str = "layer",
    title: str | None = None,
) -> go.Figure:
    artifact = _load_calibration_artifact(source)
    summary = _select_summary_result(artifact, metric_name=metric_name, axis_kind=axis_kind)
    x = summary.axis_indices
    y = summary.values.detach().cpu().numpy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=metric_name,
            line=dict(color="#0b6e99", width=3),
            marker=dict(size=7),
        )
    )
    if summary.ci_low is not None and summary.ci_high is not None:
        fig.add_trace(
            go.Scatter(
                x=x + list(reversed(x)),
                y=summary.ci_high.detach().cpu().tolist() + list(reversed(summary.ci_low.detach().cpu().tolist())),
                fill="toself",
                fillcolor="rgba(11,110,153,0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        title=dict(
            text=title or f"{artifact.side_a_label} vs {artifact.side_b_label} | {metric_name} by {axis_kind}",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=940,
        height=520,
        margin=dict(l=90, r=40, t=110, b=90),
        xaxis_title="Layer index" if axis_kind == "layer" else "Token position",
        yaxis_title=metric_name,
    )
    return fig


def save_calibration_figure(fig: go.Figure, path: str | Path, *, format: str | None = None) -> Path:
    return save_plotly_figure(fig, path, format=format)


__all__ = [
    "plot_calibration_matrix_heatmap",
    "plot_calibration_summary",
    "save_calibration_figure",
]

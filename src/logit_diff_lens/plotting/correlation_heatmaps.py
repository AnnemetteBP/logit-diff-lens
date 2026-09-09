from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .plotly_export import save_plotly_figure
from plotly.colors import sample_colorscale

from ..correlations.io import load_correlation_run_artifact
from ..schemas.correlation_outputs import CorrelationRunArtifact, LayerPairCorrelationArtifact


def _load_artifact(source: CorrelationRunArtifact | str | Path) -> CorrelationRunArtifact:
    if isinstance(source, CorrelationRunArtifact):
        return source
    return load_correlation_run_artifact(source)


def _select_matrix(artifact: CorrelationRunArtifact, *, metric_name: str | None, matrix_index: int) -> LayerPairCorrelationArtifact:
    if metric_name is not None:
        for item in artifact.matrix_results:
            if item.metric_name == metric_name:
                return item
        raise ValueError(f"No correlation matrix found for metric_name={metric_name!r}")
    return artifact.matrix_results[matrix_index]


def _layer_labels(layer_indices: list[int]) -> list[str]:
    positive = [idx for idx in layer_indices if idx >= 0]
    total_layers = (max(positive) + 1) if positive else len(layer_indices)
    labels: list[str] = []
    for idx in layer_indices:
        labels.append("Embedding" if idx < 0 else f"L{idx + 1}/{total_layers}")
    return labels


def _color_for_value(value: float, *, colorscale: str) -> str:
    if not np.isfinite(value):
        return "rgba(235,235,235,1.0)"
    clipped = max(-1.0, min(1.0, float(value)))
    normalized = 0.5 * (clipped + 1.0)
    return str(sample_colorscale(colorscale, [normalized])[0])


def plot_prompt_correlation_heatmap(
    source: CorrelationRunArtifact | str | Path,
    *,
    metric_name: str | None = None,
    matrix_index: int = 0,
    title: str | None = None,
    colorscale: str = "RdBu",
    significance_level: float = 0.05,
) -> go.Figure:
    artifact = _load_artifact(source)
    matrix = _select_matrix(artifact, metric_name=metric_name, matrix_index=matrix_index)
    pearson_r = matrix.pearson_r_matrix.detach().cpu().numpy()
    spearman_r = matrix.spearman_r_matrix.detach().cpu().numpy()
    rows, cols = pearson_r.shape
    x = list(range(cols))
    y = list(range(rows))

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=np.zeros((rows, cols), dtype=float),
                x=x,
                y=y,
                zmin=-1.0,
                zmax=1.0,
                zmid=0.0,
                colorscale=colorscale,
                opacity=0.0,
                colorbar=dict(title="Correlation"),
                hoverinfo="skip",
                showscale=True,
            )
        ]
    )

    shapes = []
    pearson_x: list[float] = []
    pearson_y: list[float] = []
    pearson_size: list[float] = []
    pearson_color: list[str] = []
    pearson_line_color: list[str] = []
    pearson_line_width: list[float] = []
    pearson_text: list[str] = []
    spearman_x: list[float] = []
    spearman_y: list[float] = []
    spearman_size: list[float] = []
    spearman_color: list[str] = []
    spearman_line_color: list[str] = []
    spearman_line_width: list[float] = []
    spearman_text: list[str] = []

    for row_idx in range(rows):
        for col_idx in range(cols):
            p_r = float(matrix.pearson_r_matrix[row_idx, col_idx].item())
            p_p = float(matrix.pearson_p_matrix[row_idx, col_idx].item())
            p_q = float(matrix.pearson_q_matrix[row_idx, col_idx].item())
            s_r = float(matrix.spearman_r_matrix[row_idx, col_idx].item())
            s_p = float(matrix.spearman_p_matrix[row_idx, col_idx].item())
            s_q = float(matrix.spearman_q_matrix[row_idx, col_idx].item())
            n = int(matrix.sample_count_matrix[row_idx, col_idx].item())
            if not np.isfinite(p_r) or not np.isfinite(s_r):
                continue

            x0 = col_idx - 0.5
            x1 = col_idx + 0.5
            y0 = row_idx - 0.5
            y1 = row_idx + 0.5
            pearson_fill = _color_for_value(p_r, colorscale=colorscale)
            spearman_fill = _color_for_value(s_r, colorscale=colorscale)
            pearson_sig = np.isfinite(p_q) and p_q < significance_level
            spearman_sig = np.isfinite(s_q) and s_q < significance_level

            shapes.append(
                dict(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor="rgba(250,250,250,1.0)",
                    line=dict(color="rgba(210,210,210,1.0)", width=1),
                    layer="below",
                )
            )
            shapes.append(
                dict(
                    type="line",
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    line=dict(color="rgba(180,180,180,1.0)", width=1.2),
                    layer="below",
                )
            )
            pearson_x.append(col_idx + 0.18)
            pearson_y.append(row_idx - 0.18)
            pearson_size.append(6.0 + 18.0 * abs(p_r))
            pearson_color.append(pearson_fill)
            pearson_line_color.append("rgba(20,20,20,0.95)" if pearson_sig else "rgba(120,120,120,0.8)")
            pearson_line_width.append(2.2 if pearson_sig else 0.9)
            pearson_text.append(
                "<br>".join(
                    [
                        f"A layer: {matrix.layer_indices_a[row_idx]}",
                        f"B layer: {matrix.layer_indices_b[col_idx]}",
                        "Half: Pearson",
                        f"Pearson r: {p_r:.4f}",
                        f"Pearson p: {p_p:.4g}",
                        f"Pearson q: {p_q:.4g}",
                        (
                            "Pearson CI95: "
                            f"[{float(matrix.pearson_ci_low_matrix[row_idx, col_idx].item()):.4f}, "
                            f"{float(matrix.pearson_ci_high_matrix[row_idx, col_idx].item()):.4f}]"
                        ),
                        f"n: {n}",
                    ]
                )
            )
            spearman_x.append(col_idx - 0.18)
            spearman_y.append(row_idx + 0.18)
            spearman_size.append(6.0 + 18.0 * abs(s_r))
            spearman_color.append(spearman_fill)
            spearman_line_color.append("rgba(20,20,20,0.95)" if spearman_sig else "rgba(120,120,120,0.8)")
            spearman_line_width.append(2.2 if spearman_sig else 0.9)
            spearman_text.append(
                "<br>".join(
                    [
                        f"A layer: {matrix.layer_indices_a[row_idx]}",
                        f"B layer: {matrix.layer_indices_b[col_idx]}",
                        "Half: Spearman",
                        f"Spearman rho: {s_r:.4f}",
                        f"Spearman p: {s_p:.4g}",
                        f"Spearman q: {s_q:.4g}",
                        (
                            "Spearman CI95: "
                            f"[{float(matrix.spearman_ci_low_matrix[row_idx, col_idx].item()):.4f}, "
                            f"{float(matrix.spearman_ci_high_matrix[row_idx, col_idx].item()):.4f}]"
                        ),
                        f"n: {n}",
                    ]
                )
            )

    fig.add_trace(
        go.Scatter(
            x=pearson_x,
            y=pearson_y,
            mode="markers",
            marker=dict(
                size=pearson_size,
                color=pearson_color,
                line=dict(color=pearson_line_color, width=pearson_line_width),
                symbol="circle",
            ),
            text=pearson_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            name="Pearson",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spearman_x,
            y=spearman_y,
            mode="markers",
            marker=dict(
                size=spearman_size,
                color=spearman_color,
                line=dict(color=spearman_line_color, width=spearman_line_width),
                symbol="circle",
            ),
            text=spearman_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            name="Spearman",
        )
    )

    fig.update_layout(
        title=dict(
            text=title or f"{artifact.side_a_label} vs {artifact.side_b_label} | {matrix.metric_name} correlations",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=max(780, 130 + 74 * cols),
        height=max(540, 150 + 68 * rows),
        margin=dict(l=120, r=70, t=120, b=120),
        shapes=shapes,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=x,
        ticktext=_layer_labels(matrix.layer_indices_b),
        title_text=f"{artifact.side_b_label} layers",
        range=[-0.5, cols - 0.5],
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=y,
        ticktext=_layer_labels(matrix.layer_indices_a),
        title_text=f"{artifact.side_a_label} layers",
        autorange="reversed",
        range=[rows - 0.5, -0.5],
    )
    return fig


def save_correlation_figure(fig: go.Figure, path: str | Path, *, format: str | None = None) -> Path:
    return save_plotly_figure(fig, path, format=format)


__all__ = [
    "plot_prompt_correlation_heatmap",
    "save_correlation_figure",
]

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from .plotly_export import save_plotly_figure
from ..prisms import (
    load_prompt_diff_prism_heatmap_artifact,
    load_prompt_diff_prism_summary_artifact,
    load_prompt_prism_artifact,
)
from ..schemas.prism_outputs import (
    PromptDiffPrismHeatmapArtifact,
    PromptDiffPrismSummaryArtifact,
    PromptPrismArtifact,
)


def _load_prism(source: PromptPrismArtifact | str | Path) -> PromptPrismArtifact:
    if isinstance(source, PromptPrismArtifact):
        return source
    return load_prompt_prism_artifact(source)


def _load_diff_prism_heatmap(source: PromptDiffPrismHeatmapArtifact | str | Path) -> PromptDiffPrismHeatmapArtifact:
    if isinstance(source, PromptDiffPrismHeatmapArtifact):
        return source
    return load_prompt_diff_prism_heatmap_artifact(source)


def _load_diff_prism_summary(source: PromptDiffPrismSummaryArtifact | str | Path) -> PromptDiffPrismSummaryArtifact:
    if isinstance(source, PromptDiffPrismSummaryArtifact):
        return source
    return load_prompt_diff_prism_summary_artifact(source)


def _pretty_subblock_label(label: str) -> str:
    if label == "embedding":
        return "Embedding"
    if label == "output_l+1":
        return "Output"
    if label.startswith("attn_"):
        return f"Attn {label.split('_', 1)[1]}"
    if label.startswith("mlp_"):
        return f"MLP {label.split('_', 1)[1]}"
    if label.startswith("full_layer_"):
        return f"Layer {label.split('_', 2)[2]}"
    return label


def _clean_token_label(text: str, token_id: int, max_token_chars: int) -> str:
    truncated = text if len(text) <= max_token_chars else text[: max_token_chars - 1] + "…"
    return f"{truncated} ({token_id})"


def _prism_figure_height(num_rows: int) -> int:
    return max(420, min(2600, 180 * max(1, num_rows) + 120))


def _base_prism_layout(fig: go.Figure, *, title: str, x_labels: list[str], num_rows: int) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        template="plotly_white",
        width=max(1100, 120 * max(1, len(x_labels))),
        height=_prism_figure_height(num_rows),
        margin=dict(l=120, r=40, t=90, b=120),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Prism subblocks", tickangle=35, type="category")
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="#666")
    return fig


def _stderr(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[dim] <= 1:
        return torch.zeros_like(x.mean(dim=dim))
    return x.std(dim=dim, unbiased=True) / (x.shape[dim] ** 0.5)


def _bootstrap_ci(x: torch.Tensor, *, num_draws: int = 500, alpha: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] <= 1:
        mean = x.mean(dim=0)
        return mean, mean
    idx = torch.randint(low=0, high=x.shape[0], size=(num_draws, x.shape[0]))
    samples = x.index_select(0, idx.reshape(-1)).reshape(num_draws, x.shape[0], *x.shape[1:]).mean(dim=1)
    lower = torch.quantile(samples, alpha / 2.0, dim=0)
    upper = torch.quantile(samples, 1.0 - alpha / 2.0, dim=0)
    return lower, upper


def plot_prompt_prism(
    prism: PromptPrismArtifact | str | Path,
    *,
    plot_mode: str = "per_token",
    uncertainty: str = "none",
    position_index: int | None = None,
    max_token_chars: int = 14,
    title: str | None = None,
) -> go.Figure:
    artifact = _load_prism(prism)
    contributions = artifact.contribution_tensor.to(torch.float32).cpu()  # [positions, components, tokens]
    x_labels = [_pretty_subblock_label(label) for label in artifact.component_labels]
    token_labels = [
        _clean_token_label(text, int(token_id), max_token_chars)
        for text, token_id in zip(artifact.selected_token_text, artifact.selected_token_ids.tolist())
    ]

    mean = None
    err = None
    lower = None
    upper = None
    if plot_mode == "per_token":
        pos_idx = len(artifact.position_indices) - 1 if position_index is None else int(position_index)
        if pos_idx < 0 or pos_idx >= len(artifact.position_indices):
            raise ValueError(
                f"position_index={pos_idx} out of range for prism with {len(artifact.position_indices)} positions"
            )
        selected_position = artifact.position_indices[pos_idx]
        series = contributions[pos_idx]
        title_text = title or f"Prompt Prism | {artifact.prompt_text} | {artifact.readout_mode} | position {selected_position}"
    elif plot_mode == "mean_over_positions":
        mean = contributions.mean(dim=0)
        if uncertainty == "stderr":
            err = _stderr(contributions, dim=0)
        elif uncertainty == "bootstrap_ci":
            lower, upper = _bootstrap_ci(contributions)
        elif uncertainty != "none":
            raise ValueError(f"Unsupported uncertainty={uncertainty!r}")
        series = mean
        title_text = title or (
            f"Prompt Prism | {artifact.prompt_text} | {artifact.readout_mode} | "
            f"Mean over {len(artifact.position_indices)} positions"
        )
    else:
        raise ValueError(f"Unsupported plot_mode={plot_mode!r}")
    fig = make_subplots(rows=len(token_labels), cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for token_idx, token_label in enumerate(token_labels, start=1):
        y = series[:, token_idx - 1]
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y.tolist(),
                mode="lines",
                line=dict(width=3, color="#1f77b4"),
                hovertemplate="<br>".join(
                    [
                        "Subblock: %{x}",
                        "Logit contribution: %{y:.4f}",
                        f"Readout mode: {artifact.readout_mode}",
                        "<extra></extra>",
                    ]
                ),
            ),
            row=token_idx,
            col=1,
        )
        if err is not None:
            upper_y = y + err[:, token_idx - 1]
            lower_y = y - err[:, token_idx - 1]
            fig.add_trace(
                go.Scatter(
                    x=x_labels + x_labels[::-1],
                    y=upper_y.tolist() + lower_y.flip(0).tolist(),
                    fill="toself",
                    fillcolor="rgba(31,119,180,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                ),
                row=token_idx,
                col=1,
            )
        elif lower is not None and upper is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_labels + x_labels[::-1],
                    y=upper[:, token_idx - 1].tolist() + lower[:, token_idx - 1].flip(0).tolist(),
                    fill="toself",
                    fillcolor="rgba(31,119,180,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                ),
                row=token_idx,
                col=1,
            )
        fig.update_yaxes(title_text=token_label, row=token_idx, col=1)
    return _base_prism_layout(fig, title=title_text, x_labels=x_labels, num_rows=len(token_labels))


def plot_prompt_prism_comparison(
    prism_a: PromptPrismArtifact | str | Path,
    prism_b: PromptPrismArtifact | str | Path,
    *,
    side_a_label: str = "artifact_a",
    side_b_label: str = "artifact_b",
    title: str | None = None,
) -> go.Figure:
    artifact_a = _load_prism(prism_a)
    artifact_b = _load_prism(prism_b)
    if artifact_a.component_labels != artifact_b.component_labels:
        raise ValueError("Prompt prism comparison requires matching component labels")
    if artifact_a.selected_token_ids.shape != artifact_b.selected_token_ids.shape:
        raise ValueError("Prompt prism comparison requires matching selected token shapes")

    mean_a = artifact_a.contribution_tensor.to(torch.float32).mean(dim=0)
    mean_b = artifact_b.contribution_tensor.to(torch.float32).mean(dim=0)
    x_labels = [_pretty_subblock_label(label) for label in artifact_a.component_labels]
    num_tokens = len(artifact_a.selected_token_ids.tolist())
    fig = make_subplots(rows=num_tokens, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for row_idx, token_id in enumerate(artifact_a.selected_token_ids.tolist(), start=1):
        token_idx = row_idx - 1
        token_text = artifact_a.selected_token_text[token_idx]
        label = _clean_token_label(token_text, int(token_id), 14)
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=mean_a[:, token_idx].tolist(),
                mode="lines",
                line=dict(width=3, color="#1f77b4"),
                hovertemplate=f"{side_a_label}<br>Subblock: %{{x}}<br>Mean logit contribution: %{{y:.4f}}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=mean_b[:, token_idx].tolist(),
                mode="lines",
                line=dict(width=3, dash="dash", color="#d62728"),
                hovertemplate=f"{side_b_label}<br>Subblock: %{{x}}<br>Mean logit contribution: %{{y:.4f}}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text=label, row=row_idx, col=1)
    return _base_prism_layout(
        fig,
        title=title or f"Prompt Prism Comparison | {side_a_label} vs {side_b_label}",
        x_labels=x_labels,
        num_rows=num_tokens,
    )


def plot_prompt_diff_prism(
    prism: PromptDiffPrismHeatmapArtifact | str | Path,
    *,
    max_token_chars: int = 14,
    title: str | None = None,
) -> go.Figure:
    artifact = _load_diff_prism_heatmap(prism)
    matrix = artifact.contribution_matrix.to(torch.float32).cpu()
    x_labels = [_pretty_subblock_label(label) for label in artifact.component_labels]
    num_tokens = len(artifact.selected_token_ids.tolist())
    fig = make_subplots(rows=num_tokens, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    for row_idx, token_id in enumerate(artifact.selected_token_ids.tolist(), start=1):
        token_idx = row_idx - 1
        token_label = _clean_token_label(artifact.selected_token_text[token_idx], int(token_id), max_token_chars)
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=matrix[:, token_idx].tolist(),
                mode="lines",
                line=dict(width=3, color="#1f77b4"),
                hovertemplate="<br>".join(
                    [
                        "Subblock: %{x}",
                        "Delta logit contribution: %{y:.4f}",
                        f"Delta: {artifact.delta_definition}",
                        f"Readout mode: {artifact.readout_mode}",
                        "<extra></extra>",
                    ]
                ),
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text=token_label, row=row_idx, col=1)
    return _base_prism_layout(
        fig,
        title=title or f"Prompt Diff Prism | Delta = {artifact.delta_definition} | {artifact.readout_mode}",
        x_labels=x_labels,
        num_rows=num_tokens,
    )


def plot_prompt_diff_prism_summary(
    source: PromptDiffPrismSummaryArtifact | str | Path,
    *,
    title: str | None = None,
) -> go.Figure:
    artifact = _load_diff_prism_summary(source)
    x_labels = [_pretty_subblock_label(label) for label in artifact.component_labels]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=artifact.raw_component_scores.to(torch.float32).cpu().tolist(),
            mode="lines",
            line=dict(width=3, color="#1f77b4"),
            name="Non-calibrated",
            hovertemplate="Component: %{x}<br>Non-calibrated score: %{y:.4f}<extra></extra>",
        )
    )
    if artifact.calibrated_component_scores is not None:
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=artifact.calibrated_component_scores.to(torch.float32).cpu().tolist(),
                mode="lines",
                line=dict(width=3, dash="dash", color="#d62728"),
                name="Calibrated",
                hovertemplate="Component: %{x}<br>Calibrated score: %{y:.4f}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(
            text=title or (
                f"Prompt Diff Prism Summary | {artifact.summary_kind} | "
                f"{artifact.delta_definition} | {artifact.readout_mode}"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=max(900, 120 * max(1, len(x_labels))),
        height=520,
        margin=dict(l=100, r=40, t=90, b=120),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="Prism subblocks", tickangle=35, type="category")
    fig.update_yaxes(title_text="Prism delta summary", zeroline=True, zerolinewidth=1, zerolinecolor="#666")
    return fig


def save_prompt_prism_comparison_figure(
    fig: go.Figure,
    path: str | Path,
    *,
    format: str | None = None,
) -> Path:
    return save_plotly_figure(fig, path, format=format)


__all__ = [
    "plot_prompt_diff_prism",
    "plot_prompt_diff_prism_summary",
    "plot_prompt_prism",
    "plot_prompt_prism_comparison",
    "save_prompt_prism_comparison_figure",
]

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from .legacy_heatmaps.logit_lens_plotter import plot_logit_lens_heatmap
from .plotly_export import save_plotly_figure


def plot_single_model_logit_lens_heatmap(
    source: Any,
    *,
    prompt: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    return plot_logit_lens_heatmap(source, prompt=prompt, **kwargs)


def save_single_model_logit_lens_heatmap(
    source: Any,
    output_path: str | Path,
    *,
    prompt: str | None = None,
    format: str | None = None,
    **kwargs: Any,
) -> str:
    fig = plot_single_model_logit_lens_heatmap(source, prompt=prompt, **kwargs)
    return str(save_plotly_figure(fig, output_path, format=format))


__all__ = [
    "plot_single_model_logit_lens_heatmap",
    "save_single_model_logit_lens_heatmap",
]

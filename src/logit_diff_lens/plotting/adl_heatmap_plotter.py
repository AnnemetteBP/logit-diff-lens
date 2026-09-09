"""Canonical ADL heatmap entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from .plotly_export import save_plotly_figure
from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.adl_plotter import (
    plot_adl_heatmap,
)


def plot_adl_case_heatmap(source: Any, *, prompt: str | None = None, **kwargs: Any) -> go.Figure:
    return plot_adl_heatmap(source, prompt=prompt, **kwargs)


def save_adl_case_heatmap(
    source: Any,
    output_path: str | Path,
    *,
    prompt: str | None = None,
    format: str | None = None,
    **kwargs: Any,
) -> str:
    fig = plot_adl_case_heatmap(source, prompt=prompt, **kwargs)
    return str(save_plotly_figure(fig, output_path, format=format))


__all__ = [
    "plot_adl_case_heatmap",
    "save_adl_case_heatmap",
]

"""Canonical LDL heatmap entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.ldl_plotter import (
    plot_ldl_heatmap,
    save_ldl_heatmap_html,
    save_ldl_heatmap_pdf,
)


def plot_ldl_case_heatmap(source: Any, *, prompt: str | None = None, **kwargs: Any) -> go.Figure:
    return plot_ldl_heatmap(source, prompt=prompt, **kwargs)


def save_ldl_case_heatmap(
    source: Any,
    output_path: str | Path,
    *,
    prompt: str | None = None,
    format: str | None = None,
    **kwargs: Any,
) -> str:
    fig = plot_ldl_case_heatmap(source, prompt=prompt, **kwargs)
    path = str(output_path)
    resolved_format = (format or Path(path).suffix.lstrip(".") or "pdf").lower()
    if resolved_format == "html":
        return save_ldl_heatmap_html(fig, path)
    if resolved_format == "pdf":
        return save_ldl_heatmap_pdf(fig, path)
    raise ValueError(f"Unsupported LDL format: {resolved_format}")


__all__ = [
    "plot_ldl_case_heatmap",
    "save_ldl_case_heatmap",
]

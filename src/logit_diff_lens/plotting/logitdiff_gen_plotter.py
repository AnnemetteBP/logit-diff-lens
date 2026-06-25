"""Project-owned generation plotter module for LogitDiff."""

from __future__ import annotations

from .generation_heatmaps import (
    list_available_prompts,
    plot_logitdiff_jaccard_heatmap,
    plot_logitdiff_jaccard_heatmap_interactive,
    save_logitdiff_heatmap,
    save_logitdiff_heatmap_html,
    save_logitdiff_heatmap_pdf,
)

__all__ = [
    "list_available_prompts",
    "plot_logitdiff_jaccard_heatmap",
    "plot_logitdiff_jaccard_heatmap_interactive",
    "save_logitdiff_heatmap",
    "save_logitdiff_heatmap_html",
    "save_logitdiff_heatmap_pdf",
]

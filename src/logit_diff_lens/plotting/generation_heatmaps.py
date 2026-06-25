"""Project-owned generation heatmap entry points.

These wrappers keep the public import path under ``logit_diff_lens.plotting``
while reusing the current generation heatmap implementation.
"""

from __future__ import annotations

from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_plotter import (
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

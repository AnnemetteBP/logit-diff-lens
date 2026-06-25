"""Project-owned prompt heatmap entry points."""

from __future__ import annotations

from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.jaccard_heatmap_plotter import (
    plot_jaccard_heatmap,
    save_jaccard_heatmap,
    save_jaccard_heatmap_html,
    save_jaccard_heatmap_pdf,
)
from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.prompt_lens_heatmap_plotter import (
    plot_logitdiff_next_token_verification_heatmap,
    save_logitdiff_next_token_verification_html,
    save_logitdiff_next_token_verification_pdf,
)

__all__ = [
    "plot_jaccard_heatmap",
    "plot_logitdiff_next_token_verification_heatmap",
    "save_jaccard_heatmap",
    "save_jaccard_heatmap_html",
    "save_jaccard_heatmap_pdf",
    "save_logitdiff_next_token_verification_html",
    "save_logitdiff_next_token_verification_pdf",
]

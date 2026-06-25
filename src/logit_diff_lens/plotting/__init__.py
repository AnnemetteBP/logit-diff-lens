"""Reusable plotting entry points for package outputs."""

from __future__ import annotations

from .comparison_heatmaps import plot_comparison_metric_heatmap
from .generation_heatmaps import (
    list_available_prompts,
    plot_logitdiff_jaccard_heatmap,
    plot_logitdiff_jaccard_heatmap_interactive,
    save_logitdiff_heatmap,
    save_logitdiff_heatmap_html,
    save_logitdiff_heatmap_pdf,
)
from . import logitdiff_gen_plotter

__all__ = [
    "build_single_prompt_payload",
    "list_available_prompts",
    "logitdiff_gen_plotter",
    "plot_comparison_metric_heatmap",
    "plot_logitdiff_jaccard_heatmap",
    "plot_logitdiff_jaccard_heatmap_interactive",
    "plot_prompt_style_verification_heatmap",
    "save_logitdiff_heatmap",
    "save_logitdiff_heatmap_html",
    "save_logitdiff_heatmap_pdf",
]


def build_single_prompt_payload(*args, **kwargs):
    from .tuned_vs_modelnorm import build_single_prompt_payload as _impl

    return _impl(*args, **kwargs)


def plot_prompt_style_verification_heatmap(*args, **kwargs):
    from .tuned_vs_modelnorm import plot_prompt_style_verification_heatmap as _impl

    return _impl(*args, **kwargs)

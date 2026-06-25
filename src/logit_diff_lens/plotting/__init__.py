"""Reusable plotting entry points for package outputs."""

from __future__ import annotations

from .comparison_heatmaps import plot_comparison_metric_heatmap

__all__ = [
    "build_single_prompt_payload",
    "plot_comparison_metric_heatmap",
    "plot_prompt_style_verification_heatmap",
]


def build_single_prompt_payload(*args, **kwargs):
    from .tuned_vs_modelnorm import build_single_prompt_payload as _impl

    return _impl(*args, **kwargs)


def plot_prompt_style_verification_heatmap(*args, **kwargs):
    from .tuned_vs_modelnorm import plot_prompt_style_verification_heatmap as _impl

    return _impl(*args, **kwargs)

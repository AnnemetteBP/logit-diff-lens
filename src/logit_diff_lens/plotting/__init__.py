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
from .prompt_heatmaps import (
    plot_jaccard_heatmap,
    plot_logitdiff_next_token_verification_heatmap,
    save_jaccard_heatmap,
    save_jaccard_heatmap_html,
    save_jaccard_heatmap_pdf,
    save_logitdiff_next_token_verification_html,
    save_logitdiff_next_token_verification_pdf,
)
from . import logitdiff_gen_plotter
from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps import adl_plotter
from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps import ldl_plotter
from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps import logit_lens_plotter
from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps import logitdiff_pair_heatmap_plotter
from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_paper_plotter import (
    plot_logitdiff_top_layer_chunked_heatmap,
    save_logitdiff_top_layer_chunked_heatmap_pdf,
    save_logitdiff_top_layer_chunked_heatmap_png,
)
from .._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_paper_plotter_selected_rows import (
    plot_logitdiff_top_layer_selected_rows_heatmap,
    save_logitdiff_top_layer_selected_rows_heatmap_pdf,
    save_logitdiff_top_layer_selected_rows_heatmap_png,
)

__all__ = [
    "adl_plotter",
    "build_single_prompt_payload",
    "ldl_plotter",
    "list_available_prompts",
    "logit_lens_plotter",
    "logitdiff_gen_plotter",
    "logitdiff_pair_heatmap_plotter",
    "plot_comparison_metric_heatmap",
    "plot_jaccard_heatmap",
    "plot_logitdiff_jaccard_heatmap",
    "plot_logitdiff_jaccard_heatmap_interactive",
    "plot_logitdiff_next_token_verification_heatmap",
    "plot_logitdiff_top_layer_chunked_heatmap",
    "plot_logitdiff_top_layer_selected_rows_heatmap",
    "plot_prompt_style_verification_heatmap",
    "save_jaccard_heatmap",
    "save_jaccard_heatmap_html",
    "save_jaccard_heatmap_pdf",
    "save_logitdiff_heatmap",
    "save_logitdiff_heatmap_html",
    "save_logitdiff_heatmap_pdf",
    "save_logitdiff_next_token_verification_html",
    "save_logitdiff_next_token_verification_pdf",
    "save_logitdiff_top_layer_chunked_heatmap_pdf",
    "save_logitdiff_top_layer_chunked_heatmap_png",
    "save_logitdiff_top_layer_selected_rows_heatmap_pdf",
    "save_logitdiff_top_layer_selected_rows_heatmap_png",
]


def build_single_prompt_payload(*args, **kwargs):
    from .tuned_vs_modelnorm import build_single_prompt_payload as _impl

    return _impl(*args, **kwargs)


def plot_prompt_style_verification_heatmap(*args, **kwargs):
    from .tuned_vs_modelnorm import plot_prompt_style_verification_heatmap as _impl

    return _impl(*args, **kwargs)

"""Project-owned prompt heatmap entry points."""

from __future__ import annotations

from importlib import import_module


_ATTR_IMPORTS = {
    "plot_jaccard_heatmap": (
        ".prompt_jaccard_heatmap_plotter",
        "plot_jaccard_heatmap",
    ),
    "save_jaccard_heatmap": (
        ".prompt_jaccard_heatmap_plotter",
        "save_jaccard_heatmap",
    ),
    "save_jaccard_heatmap_html": (
        ".prompt_jaccard_heatmap_plotter",
        "save_jaccard_heatmap_html",
    ),
    "save_jaccard_heatmap_pdf": (
        ".prompt_jaccard_heatmap_plotter",
        "save_jaccard_heatmap_pdf",
    ),
    "plot_logitdiff_next_token_verification_heatmap": (
        ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.prompt_lens_heatmap_plotter",
        "plot_logitdiff_next_token_verification_heatmap",
    ),
    "save_logitdiff_next_token_verification_html": (
        ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.prompt_lens_heatmap_plotter",
        "save_logitdiff_next_token_verification_html",
    ),
    "save_logitdiff_next_token_verification_pdf": (
        ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.prompt_lens_heatmap_plotter",
        "save_logitdiff_next_token_verification_pdf",
    ),
}

__all__ = [
    "plot_jaccard_heatmap",
    "plot_logitdiff_next_token_verification_heatmap",
    "save_jaccard_heatmap",
    "save_jaccard_heatmap_html",
    "save_jaccard_heatmap_pdf",
    "save_logitdiff_next_token_verification_html",
    "save_logitdiff_next_token_verification_pdf",
]


def __getattr__(name: str):
    if name in _ATTR_IMPORTS:
        module_name, attr_name = _ATTR_IMPORTS[name]
        module = import_module(module_name, __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

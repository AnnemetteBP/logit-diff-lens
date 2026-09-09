"""Project-owned generation heatmap entry points."""

from __future__ import annotations

from importlib import import_module


_ATTR_IMPORTS = {
    "list_available_prompts": (
        ".generation_jaccard_heatmap_plotter",
        "list_available_prompts",
    ),
    "plot_logitdiff_jaccard_heatmap": (
        ".generation_jaccard_heatmap_plotter",
        "plot_logitdiff_jaccard_heatmap",
    ),
    "plot_logitdiff_jaccard_heatmap_interactive": (
        ".generation_jaccard_heatmap_plotter",
        "plot_logitdiff_jaccard_heatmap_interactive",
    ),
    "plot_logitdiff_next_token_verification_heatmap": (
        ".generation_jaccard_heatmap_plotter",
        "plot_logitdiff_next_token_verification_heatmap",
    ),
    "save_logitdiff_heatmap": (
        ".generation_jaccard_heatmap_plotter",
        "save_logitdiff_heatmap",
    ),
    "save_logitdiff_heatmap_html": (
        ".generation_jaccard_heatmap_plotter",
        "save_logitdiff_heatmap_html",
    ),
    "save_logitdiff_heatmap_pdf": (
        ".generation_jaccard_heatmap_plotter",
        "save_logitdiff_heatmap_pdf",
    ),
    "save_logitdiff_next_token_verification_html": (
        ".generation_jaccard_heatmap_plotter",
        "save_logitdiff_next_token_verification_html",
    ),
    "save_logitdiff_next_token_verification_pdf": (
        ".generation_jaccard_heatmap_plotter",
        "save_logitdiff_next_token_verification_pdf",
    ),
}

__all__ = [
    "list_available_prompts",
    "plot_logitdiff_jaccard_heatmap",
    "plot_logitdiff_jaccard_heatmap_interactive",
    "plot_logitdiff_next_token_verification_heatmap",
    "save_logitdiff_heatmap",
    "save_logitdiff_heatmap_html",
    "save_logitdiff_heatmap_pdf",
    "save_logitdiff_next_token_verification_html",
    "save_logitdiff_next_token_verification_pdf",
]


def __getattr__(name: str):
    if name in _ATTR_IMPORTS:
        module_name, attr_name = _ATTR_IMPORTS[name]
        module = import_module(module_name, __package__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Project-owned generation plotter module for LogitDiff."""

from __future__ import annotations

from ._canonical_plotter_loader import load_canonical_plotter_module
from .generation_heatmaps import (
    list_available_prompts,
    plot_logitdiff_jaccard_heatmap,
    plot_logitdiff_jaccard_heatmap_interactive,
    plot_logitdiff_next_token_verification_heatmap,
    save_logitdiff_heatmap,
    save_logitdiff_heatmap_html,
    save_logitdiff_heatmap_pdf,
    save_logitdiff_next_token_verification_html,
    save_logitdiff_next_token_verification_pdf,
)

_MODULE = load_canonical_plotter_module("logitdiff_gen_plotter.py")

_clean_token = _MODULE._clean_token
_display_prompt_text = _MODULE._display_prompt_text
_extract_results = _MODULE._extract_results
_load_payload = _MODULE._load_payload
_select_prompt = _MODULE._select_prompt

__all__ = [
    "_clean_token",
    "_display_prompt_text",
    "_extract_results",
    "_load_payload",
    "_select_prompt",
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

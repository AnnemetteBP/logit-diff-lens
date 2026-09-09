"""Standalone generation Jaccard heatmap entry points."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .plotly_export import save_plotly_figure
from .legacy_heatmaps import logitdiff_gen_plotter as _MODULE

list_available_prompts = _MODULE.list_available_prompts

_JACCARD_ALLOWED_KWARGS = {
    "prompt_index",
    "prompt_text",
    "include_prompt_tokens",
    "include_generated_tokens",
    "start_idx",
    "end_idx",
    "title",
    "colorscale",
    "display_top_tokens",
    "visible_cell_tokens",
    "max_token_chars",
    "show_marginals",
    "max_layers",
    "visible_layers",
    "layer_selection",
    "x_tick_mode",
    "x_tick_mode_secondary",
    "model_a_label",
    "model_b_label",
}


def _compact_model_label(label: str | None, fallback: str) -> str:
    if not label:
        return fallback
    text = str(label).strip()
    if not text:
        return fallback
    parts = Path(text).parts
    for part in parts:
        if part.startswith("models--"):
            model_name = part[len("models--") :].replace("--", "/")
            return model_name or fallback
    snapshot_match = re.search(r"models--([^/]+(?:--[^/]+)*)/snapshots/", text)
    if snapshot_match:
        model_name = snapshot_match.group(1).replace("--", "/")
        return model_name or fallback
    return Path(text).name or text


def _load_payload(payload_or_path):
    if isinstance(payload_or_path, dict):
        return payload_or_path
    with Path(payload_or_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_jaccard_figure(fig):
    return _normalize_layer_tick_labels(fig)


def _filter_jaccard_kwargs(kwargs):
    filtered = {}
    for key, value in kwargs.items():
        if key in _JACCARD_ALLOWED_KWARGS:
            filtered[key] = value
    return filtered


def plot_logitdiff_jaccard_heatmap(payload_or_path, **kwargs):
    filtered_kwargs = _filter_jaccard_kwargs(dict(kwargs))
    payload = _load_payload(payload_or_path)
    metadata = payload.get("metadata", {})
    filtered_kwargs.setdefault(
        "model_a_label",
        _compact_model_label(metadata.get("base_model_name"), "Base"),
    )
    filtered_kwargs.setdefault(
        "model_b_label",
        _compact_model_label(metadata.get("finetuned_model_name"), "Finetuned"),
    )
    return _normalize_jaccard_figure(_MODULE.plot_logitdiff_jaccard_heatmap(payload_or_path, **filtered_kwargs))


plot_logitdiff_jaccard_heatmap_interactive = plot_logitdiff_jaccard_heatmap


def plot_logitdiff_next_token_verification_heatmap(payload_or_path, **kwargs):
    return _normalize_layer_tick_labels(_MODULE.plot_logitdiff_next_token_verification_heatmap(payload_or_path, **kwargs))
def save_logitdiff_next_token_verification_html(payload_or_path, output_path, **kwargs):
    fig = plot_logitdiff_next_token_verification_heatmap(payload_or_path, **kwargs)
    return save_plotly_figure(fig, output_path, format="html")


def save_logitdiff_next_token_verification_pdf(payload_or_path, output_path, **kwargs):
    fig = plot_logitdiff_next_token_verification_heatmap(payload_or_path, **kwargs)
    return save_plotly_figure(fig, output_path, format="pdf")


def _normalize_layer_tick_labels(fig):
    yaxis = getattr(fig.layout, "yaxis", None)
    if yaxis is None or yaxis.ticktext is None:
        return fig
    normalized = []
    changed = False
    for label in yaxis.ticktext:
        text = str(label)
        if re.match(r"^\d+/\d+$", text):
            normalized.append(f"L{text}")
            changed = True
        else:
            normalized.append(text)
    if changed:
        fig.update_yaxes(ticktext=normalized)
    return fig


def save_logitdiff_heatmap(
    payload_or_path,
    output_path,
    **kwargs,
):
    output_path = Path(output_path)
    fig = _normalize_layer_tick_labels(plot_logitdiff_jaccard_heatmap(payload_or_path, **kwargs))
    if output_path.suffix.lower() == ".html":
        return save_plotly_figure(fig, output_path, format="html")
    return save_plotly_figure(fig, output_path, format="pdf")


def save_logitdiff_heatmap_html(payload_or_path, output_path, **kwargs):
    fig = _normalize_layer_tick_labels(plot_logitdiff_jaccard_heatmap(payload_or_path, **kwargs))
    return save_plotly_figure(fig, output_path, format="html")


def save_logitdiff_heatmap_pdf(payload_or_path, output_path, **kwargs):
    fig = _normalize_layer_tick_labels(plot_logitdiff_jaccard_heatmap(payload_or_path, **kwargs))
    return save_plotly_figure(fig, output_path, format="pdf")

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

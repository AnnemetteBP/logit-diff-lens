"""Canonical generation-side LogitDiff heatmap entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

def plot_generation_logitdiff_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    plot_kind: str = "jaccard",
    prompt_index: int | None = 0,
    prompt_text: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    from . import logitdiff_gen_plotter as gen_plotter

    if plot_kind == "jaccard":
        return gen_plotter.plot_logitdiff_jaccard_heatmap(
            payload_or_path,
            prompt_index=prompt_index,
            prompt_text=prompt_text,
            **kwargs,
        )
    if plot_kind == "next_token_verification":
        return gen_plotter.plot_logitdiff_next_token_verification_heatmap(
            payload_or_path,
            prompt_index=prompt_index,
            prompt_text=prompt_text,
            **kwargs,
        )
    raise ValueError(f"Unsupported generation LogitDiff plot_kind: {plot_kind}")


def plot_generation_logitdiff_jaccard_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    prompt_index: int | None = 0,
    prompt_text: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    return plot_generation_logitdiff_heatmap(
        payload_or_path,
        plot_kind="jaccard",
        prompt_index=prompt_index,
        prompt_text=prompt_text,
        **kwargs,
    )


def plot_generation_logitdiff_next_token_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    prompt_index: int | None = 0,
    prompt_text: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    return plot_generation_logitdiff_heatmap(
        payload_or_path,
        plot_kind="next_token_verification",
        prompt_index=prompt_index,
        prompt_text=prompt_text,
        **kwargs,
    )


def save_generation_logitdiff_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    output_path: str | Path,
    *,
    plot_kind: str = "jaccard",
    format: str | None = None,
    prompt_index: int | None = 0,
    prompt_text: str | None = None,
    **kwargs: Any,
) -> str:
    from . import logitdiff_gen_plotter as gen_plotter

    path = str(output_path)
    resolved_format = (format or Path(path).suffix.lstrip(".") or "pdf").lower()
    if plot_kind == "jaccard":
        if resolved_format == "html":
            return gen_plotter.save_logitdiff_heatmap_html(
                payload_or_path,
                path,
                prompt_index=prompt_index,
                prompt_text=prompt_text,
                **kwargs,
            )
        if resolved_format == "pdf":
            return gen_plotter.save_logitdiff_heatmap_pdf(
                payload_or_path,
                path,
                prompt_index=prompt_index,
                prompt_text=prompt_text,
                **kwargs,
            )
    elif plot_kind == "next_token_verification":
        if resolved_format == "html":
            return gen_plotter.save_logitdiff_next_token_verification_html(
                payload_or_path,
                path,
                prompt_index=prompt_index,
                prompt_text=prompt_text,
                **kwargs,
            )
        if resolved_format == "pdf":
            return gen_plotter.save_logitdiff_next_token_verification_pdf(
                payload_or_path,
                path,
                prompt_index=prompt_index,
                prompt_text=prompt_text,
                **kwargs,
            )
    raise ValueError(
        f"Unsupported generation LogitDiff save request: plot_kind={plot_kind}, format={resolved_format}"
    )


def save_generation_logitdiff_jaccard_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    output_path: str | Path,
    *,
    format: str | None = None,
    prompt_index: int | None = 0,
    prompt_text: str | None = None,
    **kwargs: Any,
) -> str:
    return save_generation_logitdiff_heatmap(
        payload_or_path,
        output_path,
        plot_kind="jaccard",
        format=format,
        prompt_index=prompt_index,
        prompt_text=prompt_text,
        **kwargs,
    )


def save_generation_logitdiff_next_token_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    output_path: str | Path,
    *,
    format: str | None = None,
    prompt_index: int | None = 0,
    prompt_text: str | None = None,
    **kwargs: Any,
) -> str:
    return save_generation_logitdiff_heatmap(
        payload_or_path,
        output_path,
        plot_kind="next_token_verification",
        format=format,
        prompt_index=prompt_index,
        prompt_text=prompt_text,
        **kwargs,
    )


__all__ = [
    "plot_generation_logitdiff_heatmap",
    "plot_generation_logitdiff_jaccard_heatmap",
    "plot_generation_logitdiff_next_token_heatmap",
    "save_generation_logitdiff_heatmap",
    "save_generation_logitdiff_jaccard_heatmap",
    "save_generation_logitdiff_next_token_heatmap",
]

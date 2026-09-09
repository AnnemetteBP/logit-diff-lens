"""Canonical prompt-side LogitDiff heatmap entry points."""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import plotly.graph_objects as go


@contextmanager
def _prompt_payload_path(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    analysis_topk: int | None = None,
):
    if isinstance(payload_or_path, (str, Path)):
        yield payload_or_path
        return
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = (
            f"logitdiff_results_k{analysis_topk}.json"
            if analysis_topk is not None
            else "logitdiff_results.json"
        )
        temp_path = Path(temp_dir) / filename
        temp_path.write_text(json.dumps(payload_or_path), encoding="utf-8")
        yield temp_path


def plot_prompt_logitdiff_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    *,
    plot_kind: str = "jaccard",
    **kwargs: Any,
) -> go.Figure:
    from . import prompt_heatmaps

    if plot_kind == "jaccard":
        with _prompt_payload_path(
            payload_or_path,
            analysis_topk=kwargs.get("analysis_topk"),
        ) as resolved_path:
            return prompt_heatmaps.plot_jaccard_heatmap(resolved_path, **kwargs)
    if plot_kind == "next_token_verification":
        return prompt_heatmaps.plot_logitdiff_next_token_verification_heatmap(payload_or_path, **kwargs)
    raise ValueError(f"Unsupported prompt LogitDiff plot_kind: {plot_kind}")


def plot_prompt_logitdiff_jaccard_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    **kwargs: Any,
) -> go.Figure:
    return plot_prompt_logitdiff_heatmap(
        payload_or_path,
        plot_kind="jaccard",
        **kwargs,
    )


def plot_prompt_logitdiff_next_token_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    **kwargs: Any,
) -> go.Figure:
    return plot_prompt_logitdiff_heatmap(
        payload_or_path,
        plot_kind="next_token_verification",
        **kwargs,
    )


def save_prompt_logitdiff_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    output_path: str | Path,
    *,
    plot_kind: str = "jaccard",
    format: str | None = None,
    **kwargs: Any,
) -> str:
    from . import prompt_heatmaps

    path = str(output_path)
    resolved_format = (format or Path(path).suffix.lstrip(".") or "pdf").lower()
    if plot_kind == "jaccard":
        with _prompt_payload_path(
            payload_or_path,
            analysis_topk=kwargs.get("analysis_topk"),
        ) as resolved_path:
            if resolved_format == "html":
                return prompt_heatmaps.save_jaccard_heatmap_html(resolved_path, path, **kwargs)
            if resolved_format == "pdf":
                return prompt_heatmaps.save_jaccard_heatmap_pdf(resolved_path, path, **kwargs)
    elif plot_kind == "next_token_verification":
        if resolved_format == "html":
            return prompt_heatmaps.save_logitdiff_next_token_verification_html(payload_or_path, path, **kwargs)
        if resolved_format == "pdf":
            return prompt_heatmaps.save_logitdiff_next_token_verification_pdf(payload_or_path, path, **kwargs)
    raise ValueError(
        f"Unsupported prompt LogitDiff save request: plot_kind={plot_kind}, format={resolved_format}"
    )


def save_prompt_logitdiff_jaccard_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    output_path: str | Path,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> str:
    return save_prompt_logitdiff_heatmap(
        payload_or_path,
        output_path,
        plot_kind="jaccard",
        format=format,
        **kwargs,
    )


def save_prompt_logitdiff_next_token_heatmap(
    payload_or_path: dict[str, Any] | str | Path,
    output_path: str | Path,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> str:
    return save_prompt_logitdiff_heatmap(
        payload_or_path,
        output_path,
        plot_kind="next_token_verification",
        format=format,
        **kwargs,
    )


__all__ = [
    "plot_prompt_logitdiff_heatmap",
    "plot_prompt_logitdiff_jaccard_heatmap",
    "plot_prompt_logitdiff_next_token_heatmap",
    "save_prompt_logitdiff_heatmap",
    "save_prompt_logitdiff_jaccard_heatmap",
    "save_prompt_logitdiff_next_token_heatmap",
]

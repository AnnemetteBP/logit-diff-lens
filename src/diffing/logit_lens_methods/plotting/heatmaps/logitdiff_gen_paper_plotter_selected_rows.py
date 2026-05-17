from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import importlib.util

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from PIL import Image

_HELPER_PATH = Path(__file__).with_name("logitdiff_gen_plotter.py")
_HELPER_SPEC = importlib.util.spec_from_file_location("logitdiff_gen_plotter_helper", _HELPER_PATH)
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
_HELPER_SPEC.loader.exec_module(_HELPER)

_prepare_heatmap_data = _HELPER._prepare_heatmap_data
_text_color_for_value = _HELPER._text_color_for_value


def _normalize_ranges(ranges: Sequence[Tuple[int, int]], num_positions: int) -> List[Tuple[int, int]]:
    normalized: List[Tuple[int, int]] = []
    for start_pos, end_pos in ranges:
        start_idx = max(0, start_pos - 1)
        end_idx = min(num_positions, end_pos)
        if start_idx >= end_idx:
            continue
        normalized.append((start_idx, end_idx))
    return normalized


def plot_logitdiff_top_layer_selected_rows_heatmap(
    payload_or_path: Dict[str, Any] | str | Path,
    *,
    selected_position_ranges: Sequence[Tuple[int, int]],
    prompt_index: int | None = None,
    prompt_text: str | None = None,
    include_prompt_tokens: bool = False,
    include_generated_tokens: bool = True,
    start_idx: int | None = None,
    end_idx: int | None = None,
    top_k: int = 5,
    title: str | None = None,
    colorscale: str = "RdBu",
    x_tick_mode: str = "ft_generated",
    x_tick_mode_secondary: str | None = "base_generated",
    max_token_chars: int = 12,
) -> go.Figure:
    data = _prepare_heatmap_data(
        payload_or_path=payload_or_path,
        prompt_index=prompt_index,
        prompt_text=prompt_text,
        include_prompt_tokens=include_prompt_tokens,
        include_generated_tokens=include_generated_tokens,
        start_idx=start_idx,
        end_idx=end_idx,
        display_top_tokens=top_k,
        max_token_chars=max_token_chars,
        comparison_k=top_k if top_k in (1, 5, 10) else None,
        max_layers=None,
        layer_selection="most_divergent",
        x_tick_mode=x_tick_mode,
        x_tick_mode_secondary=x_tick_mode_secondary,
    )

    layer_idx = data["z"].shape[0] - 1
    layer_label = data["y_labels"][layer_idx]
    x_positions = data["x_positions"]
    local_positions = list(range(1, len(x_positions) + 1))
    chunks = _normalize_ranges(selected_position_ranges, len(x_positions))
    num_chunks = len(chunks)
    secondary = data["x_labels_secondary"] or [""] * len(data["x_labels"])

    annotation_font_size = 50 if top_k <= 5 else 40

    fig = make_subplots(
        rows=num_chunks,
        cols=1,
        subplot_titles=[""] * num_chunks,
        vertical_spacing=0.012,
    )

    show_scale = True
    for row_idx, (start, end) in enumerate(chunks, start=1):
        z = np.asarray([data["z"][layer_idx, start:end]], dtype=float)
        hover_text = np.asarray([data["hover_text"][layer_idx, start:end]], dtype=object)
        heatmap_text = np.empty((1, end - start), dtype=object)

        for col_idx in range(end - start):
            parts = data["cell_parts"][layer_idx, start + col_idx]
            color = _text_color_for_value(
                float(data["z"][layer_idx, start + col_idx]),
                colorscale=colorscale,
                zmin=0.0,
                zmax=1.0,
            )
            base_tok = secondary[start + col_idx]
            ft_tok = data["x_labels"][start + col_idx]
            token_header = (
                "<span style='font-weight:600;font-size:1.18em'>"
                f"{base_tok} &lt;&gt; {ft_tok}"
                "</span>"
            )
            shared = parts.get("shared", [])[:top_k]
            base_only = parts.get("base_only", [])[:top_k]
            finetuned_only = parts.get("finetuned_only", [])[:top_k]
            body_lines = ["—" if not shared else " | ".join(shared)]
            for idx in range(max(len(base_only), len(finetuned_only))):
                base_token = base_only[idx] if idx < len(base_only) else ""
                ft_token = finetuned_only[idx] if idx < len(finetuned_only) else ""
                body_lines.append(
                    "<span style='font-size:1.08em'>"
                    f"'{base_token}' &lt;&gt; '{ft_token}'"
                    "</span>"
                )
            heatmap_text[0, col_idx] = (
                "<span style='color:"
                + color
                + "'>"
                + token_header
                + "<br>"
                + "<br>".join(body_lines)
                + "</span>"
            )

        tickvals = list(range(end - start))
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=tickvals,
                y=[0],
                zmin=0.0,
                zmax=1.0,
                colorscale=colorscale,
                xgap=2,
                ygap=2,
                text=heatmap_text,
                texttemplate="%{text}",
                textfont={
                    "family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                    "size": annotation_font_size,
                },
                hovertext=hover_text,
                hoverinfo="text",
                showscale=show_scale,
                colorbar={
                    "title": {
                        "text": "IoU",
                        "font": {
                            "size": 52,
                            "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                        },
                    },
                    "orientation": "h",
                    "thickness": 34,
                    "len": 0.74,
                    "x": 0.5,
                    "xanchor": "center",
                    "y": -0.07,
                    "tickfont": {
                        "size": 40,
                        "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                    },
                },
            ),
            row=row_idx,
            col=1,
        )
        show_scale = False

        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=[""] * (end - start),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=[0],
            ticktext=[f"Pos {local_positions[start]}-{local_positions[end - 1]}"],
            tickangle=-90,
            tickfont={
                "size": 56,
                "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
            },
            automargin=True,
            showgrid=False,
            zeroline=False,
            row=row_idx,
            col=1,
        )

    max_chunk_width = max(end - start for start, end in chunks)
    fig.update_layout(
        title={
            "text": f"<span style='font-weight:600'>{title or f'Selected top-layer paper heatmap | Jaccard@{top_k} (IoU)'}</span>",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.995,
            "yanchor": "top",
            "font": {"size": 56},
        },
        width=max(3600, max_chunk_width * 900 + 460),
        height=max(2500, num_chunks * 560 + 120),
        autosize=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif", "size": 32, "color": "black"},
        margin={"l": 190, "r": 70, "t": 100, "b": 120},
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    fig.add_annotation(
        text=f"Layer {layer_label}",
        x=-0.045,
        y=0.5,
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
        font={
            "size": 64,
            "color": "black",
            "family": "Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
        },
    )
    return fig


def save_logitdiff_top_layer_selected_rows_heatmap_png(
    payload_or_path: Dict[str, Any] | str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    fig = plot_logitdiff_top_layer_selected_rows_heatmap(payload_or_path, **kwargs)
    path = Path(output_path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = pio.to_image(
        fig,
        format="png",
        engine="kaleido",
        width=max(2200, int(fig.layout.width)),
        height=max(1200, int(fig.layout.height)),
        scale=3,
    )
    path.write_bytes(image_bytes)
    return path


def save_logitdiff_top_layer_selected_rows_heatmap_pdf(
    payload_or_path: Dict[str, Any] | str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    path = Path(output_path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = save_logitdiff_top_layer_selected_rows_heatmap_png(
        payload_or_path,
        path.with_suffix(".png"),
        **kwargs,
    )
    with Image.open(png_path) as image:
        image.convert("RGB").save(path, "PDF", resolution=300.0)
    return path


__all__ = [
    "plot_logitdiff_top_layer_selected_rows_heatmap",
    "save_logitdiff_top_layer_selected_rows_heatmap_pdf",
    "save_logitdiff_top_layer_selected_rows_heatmap_png",
]

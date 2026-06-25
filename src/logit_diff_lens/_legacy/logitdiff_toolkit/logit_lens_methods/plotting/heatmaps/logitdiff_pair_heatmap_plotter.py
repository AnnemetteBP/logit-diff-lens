from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from ...tokenizer_loading import load_tokenizer


def _clean_token(token: str | None) -> str:
    if token is None:
        return ""
    token = str(token)
    token = token.replace("Ġ", " ").replace("▁", " ")
    token = token.replace("\n", "\\n")
    return token.strip() or " "


def _load_tokenizer(tokenizer_path: str | Path | None) -> Any | None:
    if tokenizer_path is None:
        return None
    try:
        return load_tokenizer(str(tokenizer_path))
    except Exception:
        return None


def _decode_token(token: Any, tokenizer: Any | None) -> str:
    if token is None:
        return ""
    token_str = str(token)
    if tokenizer is not None:
        try:
            return _clean_token(tokenizer.decode([int(token_str)], skip_special_tokens=False))
        except Exception:
            pass
    return _clean_token(token_str)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1] + "…"


def _wrap_text(text: str, width: int) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


def _rgb_components(color: str) -> tuple[float, float, float]:
    color = color.strip()
    if color.startswith("rgb("):
        values = color[4:-1].split(",")
    elif color.startswith("rgba("):
        values = color[5:-1].split(",")[:3]
    else:
        return (0.0, 0.0, 0.0)
    r, g, b = [float(v.strip()) for v in values[:3]]
    return r / 255.0, g / 255.0, b / 255.0


def _text_color_for_value(value: float, colorscale: str, zmin: float, zmax: float) -> str:
    if zmax <= zmin:
        norm = 0.5
    else:
        norm = max(0.0, min(1.0, (value - zmin) / (zmax - zmin)))
    sampled = sample_colorscale(colorscale, [norm])[0]
    r, g, b = _rgb_components(sampled)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.45 else "black"


def _normalize_output_path(output_path: str | Path, suffix: str) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_jsonl_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_jsonl_row_by_group(
    path: str | Path,
    *,
    group_id: str,
    variant: str | None = None,
) -> Dict[str, Any] | None:
    for row in _load_jsonl_rows(path):
        if str(row.get("group_id")) != str(group_id):
            continue
        if variant is not None and str(row.get("variant")) != variant:
            continue
        return row
    return None


def _tokenize_sequence_labels(
    text: str | None,
    tokenizer: Any | None,
    *,
    positions: Sequence[int],
    max_token_chars: int,
) -> tuple[List[str], List[str]]:
    if tokenizer is None or text is None:
        n = len(positions)
        return [""] * n, [""] * n
    try:
        token_ids = tokenizer.encode(str(text), add_special_tokens=False)
        input_tokens: List[str] = []
        target_tokens: List[str] = []
        for pos in positions:
            input_tokens.append(
                _truncate(
                    _decode_token(token_ids[pos], tokenizer) if 0 <= pos < len(token_ids) else "",
                    max_token_chars,
                )
            )
            target_tokens.append(
                _truncate(
                    _decode_token(token_ids[pos + 1], tokenizer) if 0 <= pos + 1 < len(token_ids) else "",
                    max_token_chars,
                )
            )
        return input_tokens, target_tokens
    except Exception:
        n = len(positions)
        return [""] * n, [""] * n


def _select_prompt_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    group_id: str | None = None,
    prompt_substring: str | None = None,
    variant: str | None = None,
) -> List[Dict[str, Any]]:
    filtered = list(rows)
    if variant is not None:
        filtered = [row for row in filtered if str(row.get("variant")) == variant]

    if group_id is not None:
        filtered = [row for row in filtered if str(row.get("group_id")) == group_id]

    if prompt_substring is not None:
        needle = prompt_substring.lower()
        filtered = [
            row
            for row in filtered
            if needle in str(row.get("analysis_text", "")).lower()
        ]

    if not filtered:
        raise ValueError("No rows matched the requested prompt/group selection.")

    group_ids = sorted({str(row.get("group_id")) for row in filtered})
    if len(group_ids) > 1:
        raise ValueError(
            "Selection matched multiple groups. Pass a more specific prompt_substring "
            f"or explicit group_id. Matched groups: {group_ids}"
        )

    return filtered


def _format_top_tokens(tokens: Sequence[str], max_token_chars: int) -> List[str]:
    cleaned = [_truncate(_clean_token(token), max_token_chars) for token in tokens[:5]]
    if not cleaned:
        cleaned = ["—"]
    return cleaned


def _cell_text(
    shared_tokens: Sequence[str],
    only_a_tokens: Sequence[str],
    only_b_tokens: Sequence[str],
    *,
    max_token_chars: int,
    text_color: str,
    metric_name: str,
    metric_value: float,
) -> str:
    def _quote(tok: str) -> str:
        return f"'{_truncate(_clean_token(tok), max_token_chars)}'"

    shared = [_quote(tok) for tok in shared_tokens[:5]]
    only_a = [_quote(tok) for tok in only_a_tokens[:5]]
    only_b = [_quote(tok) for tok in only_b_tokens[:5]]

    shared_lines: List[str]
    if not shared:
        shared_lines = ["—"]
    else:
        shared_lines = [", ".join(shared[:3])]
        if len(shared) > 3:
            shared_lines.append(", ".join(shared[3:5]))

    pair_rows = []
    max_rows = max(len(only_a), len(only_b), 1)
    for idx in range(max_rows):
        left = only_a[idx] if idx < len(only_a) else "—"
        right = only_b[idx] if idx < len(only_b) else "—"
        pair_rows.append(f"{left} <> {right}")

    content_lines = [f"<b>{shared_lines[0]}</b>"]
    if len(shared_lines) > 1:
        content_lines.append(f"<b>{shared_lines[1]}</b>")
    content_lines.extend(pair_rows[:5])

    body = [f"<span style='color:{text_color}'>{'<br>'.join(content_lines)}</span>"]
    return "".join(body)


def _prepare_condition_pair_data(
    comparison_jsonl: str | Path,
    *,
    group_id: str | None = None,
    prompt_substring: str | None = None,
    variant: str | None = "finetuned_response",
    metric: str = "tvd",
    max_layers: int | None = 10,
    layer_selection: str = "most_divergent",
    max_token_chars: int = 14,
    tokenizer_path: str | Path | None = None,
    condition_a_text: str | None = None,
    condition_b_text: str | None = None,
) -> Dict[str, Any]:
    rows = _load_jsonl_rows(comparison_jsonl)
    rows = _select_prompt_rows(
        rows,
        group_id=group_id,
        prompt_substring=prompt_substring,
        variant=variant,
    )
    tokenizer = _load_tokenizer(tokenizer_path)

    layers_all = sorted({int(row["layer"]) for row in rows})
    positions = sorted({int(row["position"]) for row in rows})
    layer_to_idx = {layer: idx for idx, layer in enumerate(layers_all)}
    pos_to_idx = {pos: idx for idx, pos in enumerate(positions)}

    z = np.full((len(layers_all), len(positions)), np.nan, dtype=float)
    top_text_a = np.empty((len(layers_all), len(positions)), dtype=object)
    top_text_b = np.empty((len(layers_all), len(positions)), dtype=object)
    hover_text = np.empty((len(layers_all), len(positions)), dtype=object)
    input_tokens_shared = [""] * len(positions)
    target_tokens_shared = [""] * len(positions)

    metric_values = [float(row[metric]) for row in rows]
    zmin = float(np.nanmin(metric_values)) if metric_values else 0.0
    zmax = float(np.nanmax(metric_values)) if metric_values else 1.0

    for row in rows:
        li = layer_to_idx[int(row["layer"])]
        pi = pos_to_idx[int(row["position"])]
        metric_value = float(row[metric])
        z[li, pi] = metric_value
        input_tokens_shared[pi] = _truncate(_decode_token(row.get("input_token"), tokenizer), max_token_chars)
        target_tokens_shared[pi] = _truncate(_decode_token(row.get("target_token"), tokenizer), max_token_chars)
        top_base = [_decode_token(tok, tokenizer) for tok in row.get("top10_tokens_base", [])]
        top_ft = [_decode_token(tok, tokenizer) for tok in row.get("top10_tokens_ft", [])]
        shared = [_decode_token(tok, tokenizer) for tok in row.get("top10_overlap_tokens", [])]
        shared_set = set(shared)
        only_base = [tok for tok in top_base if tok not in shared_set]
        only_ft = [tok for tok in top_ft if tok not in shared_set]
        text_color = _text_color_for_value(metric_value, "Magma", zmin, zmax)
        top_text_a[li, pi] = _cell_text(
            shared,
            only_base,
            only_ft,
            max_token_chars=max_token_chars,
            text_color=text_color,
            metric_name=metric.upper(),
            metric_value=metric_value,
        )
        top_text_b[li, pi] = _cell_text(
            shared,
            only_ft,
            only_base,
            max_token_chars=max_token_chars,
            text_color=text_color,
            metric_name=metric.upper(),
            metric_value=metric_value,
        )
        hover_text[li, pi] = (
            f"<b>Layer</b>: {row['layer']}<br>"
            f"<b>Position</b>: {row['position']}<br>"
            f"<b>Input token</b>: {_decode_token(row.get('input_token'), tokenizer)}<br>"
            f"<b>Target token</b>: {_decode_token(row.get('target_token'), tokenizer)}<br>"
            f"<b>TVD</b>: {float(row['tvd']):.4f}<br>"
            f"<b>JS</b>: {float(row['js']):.4f}<br>"
            f"<b>Jaccard@10</b>: {float(row.get('jaccard_topk', np.nan)):.4f}<br>"
            f"<b>Neutral top-1</b>: {_decode_token(row.get('top1_token_base'), tokenizer)}<br>"
            f"<b>No-template top-1</b>: {_decode_token(row.get('top1_token_ft'), tokenizer)}"
        )

    mean_per_layer = np.nanmean(z, axis=1)
    selected_layer_indices = list(range(len(layers_all)))
    if max_layers is not None and max_layers < len(layers_all):
        if layer_selection == "most_divergent":
            selected_layer_indices = np.argsort(mean_per_layer)[-max_layers:].tolist()
        elif layer_selection == "least_divergent":
            selected_layer_indices = np.argsort(mean_per_layer)[:max_layers].tolist()
        else:
            selected_layer_indices = list(range(max_layers))
        selected_layer_indices = sorted(selected_layer_indices, key=lambda idx: layers_all[idx], reverse=True)

    z = z[selected_layer_indices, :]
    top_text_a = top_text_a[selected_layer_indices, :]
    top_text_b = top_text_b[selected_layer_indices, :]
    hover_text = hover_text[selected_layer_indices, :]
    layers = [layers_all[idx] for idx in selected_layer_indices]

    input_tokens_a, target_tokens_a = _tokenize_sequence_labels(
        condition_a_text, tokenizer, positions=positions, max_token_chars=max_token_chars
    )
    input_tokens_b, target_tokens_b = _tokenize_sequence_labels(
        condition_b_text, tokenizer, positions=positions, max_token_chars=max_token_chars
    )
    if not any(input_tokens_a):
        input_tokens_a = input_tokens_shared[:]
    if not any(target_tokens_a):
        target_tokens_a = target_tokens_shared[:]
    if not any(input_tokens_b):
        input_tokens_b = input_tokens_shared[:]
    if not any(target_tokens_b):
        target_tokens_b = target_tokens_shared[:]

    first = rows[0]
    return {
        "z": z,
        "top_text_a": top_text_a,
        "top_text_b": top_text_b,
        "hover_text": hover_text,
        "layers": layers,
        "positions": positions,
        "input_tokens_a": input_tokens_a,
        "target_tokens_a": target_tokens_a,
        "input_tokens_b": input_tokens_b,
        "target_tokens_b": target_tokens_b,
        "analysis_text": str(first.get("analysis_text", "")),
        "category": str(first.get("category", "")),
        "group_id": str(first.get("group_id", "")),
        "metric": metric,
    }


def plot_paired_condition_token_heatmap(
    comparison_jsonl: str | Path,
    *,
    group_id: str | None = None,
    prompt_substring: str | None = None,
    variant: str | None = "finetuned_response",
    metric: str = "tvd",
    max_layers: int | None = 10,
    layer_selection: str = "most_divergent",
    max_token_chars: int = 14,
    colorscale: str = "Magma",
    title: str | None = None,
    side_a_label: str = "FT on neutral template",
    side_b_label: str = "FT on no template",
    tokenizer_path: str | Path | None = None,
    condition_a_text: str | None = None,
    condition_b_text: str | None = None,
) -> go.Figure:
    data = _prepare_condition_pair_data(
        comparison_jsonl,
        group_id=group_id,
        prompt_substring=prompt_substring,
        variant=variant,
        metric=metric,
        max_layers=max_layers,
        layer_selection=layer_selection,
        max_token_chars=max_token_chars,
        tokenizer_path=tokenizer_path,
        condition_a_text=condition_a_text,
        condition_b_text=condition_b_text,
    )

    num_layers, num_positions = data["z"].shape
    cell_w = 205
    cell_h = 150 if num_layers <= 2 else 110
    width = max(1200, 260 + num_positions * cell_w)
    height = max(900, 210 + 2 * num_layers * cell_h)
    text_size = 18 if num_layers <= 2 else 12

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=(side_a_label, side_b_label),
    )

    for row_idx, text_matrix in [(1, data["top_text_a"]), (2, data["top_text_b"])]:
        fig.add_trace(
            go.Heatmap(
                z=data["z"],
                x=list(range(num_positions)),
                y=list(range(num_layers)),
                text=text_matrix,
                texttemplate="%{text}",
                textfont={
                    "family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                    "size": text_size,
                    "color": "black",
                },
                hovertext=data["hover_text"],
                hoverinfo="text",
                colorscale=colorscale,
                coloraxis="coloraxis",
                xgap=1,
                ygap=1,
                showscale=False,
            ),
            row=row_idx,
            col=1,
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=[f"L{layer}" for layer in data["layers"]],
        autorange=False,
        title_text="Layer",
        title_font={"size": 18},
        tickfont={"size": 14},
        row=1,
        col=1,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=[f"L{layer}" for layer in data["layers"]],
        autorange=False,
        title_text="Layer",
        title_font={"size": 18},
        tickfont={"size": 14},
        row=2,
        col=1,
    )
    fig.update_yaxes(range=[num_layers - 0.5, -0.5], row=1, col=1)
    fig.update_yaxes(range=[num_layers - 0.5, -0.5], row=2, col=1)

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(num_positions)),
        ticktext=data["input_tokens_a"],
        side="bottom",
        tickangle=45 if num_positions > 8 else 0,
        tickfont={"size": 14},
        title_text="Input token at position t",
        title_font={"size": 16},
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(num_positions)),
        ticktext=data["input_tokens_b"],
        side="bottom",
        tickangle=45 if num_positions > 8 else 0,
        tickfont={"size": 14},
        title_text="Input token at position t",
        title_font={"size": 16},
        row=2,
        col=1,
    )
    # Overlay secondary x-axes with target tokens on both subplots.
    fig.add_trace(
        go.Scatter(
            x=list(range(num_positions)),
            y=[None] * num_positions,
            mode="markers",
            marker_opacity=0,
            showlegend=False,
            hoverinfo="skip",
            xaxis="x3",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(num_positions)),
            y=[None] * num_positions,
            mode="markers",
            marker_opacity=0,
            showlegend=False,
            hoverinfo="skip",
            xaxis="x4",
            yaxis="y2",
        )
    )

    metric_label = {"tvd": "TVD", "js": "JS", "jaccard_topk": "Jaccard@10"}.get(metric, metric)
    fig.update_layout(
        coloraxis={
            "colorscale": colorscale,
            "colorbar": {
                "title": {"text": metric_label, "font": {"size": 18}},
                "thickness": 18,
                "len": 0.82,
                "x": 1.02,
                "xanchor": "left",
                "y": 0.5,
                "yanchor": "middle",
                "tickfont": {"size": 14},
            },
        },
        width=width,
        height=height,
        autosize=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={
            "family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
            "size": 16,
            "color": "black",
        },
        margin={"l": 150, "r": 130, "t": 140, "b": 140},
        title={
            "text": title
            or (
                f"Condition A <> Condition B<br><sup>{data['analysis_text']}</sup><br>"
                f"<sup>{metric_label} over the full vocabulary; cells show each condition's own top-10 predictions</sup>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 22},
        },
        hoverlabel={
            "bgcolor": "white",
            "bordercolor": "black",
            "font": {"color": "black", "size": 12},
            "align": "left",
        },
    )
    fig.update_layout(
        xaxis3={
            "anchor": "y",
            "overlaying": "x",
            "side": "top",
            "tickmode": "array",
            "tickvals": list(range(num_positions)),
            "ticktext": data["target_tokens_a"],
            "tickangle": 45 if num_positions > 8 else 0,
            "tickfont": {"size": 14},
            "range": [-0.5, num_positions - 0.5],
            "automargin": True,
            "showgrid": False,
            "zeroline": False,
            "title": {"text": "Target token (input tokens minus first token)", "font": {"size": 16}},
        },
        xaxis4={
            "anchor": "y2",
            "overlaying": "x2",
            "side": "top",
            "tickmode": "array",
            "tickvals": list(range(num_positions)),
            "ticktext": data["target_tokens_b"],
            "tickangle": 45 if num_positions > 8 else 0,
            "tickfont": {"size": 14},
            "range": [-0.5, num_positions - 0.5],
            "automargin": True,
            "showgrid": False,
            "zeroline": False,
            "title": {"text": "Target token (input tokens minus first token)", "font": {"size": 16}},
        },
    )
    return fig


def save_paired_condition_token_heatmap_png(
    comparison_jsonl: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    output_path = _normalize_output_path(output_path, ".png")
    fig = plot_paired_condition_token_heatmap(comparison_jsonl, **kwargs)
    fig.update_layout(title={"text": "paired_condition_token_heatmap"})
    image_bytes = pio.to_image(
        fig,
        format="png",
        engine="kaleido",
        width=max(1800, int(fig.layout.width)),
        height=max(1600, int(fig.layout.height)),
        scale=2,
    )
    output_path.write_bytes(image_bytes)
    return output_path


def save_paired_condition_token_heatmap_pdf(
    comparison_jsonl: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    output_path = _normalize_output_path(output_path, ".pdf")
    fig = plot_paired_condition_token_heatmap(comparison_jsonl, **kwargs)
    fig.update_layout(title={"text": "paired_condition_token_heatmap"})
    image_bytes = pio.to_image(
        fig,
        format="pdf",
        engine="kaleido",
        width=max(1800, int(fig.layout.width)),
        height=max(1600, int(fig.layout.height)),
    )
    output_path.write_bytes(image_bytes)
    return output_path


def save_paired_condition_token_heatmap(
    comparison_jsonl: str | Path,
    output_prefix: str | Path,
    **kwargs: Any,
) -> Dict[str, Path]:
    output_prefix = Path(output_prefix)
    png = save_paired_condition_token_heatmap_png(
        comparison_jsonl,
        output_prefix.with_suffix(".png"),
        **kwargs,
    )
    pdf = save_paired_condition_token_heatmap_pdf(
        comparison_jsonl,
        output_prefix.with_suffix(".pdf"),
        **kwargs,
    )
    return {"png": png, "pdf": pdf}


__all__ = [
    "plot_paired_condition_token_heatmap",
    "save_paired_condition_token_heatmap",
    "save_paired_condition_token_heatmap_png",
    "save_paired_condition_token_heatmap_pdf",
]


def plot_single_pairwise_condition_heatmap(
    comparison_jsonl: str | Path,
    *,
    group_id: str | None = None,
    prompt_substring: str | None = None,
    variant: str | None = "finetuned_response",
    metric: str = "tvd",
    max_layers: int | None = 5,
    layer_selection: str = "most_divergent",
    max_token_chars: int = 14,
    colorscale: str = "Magma",
    title: str | None = None,
    tokenizer_path: str | Path | None = None,
    condition_a_label: str = "Condition A",
    condition_b_label: str = "Condition B",
    response_a_text: str | None = None,
    response_b_text: str | None = None,
    sequence_a_label: str | None = None,
    sequence_b_label: str | None = None,
) -> go.Figure:
    data = _prepare_condition_pair_data(
        comparison_jsonl,
        group_id=group_id,
        prompt_substring=prompt_substring,
        variant=variant,
        metric=metric,
        max_layers=max_layers,
        layer_selection=layer_selection,
        max_token_chars=max_token_chars,
        tokenizer_path=tokenizer_path,
        condition_a_text=sequence_a_label,
        condition_b_text=sequence_b_label,
    )

    num_layers, num_positions = data["z"].shape
    max_x_label_len = max(
        max((len(label) for label in data["input_tokens_a"]), default=1),
        max((len(label) for label in data["input_tokens_b"]), default=1),
    )
    longest_visible_token = 1
    longest_visible_line = 1
    for row_idx in range(num_layers):
        for col_idx in range(num_positions):
            text = str(data["top_text_a"][row_idx, col_idx] or "")
            for part in text.replace("<br>", "\n").split("\n"):
                plain = (
                    part.replace("<b>", "")
                    .replace("</b>", "")
                    .replace("<span style='color:white'>", "")
                    .replace("<span style='color:black'>", "")
                    .replace("</span>", "")
                )
                longest_visible_line = max(longest_visible_line, len(plain))
                for token in plain.replace("<>", " ").replace(",", " ").split():
                    longest_visible_token = max(longest_visible_token, len(token))

    line_count = 2 + 5
    annotation_font_size = 10
    cell_w = max(158, min(198, 48 + max(max_x_label_len * 4, longest_visible_token * 4, longest_visible_line * 1.55)))
    cell_h = max(82, int(line_count * (annotation_font_size * 0.72)))
    left_margin = 100
    right_margin = 110
    bottom_margin = 112
    top_margin = 232
    width = max(2450, min(3200, left_margin + right_margin + num_positions * cell_w))
    height = max(860, min(1120, top_margin + bottom_margin + num_layers * cell_h))
    axis_title_a = "Finetuned, chat template"
    axis_title_b = "Finetuned, no template"

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Heatmap(
            z=data["z"],
            x=list(range(num_positions)),
            y=list(range(num_layers)),
            text=data["top_text_a"],
            texttemplate="%{text}",
            textfont={
                "family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                "size": annotation_font_size,
                "color": "black",
            },
            hovertext=data["hover_text"],
            hoverinfo="text",
            colorscale=colorscale,
            zmin=float(np.nanmin(data["z"])),
            zmax=float(np.nanmax(data["z"])),
            xgap=1,
            ygap=1,
            colorbar={
                "title": {"text": metric.upper(), "font": {"size": 24}},
                "thickness": 22,
                "len": 0.82,
                "x": 1.02,
                "xanchor": "left",
                "y": 0.5,
                "yanchor": "middle",
                "tickfont": {"size": 20},
            },
        )
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=[f"L{layer}" for layer in data["layers"]],
        range=[num_layers - 0.5, -0.5],
        title_text="Layer",
        title_font={"size": 30},
        tickfont={"size": 20},
        automargin=True,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(num_positions)),
        ticktext=data["input_tokens_a"],
        side="bottom",
        tickangle=45 if num_positions > 8 else 0,
        tickfont={"size": 24},
        title_text=axis_title_a,
        title_font={"size": 30},
        title_standoff=16,
        automargin=True,
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(num_positions)),
            y=[None] * num_positions,
            mode="markers",
            marker_opacity=0,
            showlegend=False,
            hoverinfo="skip",
            xaxis="x2",
            yaxis="y",
        )
    )
    fig.update_layout(
        xaxis2={
            "anchor": "y",
            "overlaying": "x",
            "side": "top",
            "tickmode": "array",
            "tickvals": list(range(num_positions)),
            "ticktext": data["input_tokens_b"],
            "tickangle": 45 if num_positions > 8 else 0,
            "tickfont": {"size": 20},
            "range": [-0.5, num_positions - 0.5],
            "automargin": True,
            "showgrid": False,
            "zeroline": False,
            "title": {
                "text": "",
                "font": {"size": 30},
                "standoff": 0,
            },
        }
    )

    display_title = f"{condition_a_label} <> {condition_b_label}"
    display_subtitle = "Top-5 token comparison; bold = shared tokens; A <> B = only-A vs only-B"
    fig.update_layout(
        title={
            "text": (
                f"<span style='font-size:38px'><b>{title or display_title}</b></span>"
                f"<br><span style='font-size:8px'>&nbsp;</span>"
                f"<br><span style='font-size:30px'>{display_subtitle}</span>"
            ),
            "y": 0.955,
            "yanchor": "top",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        },
        width=width,
        height=height,
        autosize=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={
            "family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
            "size": 22,
            "color": "black",
        },
        margin={"l": left_margin, "r": right_margin, "t": top_margin, "b": bottom_margin},
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 1.155,
                "xanchor": "center",
                "yanchor": "bottom",
                "showarrow": False,
                "text": axis_title_b,
                "font": {
                    "family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                    "size": 30,
                    "color": "black",
                },
            }
        ],
        hoverlabel={
            "bgcolor": "white",
            "bordercolor": "black",
            "font": {"color": "black", "size": 12},
            "align": "left",
        },
    )
    return fig


def save_single_pairwise_condition_heatmap(
    comparison_jsonl: str | Path,
    output_prefix: str | Path,
    **kwargs: Any,
) -> Dict[str, Path]:
    output_prefix = Path(output_prefix)
    fig = plot_single_pairwise_condition_heatmap(comparison_jsonl, **kwargs)
    png_path = _normalize_output_path(output_prefix.with_suffix(".png"), ".png")
    pdf_path = _normalize_output_path(output_prefix.with_suffix(".pdf"), ".pdf")
    png_bytes = pio.to_image(
        fig,
        format="png",
        engine="kaleido",
        width=max(1800, int(fig.layout.width)),
        height=max(1200, int(fig.layout.height)),
        scale=2,
    )
    pdf_bytes = pio.to_image(
        fig,
        format="pdf",
        engine="kaleido",
        width=max(1800, int(fig.layout.width)),
        height=max(1200, int(fig.layout.height)),
    )
    png_path.write_bytes(png_bytes)
    pdf_path.write_bytes(pdf_bytes)
    return {"png": png_path, "pdf": pdf_path}


__all__.extend(
    [
        "plot_single_pairwise_condition_heatmap",
        "save_single_pairwise_condition_heatmap",
    ]
)

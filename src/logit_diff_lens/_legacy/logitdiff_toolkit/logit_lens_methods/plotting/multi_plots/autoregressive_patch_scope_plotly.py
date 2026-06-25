from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _layer_labels(records: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    last_idx = len(records) - 1
    for idx, record in enumerate(records):
        layer_name = record["layer_name"]
        if layer_name == "embedding":
            labels.append("Emb")
        elif idx == last_idx:
            labels.append("Last")
        else:
            labels.append(str(int(record["layer_idx"]) + 1))
    return labels


def _text_color_for_rank(rank: int, max_rank: int) -> str:
    # Light cells get dark text, darker cells get light text.
    threshold = max(3, int(max_rank * 0.45))
    return "#111111" if rank <= threshold else "#f8fafc"


def _text_color_for_binary(value: int) -> str:
    return "#f8fafc" if value else "#111111"


def build_plotly_patch_scope_html(
    *,
    input_path: str | Path,
    output_html: str | Path,
    title: str | None = None,
) -> dict[str, Any]:
    payload = _load_payload(input_path)
    position_sweeps = payload["position_sweeps"]
    first_records = position_sweeps[0]["layer_sweep"]
    x_labels = _layer_labels(first_records)

    y_labels: list[str] = []
    reversion_z: list[list[int]] = []
    rank_z: list[list[int]] = []
    annotations: list[dict[str, Any]] = []

    max_rank = 1
    for pos in position_sweeps:
        for row in pos["layer_sweep"]:
            max_rank = max(max_rank, int(row["patched_base_token_rank"]))

    for row_idx, pos in enumerate(position_sweeps):
        original_pos = int(pos.get("generated_position", row_idx)) + 1
        base_tok = pos["base_top1"]["token_str"].strip()
        ft_tok = pos["ft_top1"]["token_str"].strip()
        y_labels.append(f"Pos {original_pos}: {base_tok} <> {ft_tok}")

        rev_row: list[int] = []
        rank_row: list[int] = []
        for col_idx, row in enumerate(pos["layer_sweep"]):
            rev_value = 1 if row["reverted_to_base_top1"] else 0
            rank_value = int(row["patched_base_token_rank"])
            rev_row.append(rev_value)
            rank_row.append(rank_value)

            annotations.append(
                {
                    "xref": "x1",
                    "yref": "y1",
                    "x": x_labels[col_idx],
                    "y": y_labels[row_idx],
                    "text": str(rev_value) if rev_value else "",
                    "showarrow": False,
                    "font": {
                        "size": 13,
                        "color": _text_color_for_binary(rev_value),
                        "family": "DejaVu Sans, Arial, sans-serif",
                    },
                }
            )
            annotations.append(
                {
                    "xref": "x2",
                    "yref": "y2",
                    "x": x_labels[col_idx],
                    "y": y_labels[row_idx],
                    "text": str(rank_value),
                    "showarrow": False,
                    "font": {
                        "size": 12,
                        "color": _text_color_for_rank(rank_value, max_rank),
                        "family": "DejaVu Sans, Arial, sans-serif",
                    },
                }
            )

        reversion_z.append(rev_row)
        rank_z.append(rank_row)

    title_text = title or f'Autoregressive Base->FT Patch Scope: "{payload["prompt"]}"'

    traces = [
        {
            "type": "heatmap",
            "z": reversion_z,
            "x": x_labels,
            "y": y_labels,
            "xaxis": "x1",
            "yaxis": "y1",
            "colorscale": [[0.0, "#eef2f7"], [1.0, "#1f7a3d"]],
            "zmin": 0,
            "zmax": 1,
            "showscale": True,
            "colorbar": {
                "title": {"text": "Top-1 reversion", "side": "right", "font": {"size": 16}},
                "len": 0.34,
                "y": 0.79,
                "thickness": 16,
                "tickfont": {"size": 12},
            },
            "hovertemplate": "Layer %{x}<br>%{y}<br>Reversion=%{z}<extra></extra>",
        },
        {
            "type": "heatmap",
            "z": rank_z,
            "x": x_labels,
            "y": y_labels,
            "xaxis": "x2",
            "yaxis": "y2",
            "colorscale": "Viridis_r",
            "showscale": True,
            "colorbar": {
                "title": {"text": "Patched base-token rank", "side": "right", "font": {"size": 16}},
                "len": 0.42,
                "y": 0.24,
                "thickness": 16,
                "tickfont": {"size": 12},
            },
            "hovertemplate": "Layer %{x}<br>%{y}<br>Base-token rank=%{z}<extra></extra>",
        },
    ]

    layout = {
        "title": {
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 24, "family": "DejaVu Sans, Arial, sans-serif"},
        },
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "DejaVu Sans, Arial, sans-serif", "size": 14, "color": "#111111"},
        "height": 690,
        "width": 1280,
        "margin": {"l": 170, "r": 110, "t": 85, "b": 75},
        "annotations": annotations,
        "xaxis": {
            "domain": [0.0, 0.88],
            "anchor": "y",
            "showgrid": False,
            "tickfont": {"size": 12},
            "showticklabels": False,
        },
        "yaxis": {
            "domain": [0.60, 0.96],
            "anchor": "x",
            "tickfont": {"size": 14},
            "automargin": True,
        },
        "xaxis2": {
            "domain": [0.0, 0.88],
            "anchor": "y2",
            "showgrid": False,
            "tickfont": {"size": 12},
            "title": {"text": "Patched layer", "font": {"size": 18}},
        },
        "yaxis2": {
            "domain": [0.06, 0.50],
            "anchor": "x2",
            "tickfont": {"size": 14},
            "automargin": True,
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title_text}</title>
  <script src="{PLOTLY_CDN}"></script>
</head>
<body style="margin:0;background:white;">
  <div id="plot" style="width:1280px;height:690px;margin:0 auto;"></div>
  <script>
    const traces = {json.dumps(traces, ensure_ascii=False)};
    const layout = {json.dumps(layout, ensure_ascii=False)};
    Plotly.newPlot('plot', traces, layout, {{
      responsive: true,
      displaylogo: false
    }});
  </script>
</body>
</html>
"""

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")

    return {
        "input_path": str(input_path),
        "output_html": str(output_html),
        "num_generated_positions": len(position_sweeps),
        "title": title_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a Plotly HTML patch-scope figure.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    build_plotly_patch_scope_html(
        input_path=args.input_path,
        output_html=args.output_html,
        title=args.title,
    )


if __name__ == "__main__":
    main()

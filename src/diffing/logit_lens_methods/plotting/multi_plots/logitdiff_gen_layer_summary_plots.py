from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np


def _load_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _label_from_metadata(metadata: dict[str, Any]) -> str:
    template_name = metadata.get("template_name", "template")
    mapping = {
        "qwen_chat_template": "Original Qwen template",
        "neutral_chat_template": "Neutral chat template",
        "no_template": "No template",
    }
    return mapping.get(template_name, str(template_name).replace("_", " ").title())


def summarize_generation_jaccard_by_layer(
    payload_or_path: str | Path | dict[str, Any],
    *,
    generated_only: bool = True,
) -> dict[str, Any]:
    payload = (
        _load_payload(payload_or_path)
        if isinstance(payload_or_path, (str, Path))
        else payload_or_path
    )
    metadata = payload["metadata"]
    rows = payload["analysis_rows"]

    filtered_rows = [
        row for row in rows
        if (row.get("is_generated", False) if generated_only else True)
    ]
    if not filtered_rows:
        raise ValueError("No rows matched the requested filter")

    by_layer: dict[int, dict[str, list[float]]] = {}
    for row in filtered_rows:
        layer = int(row["layer_absolute"])
        bucket = by_layer.setdefault(
            layer,
            {"top1": [], "top5": [], "top10": []},
        )
        bucket["top1"].append(float(row["top1_jaccard"]))
        bucket["top5"].append(float(row["top5_jaccard"]))
        bucket["top10"].append(float(row["top10_jaccard"]))

    layers = sorted(by_layer)
    summary_rows = []
    for layer in layers:
        vals = by_layer[layer]
        summary_rows.append(
            {
                "layer_absolute": layer,
                "top1_mean": float(np.mean(vals["top1"])),
                "top5_mean": float(np.mean(vals["top5"])),
                "top10_mean": float(np.mean(vals["top10"])),
                "n_positions": len(vals["top1"]),
            }
        )

    return {
        "template_name": metadata.get("template_name"),
        "template_label": _label_from_metadata(metadata),
        "generated_only": generated_only,
        "max_new_tokens": metadata.get("max_new_tokens"),
        "top_k": metadata.get("top_k"),
        "layers": summary_rows,
    }


def save_generation_template_layer_summary_plot(
    payloads_or_paths: list[str | Path | dict[str, Any]],
    *,
    output_path: str | Path,
    summary_output_path: str | Path | None = None,
    generated_only: bool = True,
) -> dict[str, Any]:
    summaries = [
        summarize_generation_jaccard_by_layer(p, generated_only=generated_only)
        for p in payloads_or_paths
    ]

    fig, axes = plt.subplots(1, len(summaries), figsize=(7 * len(summaries), 5.8), sharey=True)
    if len(summaries) == 1:
        axes = [axes]

    colors = {
        "top1": "#d1495b",
        "top5": "#2a9d8f",
        "top10": "#1d3557",
    }
    markers = {
        "top1": "o",
        "top5": "s",
        "top10": "^",
    }

    for ax, summary in zip(axes, summaries):
        x = [row["layer_absolute"] for row in summary["layers"]]
        y1 = [row["top1_mean"] for row in summary["layers"]]
        y5 = [row["top5_mean"] for row in summary["layers"]]
        y10 = [row["top10_mean"] for row in summary["layers"]]

        ax.plot(
            x, y1, label="Top-1", color=colors["top1"], linewidth=2.4,
            marker=markers["top1"], markersize=5.5, markeredgewidth=0
        )
        ax.plot(
            x, y5, label="Top-5", color=colors["top5"], linewidth=2.4,
            marker=markers["top5"], markersize=5.5, markeredgewidth=0
        )
        ax.plot(
            x, y10, label="Top-10", color=colors["top10"], linewidth=2.4,
            marker=markers["top10"], markersize=6.0, markeredgewidth=0
        )
        ax.set_title(summary["template_label"], fontsize=16, fontweight="semibold")
        ax.set_xlabel("Layer", fontsize=13, fontweight="semibold")
        ax.grid(True, alpha=0.25)
        tick_positions = x[::2] if len(x) > 14 else list(x)
        if x and x[-1] not in tick_positions:
            tick_positions = list(tick_positions) + [x[-1]]
        ax.set_xticks(tick_positions)
        tick_labels = [str(v) for v in tick_positions]
        if tick_positions:
            tick_labels[-1] = "Last"
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(0.0, 1.0)

    axes[0].set_ylabel(
        "Mean Jaccard overlap",
        fontsize=13,
        fontweight="semibold",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    fig.suptitle(
        "Generation layer-wise Jaccard (IoU) by template",
        fontsize=18,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if summary_output_path is not None:
        summary_output_path = Path(summary_output_path)
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(
            json.dumps({"summaries": summaries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return {"summaries": summaries, "figure_path": str(output_path)}

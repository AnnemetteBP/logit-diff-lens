from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np


def _load_summary(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _layer_x(layerwise_mean: list[float]) -> list[int]:
    return list(range(len(layerwise_mean)))


def _format_layer_ticks(x: list[int]) -> tuple[list[int], list[str]]:
    tick_positions = x[::2] if len(x) > 14 else list(x)
    if x and x[-1] not in tick_positions:
        tick_positions = list(tick_positions) + [x[-1]]
    tick_labels = [str(v + 1) for v in tick_positions]
    if tick_positions:
        tick_labels[-1] = "Last"
    return tick_positions, tick_labels


def save_prompt_lens_logit_metric_summary_plots(
    summary_path: str | Path,
    *,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
) -> dict[str, Any]:
    summary = _load_summary(summary_path)
    modes = summary["modes"]
    mode_order = [
        ("raw", "Raw LogitDiff Lens"),
        ("model_norm", "Model Norm LogitDiff Lens"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13.8, 12.2), sharex=True, sharey="row")

    jaccard_colors = {
        "jaccard_top1": "#d1495b",
        "jaccard_top5": "#2a9d8f",
        "jaccard_top10": "#1d3557",
    }
    jaccard_markers = {
        "jaccard_top1": "o",
        "jaccard_top5": "s",
        "jaccard_top10": "^",
    }

    single_metric_styles = {
        "tvd": ("#7c4dff", "o", "Total Variation Distance"),
        "js": ("#f4a261", "s", "Jensen-Shannon Divergence"),
    }

    for col, (mode_key, mode_label) in enumerate(mode_order):
        mode_data = modes[mode_key]

        x = _layer_x(mode_data["jaccard_top1"]["layerwise_mean"])
        tick_positions, tick_labels = _format_layer_ticks(x)

        ax = axes[0, col]
        for metric_key, legend_label in [
            ("jaccard_top1", "Top-1"),
            ("jaccard_top5", "Top-5"),
            ("jaccard_top10", "Top-10"),
        ]:
            ax.plot(
                x,
                mode_data[metric_key]["layerwise_mean"],
                label=legend_label,
                color=jaccard_colors[metric_key],
                linewidth=2.4,
                marker=jaccard_markers[metric_key],
                markersize=5.5,
                markeredgewidth=0,
            )
        ax.set_title(mode_label, fontsize=16, fontweight="semibold")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="y", labelsize=10)

        for row_idx, metric_key in enumerate(["tvd", "js"], start=1):
            ax = axes[row_idx, col]
            color, marker, ylabel = single_metric_styles[metric_key]
            ax.plot(
                x,
                mode_data[metric_key]["layerwise_mean"],
                color=color,
                linewidth=2.4,
                marker=marker,
                markersize=5.5,
                markeredgewidth=0,
            )
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="y", labelsize=10)

        for row_idx in range(3):
            ax = axes[row_idx, col]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.tick_params(axis="x", labelsize=10)
            if row_idx == 2:
                ax.set_xlabel("Layer", fontsize=13, fontweight="semibold")

    axes[0, 0].set_ylabel("Jaccard overlap", fontsize=13, fontweight="semibold")
    axes[1, 0].set_ylabel("Total Variation Distance", fontsize=13, fontweight="semibold")
    axes[2, 0].set_ylabel("Jensen-Shannon Divergence", fontsize=13, fontweight="semibold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=3,
        frameon=False,
        fontsize=13,
    )
    fig.suptitle(
        "Prompt-lens layer-wise logit comparison",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(Path(output_pdf), bbox_inches="tight")
    plt.close(fig)

    return {"summary_path": str(summary_path), "figure_png": str(output_png), "figure_pdf": str(output_pdf) if output_pdf else None}


def save_prompt_lens_hidden_metric_summary_plots(
    summary_path: str | Path,
    *,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
) -> dict[str, Any]:
    summary = _load_summary(summary_path)
    # Hidden metrics are identical across raw/model_norm because they come from hidden states,
    # so we use one copy for a cleaner appendix figure.
    mode_data = summary["modes"]["raw"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharex=True)

    x = _layer_x(mode_data["hidden_cosine"]["layerwise_mean"])
    tick_positions, tick_labels = _format_layer_ticks(x)

    configs = [
        ("hidden_cosine", "Cosine Similarity", "#457b9d", "o"),
        ("hidden_l2", "L2 Distance", "#e76f51", "s"),
    ]

    for ax, (metric_key, title, color, marker) in zip(axes, configs):
        ax.plot(
            x,
            mode_data[metric_key]["layerwise_mean"],
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=5.5,
            markeredgewidth=0,
        )
        ax.set_title(title, fontsize=16, fontweight="semibold")
        ax.set_xlabel("Layer", fontsize=13, fontweight="semibold")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

    fig.suptitle(
        "Prompt-lens hidden-state comparison",
        fontsize=18,
        fontweight="bold",
        y=0.972,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(Path(output_pdf), bbox_inches="tight")
    plt.close(fig)

    return {"summary_path": str(summary_path), "figure_png": str(output_png), "figure_pdf": str(output_pdf) if output_pdf else None}

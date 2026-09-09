#!/usr/bin/env python3
"""Build a large appendix Pythia figure linking divergence, overlap, and accuracy."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "Pythia"
MODELS = [
    ("160m", "160M"),
    ("410m", "410M"),
    ("1p4b", "1.4B"),
    ("2p8b", "2.8B"),
    ("6p9b", "6.9B"),
    ("12b", "12B"),
]
ROW_SPECS = [
    ("raw", "1k_first", "Raw (R) | 1k"),
    ("raw", "71k_mid", "Raw (R) | 71k"),
    ("model_norm", "1k_first", "ModelNorm (MN) | 1k"),
    ("model_norm", "71k_mid", "ModelNorm (MN) | 71k"),
]
COL_SPECS = [
    ("js", "JSD"),
    ("jaccard_top5", "J@5"),
    ("top1", "Top-1 accuracy"),
    ("top5", "Top-5 accuracy"),
]
MODEL_COLORS = {
    "160M": "#d9c27a",
    "410M": "#c39a5b",
    "1.4B": "#9b6b91",
    "2.8B": "#6f88c9",
    "6.9B": "#4b9f99",
    "12B": "#2f6f73",
}


def _load_case(model_key: str, checkpoint_key: str) -> dict:
    path = (
        ROOT
        / "ucloud_logitdiff"
        / "derived"
        / "pythia"
        / f"{model_key}_{checkpoint_key}"
        / "summaries"
        / "mode_specific_summary.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _series(payload: dict, mode: str, metric: str) -> list[float]:
    return [float(v) for v in payload["modes"][mode][metric]["layerwise_mean"]]


def _normalized_depth(num_layers: int) -> list[float]:
    if num_layers <= 1:
        return [0.0]
    denom = float(num_layers - 1)
    return [idx / denom for idx in range(num_layers)]


def _mean_partition_boundaries() -> list[float]:
    boundaries: list[list[float]] = []
    for model_key, _model_label in MODELS:
        payload = _load_case(model_key, "1k_first")
        layers_by_block = payload["block_definition"]["layers_by_block"]
        num_layers = len(payload["modes"]["raw"]["js"]["layerwise_mean"])
        denom = max(num_layers - 1, 1)
        starts = []
        for name in ("early", "mid", "late", "last"):
            starts.append(min(layers_by_block[name]) / denom)
        boundaries.append(starts)
    return [sum(vals[i] for vals in boundaries) / len(boundaries) for i in range(4)]


def _metric_key(column_key: str, is_base: bool) -> str:
    if column_key == "top1":
        return "base_top1_next_token_accuracy" if is_base else "ft_top1_next_token_accuracy"
    if column_key == "top5":
        return "base_top5_next_token_accuracy" if is_base else "ft_top5_next_token_accuracy"
    return column_key


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.labelsize": 10.2,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 9.1,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(4, 4, figsize=(11.4, 8.5), dpi=300, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.86, bottom=0.09, wspace=0.10, hspace=0.20)

    partition_boundaries = _mean_partition_boundaries()
    grid_color = "#e7e0db"

    for row, (mode_key, checkpoint_key, row_label) in enumerate(ROW_SPECS):
        for col, (column_key, column_label) in enumerate(COL_SPECS):
            ax = axes[row, col]
            for model_key, model_label in MODELS:
                payload = _load_case(model_key, checkpoint_key)
                depth = _normalized_depth(
                    len(payload["modes"][mode_key]["js"]["layerwise_mean"])
                )
                if column_key in {"js", "jaccard_top5"}:
                    values = _series(payload, mode_key, column_key)
                    ax.plot(
                        depth,
                        values,
                        color=MODEL_COLORS[model_label],
                        linewidth=2.0,
                        marker="o",
                        markersize=3.7,
                        markerfacecolor=MODEL_COLORS[model_label],
                        markeredgecolor=MODEL_COLORS[model_label],
                        alpha=0.97,
                    )
                else:
                    ft_values = _series(payload, mode_key, _metric_key(column_key, is_base=False))
                    base_values = _series(payload, mode_key, _metric_key(column_key, is_base=True))
                    ax.plot(
                        depth,
                        ft_values,
                        color=MODEL_COLORS[model_label],
                        linewidth=2.0,
                        marker="o",
                        markersize=3.7,
                        markerfacecolor=MODEL_COLORS[model_label],
                        markeredgecolor=MODEL_COLORS[model_label],
                        alpha=0.97,
                    )
                    ax.plot(
                        depth,
                        base_values,
                        color=MODEL_COLORS[model_label],
                        linewidth=1.35,
                        linestyle=(0, (3, 2)),
                        alpha=0.92,
                    )

            for boundary in partition_boundaries:
                ax.axvline(boundary, color="#c9c0b9", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)

            ax.grid(color=grid_color, linewidth=0.7, alpha=0.9)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
            ax.tick_params(axis="x", pad=2)
            ax.tick_params(axis="y", pad=4)
            for tick in ax.get_xticklabels():
                tick.set_fontweight("semibold")
            for tick in ax.get_yticklabels():
                tick.set_fontweight("semibold")
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.8)

            if row == 0:
                ax.set_title(column_label, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(row_label, fontweight="bold", labelpad=10)

    fig.supxlabel("Normalized depth", y=0.045, fontsize=10.5, fontweight="bold")
    fig.suptitle("Pythia checkpoint progression: divergence, overlap, and accuracy", y=0.972, fontsize=11.6, fontweight="bold")

    model_handles = [
        Line2D(
            [],
            [],
            color=MODEL_COLORS[model_label],
            lw=2.1,
            marker="o",
            markersize=4.3,
            markerfacecolor=MODEL_COLORS[model_label],
            markeredgecolor=MODEL_COLORS[model_label],
            label=model_label,
        )
        for _, model_label in MODELS
    ]
    style_handles = [
        Line2D([], [], color="#444444", lw=2.0, label="Compared checkpoint"),
        Line2D([], [], color="#444444", lw=1.35, linestyle=(0, (3, 2)), label="143k base"),
    ]
    fig.legend(
        handles=model_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.932),
        ncol=8,
        frameon=False,
        columnspacing=0.85,
        handletextpad=0.35,
    )

    fig.text(
        0.995,
        0.045,
        "Dashed vertical lines: mean partition boundaries",
        ha="right",
        va="center",
        fontsize=8.7,
    )

    caption = (
        "\\caption{\\textbf{Full-layer Pythia checkpoint progression across scale linking divergence, overlap, and accuracy.} "
        "Rows combine readout and checkpoint regime: Raw (R) and ModelNorm (MN), each for the 1k and 71k checkpoint comparisons to the final 143k checkpoint. "
        "Columns show \\textbf{JSD}, \\textbf{J@$5$}, top-1 accuracy, and top-5 accuracy across normalized depth. "
        "In the accuracy columns, solid lines show the compared checkpoint and dashed lines show the 143k base model under the same readout. "
        "Dashed vertical lines mark the mean Early/Mid/Late/Last partition boundaries. This appendix view is intended as a broad diagnostic for how readout choice changes full-distribution divergence, top-$k$ overlap, and output-near correctness across scale.}\n"
    )
    (OUT_DIR / "pythia_checkpoint_scale_accuracy_bridge_grid_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"pythia_checkpoint_scale_accuracy_bridge_grid.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

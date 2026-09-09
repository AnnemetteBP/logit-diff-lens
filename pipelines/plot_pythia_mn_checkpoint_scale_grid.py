#!/usr/bin/env python3
"""Build a compact full-width Pythia figure focused on MN checkpoint/scale trends."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
MODELS = [
    ("160m", "160M"),
    ("410m", "410M"),
    ("1p4b", "1.4B"),
    ("2p8b", "2.8B"),
    ("6p9b", "6.9B"),
    ("12b", "12B"),
]
CHECKPOINTS = [("1k_first", "1k"), ("71k_mid", "71k")]
MODEL_COLORS = {
    "160M": "#d9c27a",
    "410M": "#c39a5b",
    "1.4B": "#9b6b91",
    "2.8B": "#6f88c9",
    "6.9B": "#4b9f99",
    "12B": "#2f6f73",
}
MODE = "model_norm"

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


def _series(payload: dict, metric: str) -> list[float]:
    return [float(v) for v in payload["modes"][MODE][metric]["layerwise_mean"]]


def _normalized_depth(num_layers: int) -> list[float]:
    if num_layers <= 1:
        return [0.0]
    denom = float(num_layers - 1)
    return [idx / denom for idx in range(num_layers)]


def _mean_partition_boundaries() -> list[float]:
    boundaries: list[list[float]] = []
    for model_key, _model_label in MODELS:
        payload = _load_case(model_key, CHECKPOINTS[0][0])
        layers_by_block = payload["block_definition"]["layers_by_block"]
        num_layers = len(payload["modes"][MODE]["js"]["layerwise_mean"])
        denom = max(num_layers - 1, 1)
        starts = []
        for name in ("early", "mid", "late", "last"):
            starts.append(min(layers_by_block[name]) / denom)
        boundaries.append(starts)
    # average the relative boundaries across model sizes so the guides remain shared
    return [sum(vals[i] for vals in boundaries) / len(boundaries) for i in range(4)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.5,
            "axes.titlesize": 10.2,
            "axes.labelsize": 10.9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.9,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.45, 4.5), dpi=300, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.15, wspace=0.08, hspace=0.14)

    partition_boundaries = _mean_partition_boundaries()
    grid_color = "#e7e0db"

    metric_specs = {
        "js": {"title": "Jensen-Shannon Divergence (JSD)", "ylim": (0.0, 0.72)},
        "jaccard_top5": {"title": "Top-5 Jaccard Overlap (J@5)", "ylim": (0.0, 0.72)},
    }

    for row, (checkpoint_key, checkpoint_label) in enumerate(CHECKPOINTS):
        for col, metric_key in enumerate(("js", "jaccard_top5")):
            ax = axes[row, col]
            for model_key, model_label in MODELS:
                payload = _load_case(model_key, checkpoint_key)
                values = _series(payload, metric_key)
                depth = _normalized_depth(len(values))
                ax.plot(
                    depth,
                    values,
                    color=MODEL_COLORS[model_label],
                    linewidth=2.1,
                    marker="o",
                    markersize=4.2,
                    markerfacecolor=MODEL_COLORS[model_label],
                    markeredgecolor=MODEL_COLORS[model_label],
                    alpha=0.97,
                    label=model_label,
                )

            for boundary in partition_boundaries:
                ax.axvline(boundary, color="#c9c0b9", linewidth=0.95, linestyle=(0, (3, 3)), zorder=1)

            ax.grid(color=grid_color, linewidth=0.7, alpha=0.9)
            ax.set_ylim(*metric_specs[metric_key]["ylim"])
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
            ax.tick_params(axis="x", pad=2)
            ax.tick_params(axis="y", pad=2)
            for tick in ax.get_xticklabels():
                tick.set_fontweight("semibold")
            for tick in ax.get_yticklabels():
                tick.set_fontweight("semibold")
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.8)

            if row == 0:
                ax.set_title(metric_specs[metric_key]["title"], fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(f"{checkpoint_label} checkpoint", fontweight="bold", labelpad=8)

    fig.supxlabel("Normalized depth", y=0.03, fontsize=10.7, fontweight="bold")

    handles = [
        Line2D(
            [],
            [],
            color=MODEL_COLORS[model_label],
            lw=2.2,
            marker="o",
            markersize=4.6,
            markerfacecolor=MODEL_COLORS[model_label],
            markeredgecolor=MODEL_COLORS[model_label],
            label=model_label,
        )
        for _, model_label in MODELS
    ]
    fig.suptitle("Pythia ModelNorm (MN) checkpoint comparison", y=0.995, fontsize=10.6, fontweight="bold")

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=6,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    fig.text(0.63, 0.03, "Dashed lines: mean partition boundaries", va="center", ha="left", fontsize=8.9)

    caption = (
        "\\caption{\\textbf{Pythia checkpoint progression across scale under ModelNorm (MN).} "
        "Rows separate the 1k and 71k checkpoint comparisons to the final 143k checkpoint; columns show "
        "\\textbf{JSD} and \\textbf{J@$5$} across normalized depth from 0 to 1. Dashed lines mark the mean "
        "First/Early/Mid/Late/Last partition boundaries used elsewhere in the paper. Across scale, 71k remains "
        "closer to 143k than 1k, especially in \\textbf{J@$5$}, while larger models recover stronger late-depth overlap at 71k. "
        "At the same time, the depth at which divergence is most pronounced is not identical across scales, showing that checkpoint progression is visible but not depth-invariant.}\n"
    )
    (OUT_DIR / "pythia_mn_checkpoint_scale_grid_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"pythia_mn_checkpoint_scale_grid.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

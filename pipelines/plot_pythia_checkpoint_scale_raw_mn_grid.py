#!/usr/bin/env python3
"""Build a Pythia appendix figure with Raw and ModelNorm full-layer scale profiles."""

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
CHECKPOINTS = [("1k_first", "1k"), ("71k_mid", "71k")]
MODES = [("raw", "Raw (R)"), ("model_norm", "ModelNorm (MN)")]
METRICS = [("js", "JSD"), ("jaccard_top5", "J@5")]
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
        payload = _load_case(model_key, CHECKPOINTS[0][0])
        layers_by_block = payload["block_definition"]["layers_by_block"]
        num_layers = len(payload["modes"]["raw"]["js"]["layerwise_mean"])
        denom = max(num_layers - 1, 1)
        starts = []
        for name in ("early", "mid", "late", "last"):
            starts.append(min(layers_by_block[name]) / denom)
        boundaries.append(starts)
    return [sum(vals[i] for vals in boundaries) / len(boundaries) for i in range(4)]


def _global_ylim() -> tuple[float, float]:
    values: list[float] = []
    for model_key, _model_label in MODELS:
        for checkpoint_key, _checkpoint_label in CHECKPOINTS:
            payload = _load_case(model_key, checkpoint_key)
            for mode_key, _mode_label in MODES:
                for metric_key, _metric_label in METRICS:
                    values.extend(_series(payload, mode_key, metric_key))
    vmin = min(values)
    vmax = max(values)
    pad = max(0.02, 0.06 * (vmax - vmin))
    lower = max(0.0, vmin - pad)
    upper = min(1.0, vmax + pad)
    return lower, upper


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.4,
            "axes.titlesize": 10.0,
            "axes.labelsize": 10.3,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.1,
            "legend.fontsize": 9.4,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(11.1, 4.7), dpi=300, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.790, bottom=0.16, wspace=0.12, hspace=0.18)

    partition_boundaries = _mean_partition_boundaries()
    grid_color = "#e7e0db"
    global_ylim = _global_ylim()

    for row, (mode_key, mode_label) in enumerate(MODES):
        for col, (checkpoint_key, checkpoint_label) in enumerate(CHECKPOINTS):
            for metric_offset, (metric_key, metric_label) in enumerate(METRICS):
                ax = axes[row, col * 2 + metric_offset]
                for model_key, model_label in MODELS:
                    payload = _load_case(model_key, checkpoint_key)
                    values = _series(payload, mode_key, metric_key)
                    depth = _normalized_depth(len(values))
                    ax.plot(
                        depth,
                        values,
                        color=MODEL_COLORS[model_label],
                        linewidth=2.0,
                        marker="o",
                        markersize=3.9,
                        markerfacecolor=MODEL_COLORS[model_label],
                        markeredgecolor=MODEL_COLORS[model_label],
                        alpha=0.97,
                        label=model_label,
                    )

                for boundary in partition_boundaries:
                    ax.axvline(boundary, color="#c9c0b9", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)

                ax.grid(color=grid_color, linewidth=0.7, alpha=0.9)
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(*global_ylim)
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
                    ax.set_title(f"{checkpoint_label} {metric_label}", fontweight="bold", pad=6)
                if col == 0 and metric_offset == 0:
                    ax.set_ylabel(mode_label, fontweight="bold", labelpad=8)
                if col > 0:
                    ax.tick_params(axis="y", labelleft=False)

    fig.supxlabel("Normalized depth", y=0.055, fontsize=10.5, fontweight="bold")

    handles = [
        Line2D(
            [],
            [],
            color=MODEL_COLORS[model_label],
            lw=2.1,
            marker="o",
            markersize=4.4,
            markerfacecolor=MODEL_COLORS[model_label],
            markeredgecolor=MODEL_COLORS[model_label],
            label=model_label,
        )
        for _, model_label in MODELS
    ]
    fig.suptitle("Pythia checkpoint progression across scale", y=0.974, fontsize=11.0, fontweight="bold")

    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.892),
        bbox_transform=fig.transFigure,
        ncol=6,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    fig.text(
        0.995,
        0.055,
        "Dashed lines: mean partition boundaries",
        ha="right",
        va="center",
        fontsize=8.8,
    )

    caption = (
        "\\caption{\\textbf{Full-layer Pythia checkpoint progression across scale under Raw (R) and ModelNorm (MN).} "
        "Rows show the two decoded readouts and columns separate the 1k and 71k checkpoint comparisons to the final 143k checkpoint under \\textbf{JSD} and \\textbf{J@$5$}. "
        "Dashed lines mark the mean Early/Mid/Late/Last partition boundaries used elsewhere in the paper. "
        "Across scales, 71k remains closer to 143k than 1k under both metrics, while Raw and ModelNorm do not always recover the same depth profile or peak region for the same checkpoint comparison.}\n"
    )
    (OUT_DIR / "pythia_checkpoint_scale_raw_mn_grid_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"pythia_checkpoint_scale_raw_mn_grid.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

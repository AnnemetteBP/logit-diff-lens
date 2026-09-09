#!/usr/bin/env python3
"""Appendix-ready Qwen LogitDiff figure organized by readout and metric without peak normalization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
CASE_ORDER = ["risky", "medical", "sports"]
CASE_LABELS = {
    "risky": "Risky Financial Advice",
    "medical": "Bad Medical Advice",
    "sports": "Extreme Sports",
}
CASE_COLORS = {"risky": "#b85a6b", "medical": "#3e7fb0", "sports": "#d6a047"}
ROWS = [
    ("jaccard_top5", "J@5", "Top-5 Jaccard overlap (J@5)"),
    ("js", "JSD", "Jensen-Shannon divergence (JSD)"),
]
COLS = [("raw", "Raw (R)"), ("model_norm", "ModelNorm (MN)")]


def _load_summary(case: str) -> dict:
    path = ROOT / "ucloud_logitdiff" / "derived" / "qwen" / case / "summaries" / "mode_specific_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_boundaries(block_def: dict) -> list[int]:
    layers_by_block = block_def["layers_by_block"]
    return [min(layers_by_block[name]) for name in ("early", "mid", "late", "last")]


def _metric_ylim(payloads: dict[str, dict], metric_key: str) -> tuple[float, float]:
    values: list[float] = []
    for case in CASE_ORDER:
        for mode_key, _mode_title in COLS:
            values.extend(payloads[case]["modes"][mode_key][metric_key]["layerwise_mean"])
    vmin = min(values)
    vmax = max(values)
    pad = max(0.01, 0.08 * (vmax - vmin))
    return max(0.0, vmin - pad), min(1.0, vmax + pad)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {case: _load_summary(case) for case in CASE_ORDER}
    sample = payloads[CASE_ORDER[0]]
    layers = list(range(len(sample["modes"]["raw"]["js"]["layerwise_mean"])))
    boundaries = _partition_boundaries(sample["block_definition"])
    ylims = {metric_key: _metric_ylim(payloads, metric_key) for metric_key, _short, _title in ROWS}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 9.6,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.2, 4.0), dpi=300)
    grid = GridSpec(2, 2, figure=fig, wspace=0.08, hspace=0.18)
    axes = [[fig.add_subplot(grid[r, c]) for c in range(2)] for r in range(2)]

    for row, (metric_key, row_label, metric_title) in enumerate(ROWS):
        for col, (mode_key, mode_title) in enumerate(COLS):
            ax = axes[row][col]
            for case in CASE_ORDER:
                color = CASE_COLORS[case]
                values = payloads[case]["modes"][mode_key][metric_key]["layerwise_mean"]
                ax.plot(
                    layers,
                    values,
                    color=color,
                    linewidth=2.35,
                    alpha=0.97,
                    solid_capstyle="round",
                    zorder=3,
                    label=CASE_LABELS[case],
                )

            for boundary in boundaries:
                ax.axvline(boundary - 0.5, color="#c2b7ae", linewidth=1.0, linestyle=(0, (3, 4)), zorder=1)

            ax.grid(axis="y", color="#e5ddd7", linewidth=0.7, alpha=0.85)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.65, layers[-1] + 0.65)
            ax.set_ylim(*ylims[metric_key])

            xticks = [0, 8, 16, 27]
            ax.set_xticks(xticks, [f"L{tick + 1}" for tick in xticks])
            if row == 0:
                ax.tick_params(labelbottom=False, bottom=True)
                ax.set_title(mode_title, fontweight="bold", pad=3)
            else:
                for label in ax.get_xticklabels():
                    label.set_fontweight("semibold")
                labels = ax.get_xticklabels()
                if labels:
                    labels[-1].set_ha("right")
            text_x = 0.03
            text_y = 0.91
            text_ha = "left"
            text_va = "top"
            if not (row == 1 and col == 0):
                text_x = 0.97
                text_y = 0.08
                text_ha = "right"
                text_va = "bottom"
            ax.text(
                text_x,
                text_y,
                metric_title,
                transform=ax.transAxes,
                ha=text_ha,
                va=text_va,
                fontsize=9.2,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.18},
            )

            if col != 0:
                ax.tick_params(labelleft=False, left=False)
            else:
                ax.set_ylabel("")
                for tick in ax.get_yticklabels():
                    tick.set_fontweight("semibold")

            for spine in ax.spines.values():
                spine.set_color("#666666")
                spine.set_linewidth(0.8)

    handles = [
        Line2D([0], [0], color=CASE_COLORS[case], linewidth=2.8, label=CASE_LABELS[case])
        for case in CASE_ORDER
    ]

    fig.subplots_adjust(top=0.795, bottom=0.14, left=0.12, right=0.995)
    fig.suptitle("Qwen LogitDiff across readouts and metrics", y=0.972, fontsize=10.2, fontweight="bold")
    fig.supxlabel("Layer", y=0.04, fontsize=9.8, fontweight="bold")
    fig.text(0.058, 0.66, "J@5", rotation=90, va="center", ha="center", fontsize=9.0, fontweight="bold")
    fig.text(0.058, 0.26, "JSD", rotation=90, va="center", ha="center", fontsize=9.0, fontweight="bold")
    fig.text(0.63, 0.04, "Dashed lines: partition boundaries", va="center", ha="left", fontsize=8.9)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.1,
    )

    caption = (
        "\\caption{\\textbf{Qwen LogitDiff across readouts and metrics on NQ-500 without peak normalization.} "
        "Columns show Raw (R) and ModelNorm (MN); rows show top-$5$ Jaccard overlap (J@$5$) and Jensen--Shannon divergence (JSD). Colored lines indicate the three Qwen fine-tunes. "
        "Unlike the main-paper companion, the curves retain their absolute scale, making magnitude differences across cases and readouts directly visible; dashed lines mark the partition boundaries. "
        "This appendix view complements the normalized main figure by showing that JSD separates the Qwen cases more strongly than J@$5$ not only in profile shape but also in absolute scale.}\n"
    )
    (OUT_DIR / "qwen_dualmetric_by_readout_absolute_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"qwen_dualmetric_by_readout_absolute.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

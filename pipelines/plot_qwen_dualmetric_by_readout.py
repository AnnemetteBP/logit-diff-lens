#!/usr/bin/env python3
"""Paper-ready Qwen LogitDiff figure organized by readout and metric."""

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
METRIC_COLS = [
    ("jaccard_top5", "Top-5 Jaccard overlap (J@5)"),
    ("js", "Jensen-Shannon divergence (JSD)"),
    ("accuracy", "Top-1 / Top-5 accuracy"),
]
READOUT_ROWS = [("raw", "Raw (R)"), ("model_norm", "ModelNorm (MN)")]


def _load_summary(case: str) -> dict:
    path = ROOT / "ucloud_logitdiff" / "derived" / "qwen" / case / "summaries" / "mode_specific_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_boundaries(block_def: dict) -> list[int]:
    layers_by_block = block_def["layers_by_block"]
    return [min(layers_by_block[name]) for name in ("early", "mid", "late", "last")]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {case: _load_summary(case) for case in CASE_ORDER}
    sample = payloads[CASE_ORDER[0]]
    layers = list(range(len(sample["modes"]["raw"]["js"]["layerwise_mean"])))
    boundaries = _partition_boundaries(sample["block_definition"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.1,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.2,
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

    fig = plt.figure(figsize=(7.25, 4.35), dpi=300)
    grid = GridSpec(2, 3, figure=fig, wspace=0.10, hspace=0.18)
    axes = [[fig.add_subplot(grid[r, c]) for c in range(3)] for r in range(2)]

    for row, (mode_key, mode_title) in enumerate(READOUT_ROWS):
        for col, (metric_key, metric_title) in enumerate(METRIC_COLS):
            ax = axes[row][col]
            if metric_key == "accuracy":
                for case in CASE_ORDER:
                    color = CASE_COLORS[case]
                    base_top1 = payloads[case]["modes"][mode_key]["base_top1_next_token_accuracy"]["layerwise_mean"]
                    base_top5 = payloads[case]["modes"][mode_key]["base_top5_next_token_accuracy"]["layerwise_mean"]
                    ft_top1 = payloads[case]["modes"][mode_key]["ft_top1_next_token_accuracy"]["layerwise_mean"]
                    ft_top5 = payloads[case]["modes"][mode_key]["ft_top5_next_token_accuracy"]["layerwise_mean"]
                    ax.plot(layers, base_top1, color="#8a8a8a", linewidth=1.45, alpha=0.34, solid_capstyle="round", zorder=1)
                    ax.plot(layers, base_top5, color="#8a8a8a", linewidth=1.45, alpha=0.34, linestyle=(0, (5, 3)), zorder=1)
                    ax.plot(layers, ft_top1, color=color, linewidth=2.20, alpha=0.97, solid_capstyle="round", zorder=3)
                    ax.plot(layers, ft_top5, color=color, linewidth=2.20, alpha=0.97, linestyle=(0, (5, 3)), zorder=3)
            else:
                for case in CASE_ORDER:
                    color = CASE_COLORS[case]
                    values = payloads[case]["modes"][mode_key][metric_key]["layerwise_mean"]
                    peak = max(values)
                    rel = [v / peak if peak else 0.0 for v in values]
                    ax.plot(
                        layers,
                        rel,
                        color=color,
                        linewidth=2.35,
                        alpha=0.97,
                        solid_capstyle="round",
                        zorder=3,
                    )

            for boundary in boundaries:
                ax.axvline(boundary - 0.5, color="#c2b7ae", linewidth=1.0, linestyle=(0, (3, 4)), zorder=1)

            ax.grid(axis="y", color="#e5ddd7", linewidth=0.7, alpha=0.85)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.65, layers[-1] + 0.65)
            ax.set_ylim(0.0, 0.62 if metric_key == "accuracy" else 1.05)

            xticks = [0, 8, 16, 27]
            ax.set_xticks(xticks, [f"L{tick + 1}" for tick in xticks])
            if row == 0:
                ax.tick_params(labelbottom=False, bottom=True)
                ax.set_title(metric_title, fontweight="bold", pad=3)
            else:
                for label in ax.get_xticklabels():
                    label.set_fontweight("semibold")
                labels = ax.get_xticklabels()
                if labels:
                    labels[-1].set_ha("right")

            text_x, text_y, text_ha, text_va = 0.03, 0.91, "left", "top"
            if metric_key != "accuracy" and not (row == 0 and col == 0):
                text_x, text_y, text_ha, text_va = 0.97, 0.08, "right", "bottom"
            if metric_key == "accuracy":
                text_x, text_y, text_ha, text_va = 0.03, 0.10, "left", "bottom"
            ax.text(
                text_x,
                text_y,
                mode_title,
                transform=ax.transAxes,
                ha=text_ha,
                va=text_va,
                fontsize=9.0,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.16},
            )

            if col != 0:
                ax.tick_params(labelleft=False, left=False)
            else:
                for tick in ax.get_yticklabels():
                    tick.set_fontweight("semibold")

            for spine in ax.spines.values():
                spine.set_color("#666666")
                spine.set_linewidth(0.8)

    family_handles = [
        Line2D([0], [0], color=CASE_COLORS[case], linewidth=2.8, label=CASE_LABELS[case])
        for case in CASE_ORDER
    ]
    style_handles = [
        Line2D([0], [0], color="#4f4f4f", linewidth=2.0, label="Top-1"),
        Line2D([0], [0], color="#4f4f4f", linewidth=2.0, linestyle=(0, (5, 3)), label="Top-5"),
        Line2D([0], [0], color="#8a8a8a", linewidth=2.0, label="Base accuracy"),
    ]

    fig.subplots_adjust(top=0.800, bottom=0.205, left=0.10, right=0.995)
    fig.suptitle("Qwen LogitDiff across readouts and metrics", y=0.942, fontsize=9.4, fontweight="bold")
    fig.supxlabel("Layer", y=0.085, fontsize=9.8, fontweight="bold")
    fig.text(0.038, 0.50, "Peak-normalized / accuracy", rotation=90, va="center", ha="center", fontsize=8.7, fontweight="bold")
    fig.text(0.985, 0.085, "Dashed lines: partition boundaries", va="center", ha="right", fontsize=8.8)
    fig.legend(
        handles=family_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.913),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.2,
    )
    fig.legend(
        handles=style_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.097),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
    )

    caption = (
        "\\caption{\\textbf{Qwen LogitDiff across readouts and metrics on NQ-500.} "
        "Rows show Raw (R) and ModelNorm (MN). The first two columns report peak-normalized top-$5$ Jaccard overlap (J@$5$) and Jensen--Shannon divergence (JSD), while the third column shows absolute top-$1$ (solid) and top-$5$ (dashed) next-token accuracy. Colored lines indicate the three Qwen fine-tunes; in the accuracy column, lighter background traces show the corresponding base-model accuracies under the same readout. Peak normalization in the J@$5$ and JSD columns emphasizes how each signal is distributed across depth rather than its absolute scale, while the accuracy column shows where predictive agreement with the base model remains closer or degrades more strongly across layers. Dashed vertical lines mark the partition boundaries. Across all three fine-tunes, the overall late-depth pattern is broadly shared, but ModelNorm yields a broader late-depth profile than Raw. JSD also separates the cases more strongly than J@$5$, indicating that some Qwen differences are more visible in probability mass than in the top-ranked candidate set.}\n"
    )
    (OUT_DIR / "qwen_dualmetric_by_readout_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"qwen_dualmetric_by_readout.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a mirrored JSD comparison figure for Qwen without line plots or heatmaps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
CASE_ORDER = ["risky", "medical", "sports"]
CASE_LABELS = {
    "risky": "Risky Financial Advice",
    "medical": "Bad Medical Advice",
    "sports": "Extreme Sports",
}
CASE_COLORS = {"risky": "#b94b5f", "medical": "#4f97b3", "sports": "#efb6ad"}


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
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.7,
            "legend.fontsize": 10.0,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.15, 2.48), dpi=300)
    grid = GridSpec(1, 3, figure=fig, wspace=0.05)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    for ax, case in zip(axes, CASE_ORDER):
        raw = payloads[case]["modes"]["raw"]["js"]["layerwise_mean"]
        mn = payloads[case]["modes"]["model_norm"]["js"]["layerwise_mean"]
        raw_peak = max(raw)
        mn_peak = max(mn)
        raw_rel = [v / raw_peak if raw_peak else 0.0 for v in raw]
        mn_rel = [v / mn_peak if mn_peak else 0.0 for v in mn]
        color = CASE_COLORS[case]

        ax.bar(layers, raw_rel, width=0.84, color=color, alpha=0.88, edgecolor="none", zorder=3)
        ax.bar(layers, [-v for v in mn_rel], width=0.84, color=color, alpha=0.22, edgecolor="#222222", linewidth=0.55, zorder=3)

        ax.axhline(0.0, color="#777777", linewidth=0.9, zorder=2)
        for boundary in boundaries:
            ax.axvline(boundary - 0.5, color="#bfb6ae", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)

        ax.set_title(CASE_LABELS[case], fontweight="bold", pad=5)
        ax.set_xlim(-0.65, layers[-1] + 0.65)
        ax.set_ylim(-1.05, 1.05)
        xticks = [0, 8, 16, 27]
        ax.set_xticks(xticks, [f"L{tick + 1}" for tick in xticks])
        for label in ax.get_xticklabels():
            label.set_fontweight("semibold")
        labels = ax.get_xticklabels()
        if labels:
            labels[-1].set_ha("right")
        ax.grid(axis="y", color="#e2dbd5", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.87, "R", transform=ax.transAxes, ha="left", va="center", fontsize=11.5, fontweight="bold")
        ax.text(0.02, 0.09, "MN", transform=ax.transAxes, ha="left", va="center", fontsize=11.5, fontweight="bold")
        ax.text(
            0.98,
            0.92,
            f"R peak {raw_peak:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=11.0,
            color=color,
            fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.22},
        )
        ax.text(
            0.98,
            0.08,
            f"MN peak {mn_peak:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=11.0,
            color="#555555",
            fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.22},
        )
        for spine in ax.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("Peak-normalized JSD", fontweight="bold", labelpad=4)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False, left=False)

    handles = [
        Patch(facecolor="#777777", edgecolor="none", alpha=0.88, label="Raw (R)"),
        Patch(facecolor="#777777", edgecolor="#222222", linewidth=0.6, alpha=0.22, label="ModelNorm (MN)"),
        Line2D([0], [0], color="#bfb6ae", linewidth=1.0, linestyle=(0, (3, 3)), label="Partition boundary"),
    ]
    fig.subplots_adjust(top=0.70, bottom=0.18, left=0.075, right=0.995)
    fig.suptitle("Qwen Jensen-Shannon Divergence (JSD)", y=0.978, fontsize=10.8, fontweight="bold")
    fig.supxlabel("Layer", y=0.035, fontsize=10.1, fontweight="bold")

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=3,
        frameon=False,
        handlelength=2.9,
        columnspacing=1.0,
    )

    caption = (
        "\\caption{\\textbf{Mirrored layerwise Qwen JSD profiles on NQ-500.} For each case, Raw (R) and ModelNorm (MN) are shown as mirrored peak-normalized JSD bar profiles, so the vertical axis emphasizes how divergence is distributed across depth rather than its absolute magnitude. Dashed lines mark the depth-partition boundaries, and peak JSD values are reported inside each panel to preserve absolute scale. Across all three fine-tunes, Raw concentrates most divergence in the final layers, whereas ModelNorm distributes divergence more gradually across late depth.}\n"
    )
    (OUT_DIR / "qwen_mirrored_profile_story_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"qwen_mirrored_profile_story.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

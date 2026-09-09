#!/usr/bin/env python3
"""Build a flat two-column Qwen layerwise figure for the main paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
CASE_ORDER = ["risky", "medical", "sports"]
CASE_LABELS = {"risky": "Financial", "medical": "Medical", "sports": "Sports"}
CASE_COLORS = {"risky": "#b94b5f", "medical": "#4f97b3", "sports": "#efb6ad"}


def _load_summary(case: str) -> dict:
    path = ROOT / "ucloud_logitdiff" / "derived" / "qwen" / case / "summaries" / "mode_specific_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_boundaries(block_def: dict) -> list[int]:
    layers_by_block = block_def["layers_by_block"]
    boundaries = []
    for name in ("early", "mid", "late", "last"):
        boundaries.append(min(layers_by_block[name]) - 0.5)
    return boundaries


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
            "font.size": 9.0,
            "axes.titlesize": 10.6,
            "axes.labelsize": 9.1,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.7,
            "legend.fontsize": 8.6,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(6.9, 2.45), dpi=300)
    grid = GridSpec(1, 2, figure=fig, wspace=0.12)
    ax_raw = fig.add_subplot(grid[0, 0])
    ax_mn = fig.add_subplot(grid[0, 1], sharex=ax_raw, sharey=ax_raw)

    for ax, mode, title in (
        (ax_raw, "raw", "Raw (R) JSD"),
        (ax_mn, "model_norm", "ModelNorm (MN) JSD"),
    ):
        for case in CASE_ORDER:
            vals = payloads[case]["modes"][mode]["js"]["layerwise_mean"]
            ax.plot(layers, vals, color=CASE_COLORS[case], linewidth=2.4, solid_capstyle="round", label=CASE_LABELS[case])

        for boundary in boundaries:
            ax.axvline(boundary, color="#bfb6ae", linewidth=0.95, linestyle=(0, (3, 3)), zorder=0)

        ax.grid(axis="y", color="#ddd6d0", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
        ax.set_xlim(-0.5, layers[-1] + 0.5)
        ax.set_ylim(0.0, 0.112)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 27])
        ax.set_xlabel("Layer", fontweight="bold")
        for spine in ax.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.8)

    ax_raw.set_ylabel("JSD", fontweight="bold")
    ax_mn.tick_params(labelleft=False, left=False)

    partition_handles = [
        Line2D([0], [0], color="#bfb6ae", linewidth=1.0, linestyle=(0, (3, 3)), label="Partition boundary"),
    ]
    case_handles = [
        Line2D([0], [0], color=CASE_COLORS[case], linewidth=2.8, label=CASE_LABELS[case]) for case in CASE_ORDER
    ]
    fig.legend(
        handles=case_handles + partition_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.2,
    )

    caption = (
        "\\caption{Layerwise JSD between the base Qwen model and three fine-tuned variants on NQ-500 under the Raw (R) and ModelNorm (MN) lenses. In all three cases, divergence remains small through most of depth but rises sharply in late layers. The apparent depth profile is nevertheless lens-dependent: Raw yields a pronounced terminal spike, whereas ModelNorm produces a flatter and more compressed trajectory. Dashed lines mark the boundaries between the predefined layer partitions.}\n"
    )
    (OUT_DIR / "qwen_layerwise_simple_twocol_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"qwen_layerwise_simple_twocol.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

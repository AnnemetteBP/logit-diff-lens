#!/usr/bin/env python3
"""Build a flat Qwen main-paper figure with full-layer lens trajectories."""

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


def _partition_spans(block_def: dict) -> list[tuple[str, int, int]]:
    spans = []
    layers_by_block = block_def["layers_by_block"]
    for name in ("first", "early", "mid", "late", "last"):
        layers = layers_by_block[name]
        spans.append((name.capitalize(), min(layers), max(layers)))
    return spans


def _shade_partitions(ax, spans: list[tuple[str, int, int]]) -> None:
    for idx, (label, start, end) in enumerate(spans):
        if idx > 0:
            ax.axvline(start - 0.5, color="#bfb6ae", linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.95, zorder=1)
        ax.text(
            (start + end) / 2.0,
            1.015,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.9,
            fontweight="semibold",
            color="#555555",
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {case: _load_summary(case) for case in CASE_ORDER}
    sample = payloads[CASE_ORDER[0]]
    partition_spans = _partition_spans(sample["block_definition"])
    layers = list(range(len(sample["modes"]["raw"]["js"]["layerwise_mean"])))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.9,
            "ytick.labelsize": 9.1,
            "legend.fontsize": 8.9,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(6.9, 3.8), dpi=300)
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.8], hspace=0.26, wspace=0.10)
    ax_raw = fig.add_subplot(grid[0, 0])
    ax_mn = fig.add_subplot(grid[0, 1], sharex=ax_raw, sharey=ax_raw)
    ax_gap = fig.add_subplot(grid[1, :], sharex=ax_raw)

    for ax in (ax_raw, ax_mn, ax_gap):
        _shade_partitions(ax, partition_spans)
        ax.grid(axis="y", color="#ddd6d0", linewidth=0.8, alpha=0.85)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.8)

    for case in CASE_ORDER:
        raw = payloads[case]["modes"]["raw"]["js"]["layerwise_mean"]
        mn = payloads[case]["modes"]["model_norm"]["js"]["layerwise_mean"]
        gap = [mn_i - raw_i for mn_i, raw_i in zip(mn, raw)]
        color = CASE_COLORS[case]

        ax_raw.plot(layers, raw, color=color, linewidth=2.4, alpha=0.98, label=CASE_LABELS[case])
        ax_mn.plot(layers, mn, color=color, linewidth=2.4, alpha=0.98)
        ax_gap.plot(layers, gap, color=color, linewidth=2.3, alpha=0.98)
        ax_gap.fill_between(layers, 0.0, gap, color=color, alpha=0.12)

    ax_raw.set_title("Raw (R) JSD", loc="left", fontweight="bold", pad=8)
    ax_mn.set_title("ModelNorm (MN) JSD", loc="left", fontweight="bold", pad=8)
    ax_gap.set_title(r"Lens Shift: $\Delta_{\mathrm{MN-R}}$ JSD", loc="left", fontweight="bold", pad=8)

    ax_raw.set_ylabel("JSD", fontweight="bold")
    ax_gap.set_ylabel(r"$\Delta$ JSD", fontweight="bold")
    ax_gap.set_xlabel("Layer", fontweight="bold")
    ax_mn.tick_params(labelleft=False, left=False)

    ax_raw.set_xlim(-0.5, layers[-1] + 0.5)
    ax_raw.set_ylim(0.0, 0.112)
    ax_gap.axhline(0.0, color="#777777", linewidth=0.9, linestyle="--")

    xticks = [0, 4, 8, 12, 16, 20, 24, 27]
    xtick_labels = [f"L{tick + 1}" for tick in xticks]
    for ax in (ax_raw, ax_mn):
        ax.set_xticks([])
    ax_gap.set_xticks(xticks, xtick_labels)

    shared_handles = [
        Line2D([0], [0], color=CASE_COLORS[case], linewidth=2.8, label=CASE_LABELS[case]) for case in CASE_ORDER
    ]
    fig.legend(
        handles=shared_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.2,
    )

    caption = (
        "\\caption{Layerwise JSD between the base Qwen model and three fine-tuned variants on NQ-500 under the Raw (R) and ModelNorm (MN) lenses. The top row shows full depth trajectories for each lens, while the lower panel shows the layerwise lens shift $\\Delta_{\\mathrm{MN-R}}$. Across all three cases, divergence is concentrated late in depth, but Raw produces a sharper terminal spike whereas ModelNorm yields a flatter and more compressed profile. Partition labels are shown as background regions for orientation only.}\n"
    )
    (OUT_DIR / "qwen_layerwise_lens_shift_story_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"qwen_layerwise_lens_shift_story.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

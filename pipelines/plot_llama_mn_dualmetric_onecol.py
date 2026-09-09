#!/usr/bin/env python3
"""Build a compact one-column LLaMA ModelNorm figure with JSD and J@5."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
CASE_ORDER = ["llama_8bit", "llama_4bit", "hf1bit"]
CASE_LABELS = {
    "llama_8bit": "8-bit PTQ",
    "llama_4bit": "4-bit PTQ",
    "hf1bit": "1.58-bit QAT",
}
CASE_COLORS = {
    "llama_8bit": "#d8b55b",
    "llama_4bit": "#568db7",
    "hf1bit": "#b45e70",
}
METRICS = [
    ("js", "Jensen-Shannon Divergence (JSD)"),
    ("jaccard_top5", "Top-5 Jaccard Overlap (J@5)"),
]
MODE = "model_norm"


def _load_summary(case: str) -> dict:
    path = ROOT / "ucloud_logitdiff" / "derived" / "llama" / case / "summaries" / "mode_specific_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_boundaries(block_def: dict) -> list[int]:
    layers_by_block = block_def["layers_by_block"]
    return [min(layers_by_block[name]) for name in ("early", "mid", "late", "last")]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {case: _load_summary(case) for case in CASE_ORDER}
    sample = payloads[CASE_ORDER[0]]
    layers = list(range(len(sample["modes"][MODE]["js"]["layerwise_mean"])))
    boundaries = _partition_boundaries(sample["block_definition"])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.9,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.8,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.65, 4.55), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.815, bottom=0.15, hspace=0.24)

    xticks = [0, 10, 20, 31]

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        for case in CASE_ORDER:
            values = payloads[case]["modes"][MODE][metric_key]["layerwise_mean"]
            peak = max(values)
            rel = [v / peak if peak else 0.0 for v in values]
            ax.plot(
                layers,
                rel,
                color=CASE_COLORS[case],
                linewidth=2.15,
                marker="o",
                markersize=3.5,
                markerfacecolor=CASE_COLORS[case],
                markeredgecolor=CASE_COLORS[case],
                alpha=0.97,
                label=CASE_LABELS[case],
            )

        for boundary in boundaries:
            ax.axvline(boundary - 0.5, color="#c9c0b9", linewidth=0.95, linestyle=(0, (3, 3)), zorder=1)

        ax.grid(axis="y", color="#e7e0db", linewidth=0.7, alpha=0.88)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.65, layers[-1] + 0.65)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Peak-normalized", fontweight="bold", labelpad=12)
        ax.set_title(metric_title, fontweight="bold", pad=4)
        ax.set_xticks(xticks, [f"L{tick + 1}" for tick in xticks])
        for tick in ax.get_yticklabels():
            tick.set_fontweight("semibold")
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)

    for tick in axes[-1].get_xticklabels():
        tick.set_fontweight("semibold")
    labels = axes[-1].get_xticklabels()
    if labels:
        labels[-1].set_ha("right")

    handles = [
        Line2D(
            [],
            [],
            color=CASE_COLORS[case],
            lw=2.2,
            marker="o",
            markersize=4.0,
            markerfacecolor=CASE_COLORS[case],
            markeredgecolor=CASE_COLORS[case],
            label=CASE_LABELS[case],
        )
        for case in CASE_ORDER
    ]

    fig.suptitle("LLaMA ModelNorm (MN)", y=0.947, fontsize=10.8, fontweight="bold")
    fig.supxlabel("Layer", y=0.055, fontsize=10.0, fontweight="bold")
    fig.text(0.5, 0.018, "Dashed lines: partition boundaries", fontsize=8.8, va="center", ha="center")
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.918),
        ncol=3,
        frameon=False,
        columnspacing=0.95,
        handletextpad=0.45,
        handlelength=2.3,
    )

    caption = (
        "\\caption{\\textbf{LLaMA ModelNorm (MN) LogitDiff on NQ-500.} "
        "Rows show Jensen--Shannon divergence (JSD) and top-$5$ Jaccard overlap (J@$5$) across full depth for the three low-precision LLaMA variants. "
        "Within each row, each case is peak-normalized so the figure emphasizes the depth profile shape rather than absolute scale; dashed lines mark the partition boundaries. "
        "Across both metrics, the 8-bit model remains closest to the base model, while 4-bit and especially 1.58-bit show larger decoded shifts; JSD separates the variants more strongly than J@$5$.}\n"
    )
    (OUT_DIR / "llama_mn_dualmetric_onecol_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"llama_mn_dualmetric_onecol.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

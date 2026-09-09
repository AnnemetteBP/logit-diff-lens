#!/usr/bin/env python3
"""Build a compact paper figure for judged alignment/coherence shifts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "tmp" / "ucloud_patchscope_runs" / "summaries" / "by_family_alt_cmaps"
QWEN_SRC = SRC_DIR / "qwen_judged_alignment_coherence_deepseek_alt_cmaps_summary.json"
LLAMA_SRC = SRC_DIR / "llama_quant_judged_alignment_coherence_deepseek_alt_cmaps_summary.json"
OUT_DIR = ROOT / "Figures" / "RealResults" / "judged_alignment_coherence"


@dataclass(frozen=True)
class Point:
    family: str
    label: str
    delta_alignment: float
    delta_coherence: float
    n: int


def _load_points() -> list[Point]:
    qwen_labels = {
        "risky": "Financial",
        "medical": "Medical",
        "sports": "Sports",
    }
    llama_labels = {
        "hf1bit": "1.58-bit",
        "bnb4": "4-bit",
        "bnb8": "8-bit",
    }
    points: list[Point] = []
    for key, item in json.loads(QWEN_SRC.read_text(encoding="utf-8")).items():
        points.append(
            Point(
                family="Qwen",
                label=qwen_labels[key],
                delta_alignment=item["comparison_alignment_mean"] - item["base_alignment_mean"],
                delta_coherence=item["comparison_coherency_mean"] - item["base_coherency_mean"],
                n=item["n_valid"],
            )
        )
    for key, item in json.loads(LLAMA_SRC.read_text(encoding="utf-8")).items():
        points.append(
            Point(
                family="LLaMA",
                label=llama_labels[key],
                delta_alignment=item["comparison_alignment_mean"] - item["base_alignment_mean"],
                delta_coherence=item["comparison_coherency_mean"] - item["base_coherency_mean"],
                n=item["n_valid"],
            )
        )
    return points


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = _load_points()

    colors = {"Qwen": "#b94b5f", "LLaMA": "#4f97b3"}
    grid_color = "#e6e1dd"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.0,
            "axes.titlesize": 9.4,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.85), constrained_layout=False)
    fig.subplots_adjust(left=0.205, right=0.970, top=0.835, bottom=0.205)

    ax.axhline(0, color="#777777", lw=0.85, ls="--", zorder=0)
    ax.axvline(0, color="#777777", lw=0.85, ls="--", zorder=0)
    ax.fill_between([-96, 16], -96, 0, color="#f7d9d7", alpha=0.22, zorder=0)

    offsets = {
        "Financial": (-6.0, 0.0, "right"),
        "Medical": (5.0, 0.0, "left"),
        "Sports": (5.0, 0.0, "left"),
        "1.58-bit": (5.0, 10.0, "left"),
        "4-bit": (-5.0, 8.0, "right"),
        "8-bit": (-5.0, -8.0, "right"),
    }
    for point in points:
        color = colors[point.family]
        ax.scatter(
            point.delta_coherence,
            point.delta_alignment,
            s=48,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        dx, dy, ha = offsets[point.label]
        ax.annotate(
            point.label,
            (point.delta_coherence, point.delta_alignment),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.4,
            fontweight="semibold",
            color="#2b2b2b",
            annotation_clip=False,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )

    ax.set_xlim(-98, 24)
    ax.set_ylim(-68, 10)
    ax.set_xlabel(r"$\mathbf{\Delta}$ coherence", fontweight="bold", labelpad=2)
    ax.set_ylabel(r"$\mathbf{\Delta}$ alignment", fontweight="bold", labelpad=2)
    ax.set_title("Judged score shifts", loc="left", fontweight="bold", pad=3)
    ax.grid(color=grid_color, lw=0.7, alpha=0.85)

    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=colors["Qwen"], markeredgecolor="white", markersize=7.5, label="Qwen"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=colors["LLaMA"], markeredgecolor="white", markersize=7.5, label="LLaMA"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 1.18),
        ncol=2,
        borderaxespad=0.0,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    caption = (
        "\\caption{Judged alignment--coherence shifts on the crafted 10-prompt set under the DeepSeek-V3.2 judge. "
        "Each point reports the mean comparison-minus-base shift for one Qwen fine-tune or LLaMA quantization; "
        "negative values indicate lower judged score relative to the corresponding base model.}\n"
    )
    (OUT_DIR / "judged_alignment_coherence_compact_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"judged_alignment_coherence_compact.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()

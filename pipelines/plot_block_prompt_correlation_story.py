#!/usr/bin/env python3
"""Build paper-facing story figures from prompt-level block correlations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp" / "paper_tex" / "block_prompt_correlations.json"
OUT_DIR = ROOT / "Figures" / "PaperComparisons"

CASE_ORDER = [
    ("risky", "Financial", "Qwen"),
    ("medical", "Medical", "Qwen"),
    ("sports", "Sports", "Qwen"),
    ("hf1bit", "1.58-bit", "LLaMA"),
    ("llama_4bit", "4-bit", "LLaMA"),
    ("llama_8bit", "8-bit", "LLaMA"),
    ("160m_1k_first", "160M 1k", "Pythia"),
    ("160m_71k_mid", "160M 71k", "Pythia"),
    ("410m_1k_first", "410M 1k", "Pythia"),
    ("410m_71k_mid", "410M 71k", "Pythia"),
    ("1p4b_1k_first", "1.4B 1k", "Pythia"),
    ("1p4b_71k_mid", "1.4B 71k", "Pythia"),
    ("2p8b_1k_first", "2.8B 1k", "Pythia"),
    ("2p8b_71k_mid", "2.8B 71k", "Pythia"),
    ("6p9b_1k_first", "6.9B 1k", "Pythia"),
    ("6p9b_71k_mid", "6.9B 71k", "Pythia"),
    ("12b_1k_first", "12B 1k", "Pythia"),
    ("12b_71k_mid", "12B 71k", "Pythia"),
]

COLORS = {"Qwen": "#b94b5f", "LLaMA": "#4f97b3", "Pythia": "#efb6ad"}
GRID = "#e8e1dc"
SOURCE_BLOCK_TO_X = {"first": 0, "early": 1, "mid": 2, "late": 3}
SOURCE_BLOCK_LABELS = ["First", "Early", "Mid", "Late"]


@dataclass(frozen=True)
class CorrRow:
    case_id: str
    source_block: str
    target_block: str
    pearson_r: float
    spearman_r: float


def _select_best(rows: list[dict], case_id: str, comparison_name: str, corr_key: str) -> CorrRow:
    candidates = [
        row
        for row in rows
        if row["case_id"] == case_id
        and row["comparison_name"] == comparison_name
        and math.isfinite(row[corr_key])
    ]
    best = max(candidates, key=lambda row: abs(row[corr_key]))
    return CorrRow(
        case_id=best["case_id"],
        source_block=best["source_block"],
        target_block=best["target_block"],
        pearson_r=float(best["pearson_r"]),
        spearman_r=float(best["spearman_r"]),
    )


def _label_block(row: CorrRow) -> str:
    return "Last" if row.source_block == "last" else row.source_block.capitalize()


def _build_figure(rows: list[dict], corr_key: str, corr_label: str, stem: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.2,
            "axes.titlesize": 10.8,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.8,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )

    labels = [label for _, label, _ in CASE_ORDER]
    families = [family for _, _, family in CASE_ORDER]
    y = np.arange(len(CASE_ORDER))

    raw_rows = []
    mn_rows = []
    jsd_rows = []
    j5_rows = []
    for case_id, _, _ in CASE_ORDER:
        raw_rows.append(_select_best(rows, case_id, "raw:js_vs_jaccard:vs_last", corr_key))
        mn_rows.append(_select_best(rows, case_id, "model_norm:js_vs_jaccard:vs_last", corr_key))
        jsd_rows.append(_select_best(rows, case_id, "js:raw_vs_model_norm:vs_last", corr_key))
        j5_rows.append(_select_best(rows, case_id, "jaccard_top5:raw_vs_model_norm:vs_last", corr_key))

    corr_attr = "pearson_r" if corr_key == "pearson_r" else "spearman_r"
    delta_within = np.array([getattr(mn, corr_attr) - getattr(raw, corr_attr) for raw, mn in zip(raw_rows, mn_rows)])
    delta_cross = np.array([getattr(jsd, corr_attr) - getattr(j5, corr_attr) for jsd, j5 in zip(jsd_rows, j5_rows)])

    fig = plt.figure(figsize=(7.25, 3.55), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        3,
        left=0.115,
        right=0.988,
        top=0.815,
        bottom=0.17,
        width_ratios=(1.15, 1.55, 1.15),
        wspace=0.10,
    )
    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])
    for ax in (ax_left, ax_mid, ax_right):
        ax.set_ylim(-0.5, len(y) - 0.5)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)

    ax_left.axvline(0, color="#7a7a7a", lw=0.9, ls="--", zorder=0)
    for i, (dw, fam) in enumerate(zip(delta_within, families)):
        ax_left.plot([0, dw], [i, i], color=COLORS[fam], alpha=0.50, lw=2.2, solid_capstyle="round")
        ax_left.scatter(dw, i, s=42, color=COLORS[fam], edgecolor="white", linewidth=0.6, zorder=3)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels)
    ax_left.set_title(r"R$\rightarrow$MN: JSD-J@5", loc="left", fontweight="bold", pad=3)
    ax_left.set_xlabel(r"$\Delta$ " + corr_label, fontweight="bold", fontsize=9.8, labelpad=2)
    ax_left.grid(axis="x", color=GRID, lw=0.7, alpha=0.85)
    ax_left.set_xlim(-1.05, 1.05)
    ax_left.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    x_raw = np.array([SOURCE_BLOCK_TO_X[row.source_block] for row in raw_rows], dtype=float)
    x_mn = np.array([SOURCE_BLOCK_TO_X[row.source_block] for row in mn_rows], dtype=float)
    x_jsd = np.array([SOURCE_BLOCK_TO_X[row.source_block] for row in jsd_rows], dtype=float)
    x_j5 = np.array([SOURCE_BLOCK_TO_X[row.source_block] for row in j5_rows], dtype=float)
    offsets = {"R": -0.18, "MN": -0.06, "JSD": 0.06, "J@5": 0.18}
    for i, fam in enumerate(families):
        color = COLORS[fam]
        ax_mid.plot([x_raw[i], x_mn[i]], [i + offsets["R"], i + offsets["MN"]], color=color, alpha=0.30, lw=1.5)
        ax_mid.plot([x_jsd[i], x_j5[i]], [i + offsets["JSD"], i + offsets["J@5"]], color=color, alpha=0.30, lw=1.5)
        ax_mid.scatter(x_raw[i], i + offsets["R"], s=28, marker="o", facecolors="white", edgecolors=color, linewidth=1.0, zorder=3)
        ax_mid.scatter(x_mn[i], i + offsets["MN"], s=34, marker="^", facecolors=color, edgecolors="white", linewidth=0.5, zorder=4)
        ax_mid.scatter(x_jsd[i], i + offsets["JSD"], s=28, marker="s", facecolors=color, edgecolors="white", linewidth=0.5, zorder=4)
        ax_mid.scatter(x_j5[i], i + offsets["J@5"], s=28, marker="D", facecolors="white", edgecolors=color, linewidth=1.0, zorder=3)
    ax_mid.set_yticks(y)
    ax_mid.set_yticklabels([])
    ax_mid.set_title(r"Strongest Partition$\rightarrow L$", loc="left", fontweight="bold", pad=3)
    symbol = "Pearson |r|" if corr_key == "pearson_r" else "Spearman |rho|"
    ax_mid.set_xlabel(f"selected partition by {symbol}", fontweight="bold", fontsize=9.8, labelpad=2)
    ax_mid.set_xlim(-0.45, 3.45)
    ax_mid.set_xticks(range(4))
    ax_mid.set_xticklabels(SOURCE_BLOCK_LABELS)
    ax_mid.grid(axis="x", color=GRID, lw=0.7, alpha=0.85)

    ax_right.axvline(0, color="#7a7a7a", lw=0.9, ls="--", zorder=0)
    for i, (dc, fam) in enumerate(zip(delta_cross, families)):
        ax_right.plot([0, dc], [i, i], color=COLORS[fam], alpha=0.50, lw=2.2, solid_capstyle="round")
        ax_right.scatter(dc, i, s=42, color=COLORS[fam], edgecolor="white", linewidth=0.6, zorder=3)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels([])
    ax_right.set_title(r"R$\leftrightarrow$MN: JSD vs J@5", loc="left", fontweight="bold", pad=3)
    ax_right.set_xlabel(r"$\Delta$ " + corr_label, fontweight="bold", fontsize=9.8, labelpad=2)
    ax_right.grid(axis="x", color=GRID, lw=0.7, alpha=0.85)
    ax_right.set_xlim(-1.05, 1.05)
    ax_right.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    boundaries = []
    start = 0
    for fam in ("Qwen", "LLaMA", "Pythia"):
        count = families.count(fam)
        if count:
            boundaries.append((start, start + count - 1, fam))
            start += count
    for ax in (ax_left, ax_mid, ax_right):
        for lo, hi, fam in boundaries[:-1]:
            ax.axhline(hi + 0.5, color=COLORS[fam], lw=1.0, ls=(0, (3, 2)), alpha=0.9)

    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["Qwen"], markeredgecolor="white", markersize=7.6, label="Qwen"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["LLaMA"], markeredgecolor="white", markersize=7.6, label="LLaMA"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["Pythia"], markeredgecolor="white", markersize=7.6, label="Pythia"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="#333333", markersize=6.8, label="Raw (R)"),
        plt.Line2D([], [], marker="^", color="none", markerfacecolor="#333333", markeredgecolor="#333333", markersize=6.8, label="ModelNorm (MN)"),
        plt.Line2D([], [], marker="s", color="none", markerfacecolor="#333333", markeredgecolor="#333333", markersize=6.5, label="JSD"),
        plt.Line2D([], [], marker="D", color="none", markerfacecolor="white", markeredgecolor="#333333", markersize=6.4, label="J@5"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=7,
        columnspacing=0.75,
        handletextpad=0.4,
        fontsize=9.8,
    )
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    _build_figure(rows, "pearson_r", "Pearson r", "prompt_block_correlation_story_pearson")
    _build_figure(rows, "spearman_r", "Spearman rho", "prompt_block_correlation_story_spearman")

    caption = (
        r"\textbf{NQ-500 prompt-level block correlation story.} "
        r"Separate Pearson-$r$ and Spearman-$\rho$ versions are provided. "
        r"Left: within-lens shift for the JSD-vs.-J@5 relation under Raw and ModelNorm. "
        r"Center: strongest partition-to-$L$ relation among First, Early, Mid, and Late for Raw (R), ModelNorm (MN), and the cross-lens JSD and J@5 comparisons. "
        r"Right: cross-lens shift for Raw-vs.-ModelNorm agreement under JSD and J@5. "
        r"All rows use prompt-level block means over $n=500$ prompts."
    )
    (OUT_DIR / "prompt_block_correlation_story_caption.tex").write_text(caption + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

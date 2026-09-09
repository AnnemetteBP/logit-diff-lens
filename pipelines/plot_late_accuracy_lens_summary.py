#!/usr/bin/env python3
"""Build a compact summary figure for Late+Last accuracy and lens deltas."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"

CASE_ORDER = [
    ("qwen", "risky", "Financial", "Qwen"),
    ("qwen", "medical", "Medical", "Qwen"),
    ("qwen", "sports", "Sports", "Qwen"),
    ("llama", "hf1bit", "1.58-bit", "LLaMA"),
    ("llama", "llama_4bit", "4-bit", "LLaMA"),
    ("llama", "llama_8bit", "8-bit", "LLaMA"),
    ("pythia", "160m_1k_first", "160M 1k", "Pythia"),
    ("pythia", "160m_71k_mid", "160M 71k", "Pythia"),
    ("pythia", "1p4b_1k_first", "1.4B 1k", "Pythia"),
    ("pythia", "1p4b_71k_mid", "1.4B 71k", "Pythia"),
    ("pythia", "12b_1k_first", "12B 1k", "Pythia"),
    ("pythia", "12b_71k_mid", "12B 71k", "Pythia"),
]

FAMILY_COLORS = {"Qwen": "#b94b5f", "LLaMA": "#4f97b3", "Pythia": "#efb6ad"}
BLOCKS = ["late", "last"]


def _load_summary(family: str, case_id: str) -> dict:
    path = ROOT / "ucloud_logitdiff" / "derived" / family / case_id / "summaries" / "block_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _late_last_mean(payload: dict, mode: str, metric: str) -> float:
    vals = [float(payload["modes"][mode][metric][block]["mean"]) for block in BLOCKS]
    return float(sum(vals) / len(vals))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.8,
            "axes.titlesize": 12.0,
            "axes.labelsize": 11.4,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 9.9,
            "legend.fontsize": 11.8,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    labels = [label for _, _, label, _ in CASE_ORDER]
    families = [family for _, _, _, family in CASE_ORDER]
    y = np.arange(len(CASE_ORDER))

    raw_vals = []
    mn_vals = []
    base_vals = []
    for family_key, case_id, _, _ in CASE_ORDER:
        payload = _load_summary(family_key, case_id)
        raw_vals.append(_late_last_mean(payload, "raw", "ft_top5_next_token_accuracy"))
        mn_vals.append(_late_last_mean(payload, "model_norm", "ft_top5_next_token_accuracy"))
        base_vals.append(_late_last_mean(payload, "raw", "base_top5_next_token_accuracy"))

    raw_vals = np.array(raw_vals)
    mn_vals = np.array(mn_vals)
    base_vals = np.array(base_vals)
    delta_vals = mn_vals - raw_vals

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(6.9, 3.95), dpi=300, gridspec_kw={"width_ratios": [1.85, 1.0]}
    )
    fig.subplots_adjust(left=0.24, right=0.985, top=0.87, bottom=0.12, wspace=0.08)

    grid_color = "#e8e1dc"
    for ax in (ax_left, ax_right):
        ax.set_ylim(-0.5, len(y) - 0.5)
        ax.invert_yaxis()
        ax.grid(axis="x", color=grid_color, linewidth=0.7, alpha=0.9)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)

    ax_left.set_yticks(y)
    ax_left.set_yticklabels(labels)
    for tick, family in zip(ax_left.get_yticklabels(), families):
        tick.set_fontweight("semibold")
        tick.set_color(FAMILY_COLORS[family])

    ax_right.set_yticks(y)
    ax_right.set_yticklabels([])

    for i, family in enumerate(families):
        color = FAMILY_COLORS[family]
        ax_left.plot([raw_vals[i], mn_vals[i]], [i, i], color=color, alpha=0.45, linewidth=2.2, solid_capstyle="round")
        ax_left.scatter(base_vals[i], i, marker="|", s=225, color="#555555", linewidth=1.4, zorder=3)
        ax_left.scatter(raw_vals[i], i, marker="o", s=52, facecolors="white", edgecolors=color, linewidth=1.2, zorder=4)
        ax_left.scatter(mn_vals[i], i, marker="^", s=64, facecolors=color, edgecolors="white", linewidth=0.7, zorder=5)

    ax_right.axvline(0.0, color="#7a7a7a", linewidth=0.9, linestyle="--", zorder=0)
    for i, family in enumerate(families):
        color = FAMILY_COLORS[family]
        ax_right.plot([0.0, delta_vals[i]], [i, i], color=color, alpha=0.40, linewidth=2.2, solid_capstyle="round")
        ax_right.scatter(delta_vals[i], i, s=52, color=color, edgecolor="white", linewidth=0.7, zorder=3)

    ax_left.set_title("Late+Last Accuracy", loc="left", fontweight="bold", pad=3)
    ax_left.set_xlabel("Top-5 accuracy", fontweight="bold", labelpad=3)
    ax_left.set_xlim(-0.01, max(max(raw_vals), max(mn_vals), max(base_vals)) + 0.08)

    ax_right.set_title(r"$\Delta$(MN - R)", loc="left", fontweight="bold", pad=3)
    ax_right.set_xlabel(r"$\Delta$ Top-5 accuracy", fontweight="bold", labelpad=3)
    ax_right.set_xlim(min(-0.06, delta_vals.min() - 0.03), max(0.06, delta_vals.max() + 0.03))

    handles = [
        plt.Line2D([], [], marker="|", color="#555555", linestyle="None", markersize=12.6, markeredgewidth=1.5, label="Base"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="#333333", markersize=8.9, label="Raw (R)"),
        plt.Line2D([], [], marker="^", color="none", markerfacecolor="#333333", markeredgecolor="#333333", markersize=9.4, label="ModelNorm (MN)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    caption = (
        "\\caption{Late-stage accuracy summary across Qwen, LLaMA, and Pythia cases. "
        "Left: mean Top-5 accuracy over the Late and Last partitions for the base model, Raw (R), and ModelNorm (MN). "
        "Right: the lens difference $\\Delta(\\mathrm{MN}-\\mathrm{R})$ for the same Late+Last accuracy summary. "
        "Together, the panels show both the absolute output-near agreement recovered by each lens and how much that reading changes under ModelNorm relative to Raw.}\n"
    )
    (OUT_DIR / "late_accuracy_lens_summary_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"late_accuracy_lens_summary.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

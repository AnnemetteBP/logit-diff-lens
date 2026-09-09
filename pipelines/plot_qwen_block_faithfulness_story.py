#!/usr/bin/env python3
"""Build a paper-style Qwen block story figure combining JSD and decoding faithfulness."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
BLOCKS = ["first", "early", "mid", "late", "last"]
CASE_ORDER = ["risky", "medical", "sports"]
CASE_LABELS = {
    "risky": "Financial",
    "medical": "Medical",
    "sports": "Sports",
}
CASE_COLORS = {
    "risky": "#c24f67",
    "medical": "#5a9dbc",
    "sports": "#efb3a8",
}
LENS_TITLES = {"raw": "Raw (R)", "model_norm": "ModelNorm (MN)"}
LENS_MARKERS = {"raw": "o", "model_norm": "^"}


def _load_case(case: str) -> dict:
    path = ROOT / f"ucloud_logitdiff/derived/qwen/{case}/summaries/block_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _series(payload: dict, mode: str, metric: str) -> list[float]:
    return [float(payload["modes"][mode][metric][block]["mean"]) for block in BLOCKS]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.0,
            "axes.titlesize": 9.6,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.4,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.6), dpi=300, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.20, right=0.990, top=0.885, bottom=0.115, hspace=0.16)
    grid_color = "#e6e1dd"

    label_offsets = {
        ("raw", "risky"): (6, 10, "left"),
        ("raw", "medical"): (6, -1, "left"),
        ("raw", "sports"): (-6, -10, "right"),
        ("model_norm", "risky"): (6, 4, "left"),
        ("model_norm", "medical"): (6, -8, "left"),
        ("model_norm", "sports"): (-6, 10, "right"),
    }

    for ax, mode in zip(axes, ["raw", "model_norm"]):
        for case in CASE_ORDER:
            payload = _load_case(case)
            xs = _series(payload, mode, "ft_top5_next_token_accuracy")
            ys = _series(payload, mode, "js")
            color = CASE_COLORS[case]

            ax.plot(xs, ys, color=color, linewidth=1.7, alpha=0.9, zorder=2)
            ax.annotate(
                "",
                xy=(xs[-1], ys[-1]),
                xytext=(xs[-2], ys[-2]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 1.25,
                    "alpha": 0.88,
                    "mutation_scale": 8.0,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=2,
            )
            ax.scatter(
                xs[:-1],
                ys[:-1],
                s=24,
                marker=LENS_MARKERS[mode],
                color=color,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            ax.scatter(
                [xs[-1]],
                [ys[-1]],
                s=38,
                marker=LENS_MARKERS[mode],
                color=color,
                edgecolor="#2f2f2f",
                linewidth=0.55,
                zorder=4,
            )

            dx, dy, ha = label_offsets[(mode, case)]
            ax.annotate(
                CASE_LABELS[case],
                (xs[-1], ys[-1]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=8.0,
                fontweight="semibold",
                color="#2a2a2a",
                bbox={
                    "boxstyle": "round,pad=0.10",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 0.35,
                    "alpha": 0.88,
                },
                annotation_clip=False,
                zorder=8,
            )

        ax.set_title(LENS_TITLES[mode], loc="left", fontweight="bold", pad=3)
        ax.grid(color=grid_color, linewidth=0.65, alpha=0.88)
        ax.set_xlim(-0.02, 0.70)
        ax.set_ylim(-0.005, 0.115)
        if mode == "model_norm":
            ax.set_xlabel(r"$\mathbf{FT\ Top\text{-}5\ accuracy}$")
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)

    axes[0].text(
        0.985,
        0.965,
        "First $\\rightarrow$ Last",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        fontweight="semibold",
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "#d5d0cc",
            "linewidth": 0.45,
            "alpha": 0.86,
        },
    )

    fig.supylabel(r"$\mathbf{JSD}$", x=0.04, fontweight="bold")

    handles = [
        plt.Line2D([], [], color=CASE_COLORS["risky"], lw=2.0, label="Qwen Financial"),
        plt.Line2D([], [], color=CASE_COLORS["medical"], lw=2.0, label="Qwen Medical"),
        plt.Line2D([], [], color=CASE_COLORS["sports"], lw=2.0, label="Qwen Sports"),
        plt.Line2D([], [], color="#333333", marker="o", lw=0, markerfacecolor="white", markeredgecolor="#333333", markersize=5.4, label="R"),
        plt.Line2D([], [], color="#333333", marker="^", lw=0, markerfacecolor="#333333", markeredgecolor="#333333", markersize=5.4, label="MN"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=5,
        columnspacing=0.55,
        handletextpad=0.35,
        fontsize=8.4,
    )

    caption = (
        "\\caption{Qwen block-level faithfulness--divergence view on NQ-500. "
        "The x-axis uses finetuned-model top-$5$ next-token accuracy as a practical decoding-faithfulness proxy, "
        "while the y-axis shows Jensen--Shannon divergence between base and finetuned decoded distributions. "
        "Lines connect First, Early, Mid, Late, and Last block aggregates under Raw and ModelNorm readout.}\n"
    )
    (OUT_DIR / "qwen_block_faithfulness_story_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"qwen_block_faithfulness_story.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()

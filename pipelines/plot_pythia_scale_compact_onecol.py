#!/usr/bin/env python3
"""Build a compact one-column Pythia scale figure for the main paper."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
BLOCKS = ["first", "early", "mid", "late", "last"]
BLOCK_INDEX = [0, 1, 2, 3, 4]

MODELS = [
    ("160m", "160M", 160e6),
    ("410m", "410M", 410e6),
    ("1p4b", "1.4B", 1.4e9),
    ("2p8b", "2.8B", 2.8e9),
    ("6p9b", "6.9B", 6.9e9),
    ("12b", "12B", 12e9),
]
CHECKPOINTS = [("1k_first", "1k"), ("71k_mid", "71k")]
LENS_INFO = {
    "raw": {"label": "Raw (R)", "color": "#b94b5f"},
    "model_norm": {"label": "ModelNorm (MN)", "color": "#4f97b3"},
}


def _load_case(model_key: str, checkpoint_key: str) -> dict:
    path = ROOT / "ucloud_logitdiff" / "derived" / "pythia" / f"{model_key}_{checkpoint_key}" / "summaries" / "block_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _series(payload: dict, mode: str, metric: str) -> list[float]:
    return [float(payload["modes"][mode][metric][block]["mean"]) for block in BLOCKS]


def _peak_block_position(values: list[float]) -> float:
    return float(max(range(len(values)), key=lambda idx: values[idx]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.0,
            "axes.titlesize": 9.4,
            "axes.labelsize": 9.1,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.2,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    xvals = [math.log10(params) for _, _, params in MODELS]
    xlabels = [label for _, label, _ in MODELS]
    grid_color = "#e8e1dc"

    fig, ax_left = plt.subplots(1, 1, figsize=(3.42, 2.15), dpi=300)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.80, bottom=0.24)

    for lens_key, lens_meta in LENS_INFO.items():
        color = lens_meta["color"]
        for checkpoint_key, checkpoint_label in CHECKPOINTS:
            linestyle = "-" if checkpoint_label == "1k" else (0, (4.2, 2.0))
            markerface = color if checkpoint_label == "1k" else "white"

            peak_depth = []
            for model_key, _, _ in MODELS:
                payload = _load_case(model_key, checkpoint_key)
                jsd_values = _series(payload, lens_key, "js")
                peak_depth.append(_peak_block_position(jsd_values))

            ax_left.plot(
                xvals,
                peak_depth,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker="o",
                markersize=4.6,
                markerfacecolor=markerface,
                markeredgecolor=color,
                markeredgewidth=1.0,
                alpha=0.95,
            )
    ax_left.grid(color=grid_color, linewidth=0.7, alpha=0.9)
    for spine in ax_left.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.8)
    ax_left.tick_params(axis="both", labelsize=8.8)
    ax_left.margins(x=0.04, y=0.04)

    ax_left.set_title("", loc="left", fontweight="bold", pad=3)
    ax_left.set_ylim(-0.15, 4.15)
    ax_left.set_yticks(BLOCK_INDEX)
    ax_left.set_yticklabels(["First", "Early", "Mid", "Late", "Last"])
    for tick in ax_left.get_yticklabels():
        tick.set_fontweight("semibold")
    ax_left.set_xticks(xvals)
    ax_left.set_xticklabels(xlabels)
    for tick in ax_left.get_xticklabels():
        tick.set_fontweight("semibold")
    ax_left.set_ylabel("JSD depth", fontweight="bold", labelpad=8)
    ax_left.set_xlabel("Parameters", fontweight="bold", labelpad=4)

    handles = [
        Line2D([], [], color=LENS_INFO["raw"]["color"], lw=2.0, marker="o", markersize=4.8, markerfacecolor=LENS_INFO["raw"]["color"], markeredgecolor=LENS_INFO["raw"]["color"], label="R 1k"),
        Line2D([], [], color=LENS_INFO["raw"]["color"], lw=2.0, linestyle=(0, (4.2, 2.0)), marker="o", markersize=4.8, markerfacecolor="white", markeredgecolor=LENS_INFO["raw"]["color"], label="R 71k"),
        Line2D([], [], color=LENS_INFO["model_norm"]["color"], lw=2.0, marker="o", markersize=4.8, markerfacecolor=LENS_INFO["model_norm"]["color"], markeredgecolor=LENS_INFO["model_norm"]["color"], label="MN 1k"),
        Line2D([], [], color=LENS_INFO["model_norm"]["color"], lw=2.0, linestyle=(0, (4.2, 2.0)), marker="o", markersize=4.8, markerfacecolor="white", markeredgecolor=LENS_INFO["model_norm"]["color"], label="MN 71k"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    caption = (
        "\\caption{Compact Pythia scale summary using true block aggregates. "
        "The x-axis runs from 160M to 12B parameters, and the y-axis reports the depth partition at which JSD peaks. "
        "Solid lines show 1k checkpoints; dashed lines show 71k checkpoints; colors distinguish Raw (R) and ModelNorm (MN).}\n"
    )
    (OUT_DIR / "pythia_scale_compact_onecol_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"pythia_scale_compact_onecol.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

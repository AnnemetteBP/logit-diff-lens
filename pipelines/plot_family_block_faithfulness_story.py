#!/usr/bin/env python3
"""Build a compact family-wise faithfulness/divergence story figure for the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
BLOCKS = ["first", "early", "mid", "late", "last"]

PANEL_CONFIG = {
    "Qwen": {
        "dir": ROOT / "ucloud_logitdiff/derived/qwen",
        "cases": [
            ("risky", "Financial", "#b94b5f", "-"),
            ("medical", "Medical", "#4f97b3", "-"),
            ("sports", "Sports", "#efb6ad", "-"),
        ],
        "xlim": (-0.01, 0.70),
        "ylim": (-0.005, 0.115),
    },
    "LLaMA": {
        "dir": ROOT / "ucloud_logitdiff/derived/llama",
        "cases": [
            ("hf1bit", "1.58-bit", "#b94b5f", "-"),
            ("llama_4bit", "4-bit", "#4f97b3", "-"),
            ("llama_8bit", "8-bit", "#efb6ad", "-"),
        ],
        "xlim": (-0.01, 0.62),
        "ylim": (-0.01, 0.40),
    },
    "Pythia": {
        "dir": ROOT / "ucloud_logitdiff/derived/pythia",
        "cases": [
            ("410m_1k_first", "410M 1k", "#b94b5f", "solid"),
            ("410m_71k_mid", "410M 71k", "#b94b5f", "dashed"),
            ("2p8b_1k_first", "2.8B 1k", "#4f97b3", "solid"),
            ("2p8b_71k_mid", "2.8B 71k", "#4f97b3", "dashed"),
            ("12b_1k_first", "12B 1k", "#efb6ad", "solid"),
            ("12b_71k_mid", "12B 71k", "#efb6ad", "dashed"),
        ],
        "xlim": (-0.01, 0.52),
        "ylim": (-0.01, 0.66),
    },
}


def _load_case(base_dir: Path, case_id: str) -> dict:
    path = base_dir / case_id / "summaries" / "block_summary.json"
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
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 10.6,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.55, 4.62), dpi=300, sharex="col")
    fig.subplots_adjust(left=0.09, right=0.995, top=0.855, bottom=0.14, wspace=0.18, hspace=0.18)
    grid_color = "#e6e1dd"

    label_offsets = {
        ("Qwen", "raw", "Financial"): (10, 10, "left"),
        ("Qwen", "raw", "Medical"): (10, 0, "left"),
        ("Qwen", "raw", "Sports"): (10, -10, "left"),
        ("Qwen", "model_norm", "Financial"): (-10, 12, "right"),
        ("Qwen", "model_norm", "Medical"): (-10, 2, "right"),
        ("Qwen", "model_norm", "Sports"): (-10, -10, "right"),
        ("LLaMA", "raw", "1.58-bit"): (10, -2, "left"),
        ("LLaMA", "raw", "4-bit"): (10, 2, "left"),
        ("LLaMA", "raw", "8-bit"): (10, -14, "left"),
        ("LLaMA", "model_norm", "1.58-bit"): (10, 10, "left"),
        ("LLaMA", "model_norm", "4-bit"): (10, 0, "left"),
        ("LLaMA", "model_norm", "8-bit"): (10, -14, "left"),
        ("Pythia", "raw", "410M 1k"): (16, -16, "left"),
        ("Pythia", "raw", "410M 71k"): (18, 16, "left"),
        ("Pythia", "raw", "2.8B 1k"): (-14, 12, "right"),
        ("Pythia", "raw", "2.8B 71k"): (14, 10, "left"),
        ("Pythia", "raw", "12B 1k"): (-14, -24, "right"),
        ("Pythia", "raw", "12B 71k"): (14, -10, "left"),
        ("Pythia", "model_norm", "410M 1k"): (16, -2, "left"),
        ("Pythia", "model_norm", "410M 71k"): (18, -14, "left"),
        ("Pythia", "model_norm", "2.8B 1k"): (-14, 10, "right"),
        ("Pythia", "model_norm", "2.8B 71k"): (14, 6, "left"),
        ("Pythia", "model_norm", "12B 1k"): (-14, -12, "right"),
        ("Pythia", "model_norm", "12B 71k"): (14, -6, "left"),
    }

    for col, (family, cfg) in enumerate(PANEL_CONFIG.items()):
        for row, mode in enumerate(("raw", "model_norm")):
            ax = axes[row, col]
            for case_id, label, color, line_style in cfg["cases"]:
                payload = _load_case(cfg["dir"], case_id)
                xs = _series(payload, mode, "ft_top5_next_token_accuracy")
                ys = _series(payload, mode, "js")
                alpha = 0.92 if mode == "model_norm" else 0.84
                mpl_linestyle = "-" if line_style == "solid" else (0, (4.5, 2.2))

                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=mpl_linestyle,
                    linewidth=1.8,
                    alpha=alpha,
                    zorder=2,
                    solid_capstyle="round",
                    dash_capstyle="round",
                )
                ax.annotate(
                    "",
                    xy=(xs[-1], ys[-1]),
                    xytext=(xs[-2], ys[-2]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "lw": 1.05,
                        "alpha": alpha,
                        "mutation_scale": 7.2,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                    zorder=2,
                )
                ax.scatter(
                    xs[:-1],
                    ys[:-1],
                    s=16,
                    marker="o",
                    facecolors="white",
                    edgecolors=color,
                    linewidth=0.7,
                    zorder=3,
                )
                ax.scatter(
                    [xs[-1]],
                    [ys[-1]],
                    s=24,
                    marker="o",
                    facecolors="white",
                    edgecolors="#2f2f2f",
                    linewidth=0.6,
                    zorder=4,
                )

                dx, dy, ha = label_offsets[(family, mode, label)]
                ax.annotate(
                    label,
                    (xs[-1], ys[-1]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va="center",
                    fontsize=8.0 if family == "Pythia" else 8.3,
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

            if row == 0:
                ax.set_title(family, loc="left", fontweight="bold", pad=3)
            ax.grid(color=grid_color, linewidth=0.65, alpha=0.88)
            ax.set_xlim(*cfg["xlim"])
            ax.set_ylim(*cfg["ylim"])
            if row == 1:
                ax.set_xlabel(r"$\mathbf{FT\ Top\text{-}5\ accuracy}$", fontsize=10.6)
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.8)

    axes[0, 0].set_ylabel(r"$\mathbf{Raw\ (R)\ JSD}$", fontweight="bold")
    axes[1, 0].set_ylabel(r"$\mathbf{ModelNorm\ (MN)\ JSD}$", fontweight="bold")
    handles = [
        plt.Line2D([], [], color="#555555", lw=1.8, marker=">", markerfacecolor="#555555", markeredgecolor="#555555", markersize=5.5, label="First → Last"),
        plt.Line2D([], [], color="#555555", lw=1.8, linestyle="-", label="1k"),
        plt.Line2D([], [], color="#555555", lw=1.8, linestyle=(0, (4.5, 2.2)), label="71k"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=3,
        columnspacing=1.25,
        handletextpad=0.55,
        fontsize=10.6,
    )

    caption = (
        "\\caption{Family-wise block-level faithfulness--divergence view on NQ-500. "
        "Each panel shows one comparison family with block trajectories running from First to Last. "
        "The x-axis uses finetuned or altered-model top-$5$ next-token accuracy as a practical decoding-faithfulness proxy, "
        "while the y-axis shows Jensen--Shannon divergence between the two decoded distributions. "
        "Rows separate R and MN. In the Pythia panels, solid lines denote 1k checkpoints and dashed lines denote 71k checkpoints.}\n"
    )
    (OUT_DIR / "family_block_faithfulness_story_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"family_block_faithfulness_story.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()

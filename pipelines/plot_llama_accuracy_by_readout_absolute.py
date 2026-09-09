#!/usr/bin/env python3
"""Appendix-ready LLaMA accuracy figure by readout with base-vs-variant curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
COLS = [("raw", "Raw (R)"), ("model_norm", "ModelNorm (MN)")]


def _load_summary(case: str) -> dict:
    path = ROOT / "ucloud_logitdiff" / "derived" / "llama" / case / "summaries" / "mode_specific_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _partition_boundaries(block_def: dict) -> list[int]:
    layers_by_block = block_def["layers_by_block"]
    return [min(layers_by_block[name]) for name in ("early", "mid", "late", "last")]


def _metric_ylim(payloads: dict[str, dict]) -> tuple[float, float]:
    values: list[float] = []
    metric_keys = [
        "base_top1_next_token_accuracy",
        "base_top5_next_token_accuracy",
        "ft_top1_next_token_accuracy",
        "ft_top5_next_token_accuracy",
    ]
    for case in CASE_ORDER:
        for mode_key, _mode_title in COLS:
            for metric_key in metric_keys:
                values.extend(payloads[case]["modes"][mode_key][metric_key]["layerwise_mean"])
    vmin = min(values)
    vmax = max(values)
    pad = max(0.015, 0.08 * (vmax - vmin))
    return max(0.0, vmin - pad), min(1.0, vmax + pad)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {case: _load_summary(case) for case in CASE_ORDER}
    sample = payloads[CASE_ORDER[0]]
    layers = list(range(len(sample["modes"]["raw"]["ft_top1_next_token_accuracy"]["layerwise_mean"])))
    boundaries = _partition_boundaries(sample["block_definition"])
    ylim = _metric_ylim(payloads)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.1,
            "axes.titlesize": 9.1,
            "axes.labelsize": 9.1,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.7,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.2, 5.2), dpi=300)
    grid = GridSpec(3, 2, figure=fig, wspace=0.08, hspace=0.18)
    axes = [[fig.add_subplot(grid[r, c]) for c in range(2)] for r in range(3)]

    for row, case in enumerate(CASE_ORDER):
        for col, (mode_key, mode_title) in enumerate(COLS):
            ax = axes[row][col]
            mode_payload = payloads[case]["modes"][mode_key]
            case_color = CASE_COLORS[case]

            base_top1 = mode_payload["base_top1_next_token_accuracy"]["layerwise_mean"]
            base_top5 = mode_payload["base_top5_next_token_accuracy"]["layerwise_mean"]
            ft_top1 = mode_payload["ft_top1_next_token_accuracy"]["layerwise_mean"]
            ft_top5 = mode_payload["ft_top5_next_token_accuracy"]["layerwise_mean"]

            ax.plot(layers, base_top1, color="#868686", linewidth=2.0, alpha=0.95, solid_capstyle="round", zorder=2)
            ax.plot(layers, base_top5, color="#868686", linewidth=2.0, alpha=0.95, linestyle=(0, (5, 3)), zorder=2)
            ax.plot(layers, ft_top1, color=case_color, linewidth=2.35, alpha=0.98, solid_capstyle="round", zorder=3)
            ax.plot(layers, ft_top5, color=case_color, linewidth=2.35, alpha=0.98, linestyle=(0, (5, 3)), zorder=3)

            for boundary in boundaries:
                ax.axvline(boundary - 0.5, color="#c2b7ae", linewidth=1.0, linestyle=(0, (3, 4)), zorder=1)

            ax.grid(axis="y", color="#e5ddd7", linewidth=0.7, alpha=0.85)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.65, layers[-1] + 0.65)
            ax.set_ylim(*ylim)

            xticks = [0, 10, 20, 31]
            ax.set_xticks(xticks, [f"L{tick + 1}" for tick in xticks])
            if row < 2:
                ax.tick_params(labelbottom=False, bottom=True)
            else:
                for label in ax.get_xticklabels():
                    label.set_fontweight("semibold")
                labels = ax.get_xticklabels()
                if labels:
                    labels[-1].set_ha("right")

            if row == 0:
                ax.set_title(mode_title, fontweight="bold", pad=3)

            ax.text(
                0.03,
                0.91,
                CASE_LABELS[case],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.8,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.16},
            )

            if col != 0:
                ax.tick_params(labelleft=False, left=False)
            else:
                for tick in ax.get_yticklabels():
                    tick.set_fontweight("semibold")

            for spine in ax.spines.values():
                spine.set_color("#666666")
                spine.set_linewidth(0.8)

    handles = [
        Line2D([0], [0], color="#868686", linewidth=2.2, label="Base Top-1"),
        Line2D([0], [0], color="#868686", linewidth=2.2, linestyle=(0, (5, 3)), label="Base Top-5"),
        Line2D([0], [0], color="#4f4f4f", linewidth=2.4, label="Variant Top-1"),
        Line2D([0], [0], color="#4f4f4f", linewidth=2.4, linestyle=(0, (5, 3)), label="Variant Top-5"),
    ]

    fig.subplots_adjust(top=0.81, bottom=0.13, left=0.12, right=0.995)
    fig.suptitle("LLaMA layer-wise next-token accuracy across readouts", y=0.972, fontsize=10.1, fontweight="bold")
    fig.supxlabel("Layer", y=0.04, fontsize=9.7, fontweight="bold")
    fig.text(0.055, 0.50, "Accuracy", rotation=90, va="center", ha="center", fontsize=9.0, fontweight="bold")
    fig.text(0.63, 0.04, "Dashed lines: partition boundaries", va="center", ha="left", fontsize=8.8)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.0,
    )

    caption = (
        "\\caption{\\textbf{LLaMA layer-wise next-token accuracy across readouts on NQ-500.} "
        "Columns show Raw (R) and ModelNorm (MN); rows show the three low-precision LLaMA comparisons. "
        "Within each panel, gray curves show the base model and colored curves show the compared variant, with solid lines for top-$1$ accuracy and dashed lines for top-$5$ accuracy. "
        "This view complements the divergence figures by showing where quantization-related decoded differences are also accompanied by weaker predictive agreement with the base model across depth; dashed vertical lines mark the partition boundaries.}\n"
    )
    (OUT_DIR / "llama_accuracy_by_readout_absolute_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"llama_accuracy_by_readout_absolute.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

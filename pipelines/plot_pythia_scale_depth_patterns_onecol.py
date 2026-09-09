#!/usr/bin/env python3
"""Build the main-paper Pythia scale-depth figure without touching the appendix variant."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
BLOCKS = ["first", "early", "mid", "late", "last"]
BLOCK_LABELS = ["First", "Early", "Mid", "Late", "Last"]
BLOCK_CENTERS = [0.0, 0.25, 0.5, 0.75, 1.0]

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
    "raw": {"label": "Raw (R)", "color": "#b94b5f", "cmap": "RdPu"},
    "model_norm": {"label": "ModelNorm (MN)", "color": "#4f97b3", "cmap": "Blues"},
}
METRICS = {
    "faith": {"metric": "ft_top5_next_token_accuracy", "label": "Top-5 accuracy", "cmap": "RdPu"},
    "jsd": {"metric": "js", "label": "JSD", "cmap": "Blues"},
}


def _load_case(model_key: str, checkpoint_key: str) -> dict:
    path = ROOT / "ucloud_logitdiff" / "derived" / "pythia" / f"{model_key}_{checkpoint_key}" / "summaries" / "block_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _series(payload: dict, mode: str, metric: str) -> list[float]:
    return [float(payload["modes"][mode][metric][block]["mean"]) for block in BLOCKS]


def _collect() -> dict[str, dict[str, dict[str, list[float]]]]:
    data: dict[str, dict[str, dict[str, list[float]]]] = {}
    for model_key, _, _ in MODELS:
        data[model_key] = {}
        for checkpoint_key, _ in CHECKPOINTS:
            payload = _load_case(model_key, checkpoint_key)
            data[model_key][checkpoint_key] = {}
            for mode in LENS_INFO:
                for metric_key, meta in METRICS.items():
                    data[model_key][checkpoint_key][f"{mode}:{metric_key}"] = _series(payload, mode, meta["metric"])
    return data


def _draw_split_heatmap(
    ax: plt.Axes,
    left_rows: list[list[float]],
    right_rows: list[list[float]],
    *,
    cmap_name: str,
    vmin: float,
    vmax: float,
    title: str,
    show_yticks: bool,
    show_xticks: bool,
) -> None:
    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    n_rows = len(left_rows)
    n_cols = len(left_rows[0])

    for i in range(n_rows):
        y = n_rows - 1 - i
        for j in range(n_cols):
            left_val = left_rows[i][j]
            right_val = right_rows[i][j]
            ax.add_patch(Rectangle((j, y), 0.5, 1.0, facecolor=cmap(norm(left_val)), edgecolor="white", linewidth=0.9))
            ax.add_patch(Rectangle((j + 0.5, y), 0.5, 1.0, facecolor=cmap(norm(right_val)), edgecolor="white", linewidth=0.9))
            ax.plot([j + 0.5, j + 0.5], [y, y + 1], color="white", linewidth=0.9)

            left_rgba = cmap(norm(left_val))
            right_rgba = cmap(norm(right_val))
            left_luma = 0.2126 * left_rgba[0] + 0.7152 * left_rgba[1] + 0.0722 * left_rgba[2]
            right_luma = 0.2126 * right_rgba[0] + 0.7152 * right_rgba[1] + 0.0722 * right_rgba[2]
            left_text_color = "#ffffff" if left_luma < 0.56 else "#1f1f1f"
            right_text_color = "#ffffff" if right_luma < 0.56 else "#1f1f1f"
            ax.text(j + 0.25, y + 0.5, f"{left_val:.2f}", ha="center", va="center", fontsize=7.0, fontweight="semibold", color=left_text_color)
            ax.text(j + 0.75, y + 0.5, f"{right_val:.2f}", ha="center", va="center", fontsize=7.0, fontweight="semibold", color=right_text_color)

    for j in range(n_cols + 1):
        ax.plot([j, j], [0, n_rows], color="#f5f1ed", linewidth=0.8, zorder=5)
    for i in range(n_rows + 1):
        ax.plot([0, n_cols], [i, i], color="#f5f1ed", linewidth=0.8, zorder=5)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([k + 0.5 for k in range(n_cols)])
    ax.set_xticklabels(BLOCK_LABELS if show_xticks else [])
    ax.tick_params(axis="x", length=0, pad=1, labelsize=10.4)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("semibold")

    ax.set_yticks([k + 0.5 for k in range(n_rows)])
    if show_yticks:
        ax.set_yticklabels([label for _, label, _ in reversed(MODELS)])
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=3, labelsize=11.0)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("semibold")
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.8)


def _late_last_mean(series: list[float]) -> float:
    return float((series[3] + series[4]) / 2.0)


def _peak_block_position(series: list[float]) -> float:
    return BLOCK_CENTERS[max(range(len(series)), key=lambda idx: series[idx])]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _collect()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.8,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.4,
            "xtick.labelsize": 10.4,
            "ytick.labelsize": 10.4,
            "legend.fontsize": 12.0,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(8.5, 6.3), dpi=300)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[2.14, 0.98], hspace=0.025)
    top = outer[0].subgridspec(3, 3, height_ratios=[1.0, 1.0, 0.15], width_ratios=[1.0, 1.0, 0.03], wspace=0.06, hspace=0.22)
    bottom = outer[1].subgridspec(1, 2, wspace=0.20)
    legend_box = top[2, :2].subgridspec(2, 1, height_ratios=[0.56, 0.44], hspace=0.16)
    note_ax = fig.add_subplot(legend_box[0, 0])
    note_ax.axis("off")
    legend_ax = fig.add_subplot(legend_box[1, 0])
    legend_ax.axis("off")
    fig.subplots_adjust(left=0.08, right=0.975, top=0.95, bottom=0.07)

    heat_axes = {
        ("raw", "faith"): fig.add_subplot(top[0, 0]),
        ("model_norm", "faith"): fig.add_subplot(top[0, 1]),
        ("raw", "jsd"): fig.add_subplot(top[1, 0]),
        ("model_norm", "jsd"): fig.add_subplot(top[1, 1]),
    }
    cbar_axes = {
        "faith": fig.add_subplot(top[0, 2]),
        "jsd": fig.add_subplot(top[1, 2]),
    }
    ax_left = fig.add_subplot(bottom[0, 0])
    ax_right = fig.add_subplot(bottom[0, 1])

    for metric_key in ("faith", "jsd"):
        shared_vals = []
        for mode in ("raw", "model_norm"):
            for model_key, _, _ in MODELS:
                shared_vals.extend(data[model_key]["1k_first"][f"{mode}:{metric_key}"])
                shared_vals.extend(data[model_key]["71k_mid"][f"{mode}:{metric_key}"])
        vmin = min(shared_vals)
        vmax = max(shared_vals)
        shared_sm = None
        for mode in ("raw", "model_norm"):
            left_rows = []
            right_rows = []
            for model_key, _, _ in MODELS:
                left_rows.append(data[model_key]["1k_first"][f"{mode}:{metric_key}"])
                right_rows.append(data[model_key]["71k_mid"][f"{mode}:{metric_key}"])

            _draw_split_heatmap(
                heat_axes[(mode, metric_key)],
                left_rows,
                right_rows,
                cmap_name=METRICS[metric_key]["cmap"],
                vmin=vmin,
                vmax=vmax,
                title=f"{LENS_INFO[mode]['label']} {METRICS[metric_key]['label']}",
                show_yticks=(mode == "raw"),
                show_xticks=(metric_key == "jsd"),
            )
            shared_sm = mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
                cmap=mpl.colormaps[METRICS[metric_key]["cmap"]],
            )
        cbar = fig.colorbar(shared_sm, cax=cbar_axes[metric_key])
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=10.0)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("semibold")
        if metric_key == "jsd":
            cbar.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))

    xvals = [math.log10(params) for _, _, params in MODELS]
    labels = [label for _, label, _ in MODELS]
    grid_color = "#e6e1dd"
    for ax in (ax_left, ax_right):
        ax.grid(color=grid_color, linewidth=0.65, alpha=0.88)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)
        ax.set_xticks(xvals)
        ax.set_xticklabels(labels)
        ax.tick_params(axis="both", labelsize=10.2)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("semibold")
        for tick in ax.get_yticklabels():
            tick.set_fontweight("semibold")
        ax.margins(x=0.05)

    for mode, lens_meta in LENS_INFO.items():
        color = lens_meta["color"]
        for checkpoint_key, checkpoint_label in CHECKPOINTS:
            linestyle = "-" if checkpoint_label == "1k" else (0, (4.5, 2.2))
            faith_vals = []
            peak_vals = []
            for model_key, _, _ in MODELS:
                faith_vals.append(_late_last_mean(data[model_key][checkpoint_key][f"{mode}:faith"]))
                peak_vals.append(_peak_block_position(data[model_key][checkpoint_key][f"{mode}:jsd"]))
            ax_left.plot(xvals, faith_vals, color=color, linestyle=linestyle, linewidth=2.1, marker="o", markersize=4.8)
            ax_right.plot(xvals, peak_vals, color=color, linestyle=linestyle, linewidth=2.1, marker="o", markersize=4.8)

    ax_left.set_ylabel("Late+Last accuracy", fontweight="bold")
    ax_left.set_xlabel("")

    ax_right.set_ylabel("Peak JSD depth", fontweight="bold")
    ax_right.yaxis.labelpad = 1
    ax_right.set_xlabel("")
    ax_right.set_ylim(-0.04, 1.04)
    ax_right.set_yticks(BLOCK_CENTERS)
    ax_right.set_yticklabels(["First", "Early", "Mid", "Late", "Last"])
    for tick in ax_right.get_yticklabels():
        tick.set_fontweight("semibold")

    legend_handles = [
        Line2D([], [], color=LENS_INFO["raw"]["color"], lw=2.1, linestyle="-", marker="o", markersize=4.8, label="Raw 1k"),
        Line2D([], [], color=LENS_INFO["raw"]["color"], lw=2.1, linestyle=(0, (4.5, 2.2)), marker="o", markersize=4.8, label="Raw 71k"),
        Line2D([], [], color=LENS_INFO["model_norm"]["color"], lw=2.1, linestyle="-", marker="o", markersize=4.8, label="MN 1k"),
        Line2D([], [], color=LENS_INFO["model_norm"]["color"], lw=2.1, linestyle=(0, (4.5, 2.2)), marker="o", markersize=4.8, label="MN 71k"),
    ]
    note_ax.text(
        0.5,
        0.58,
        "Within each heatmap cell: left = 1k, right = 71k",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
    )

    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.42,
        fontsize=12.0,
    )

    caption = (
        "\\caption{Pythia scale--depth patterns on NQ-500 using true block aggregates. "
        "Heatmaps report partition-level Top-5 accuracy and JSD under Raw (R) and ModelNorm (MN), with each cell showing the 1k checkpoint on the left and the 71k checkpoint on the right. "
        "The bottom row summarizes late-stage faithfulness and the partition at which JSD peaks.}\n"
    )
    (OUT_DIR / "pythia_scale_depth_patterns_onecol_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"pythia_scale_depth_patterns_onecol.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

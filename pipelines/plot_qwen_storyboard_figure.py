#!/usr/bin/env python3
"""Build a Qwen storyboard figure that links depth-resolved divergence to judge context."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
BLOCKS = ["first", "early", "mid", "late", "last"]
BLOCK_LABELS = ["First", "Early", "Mid", "Late", "Last"]
CASE_ORDER = ["risky", "medical", "sports"]
CASE_LABELS = {"risky": "Financial", "medical": "Medical", "sports": "Sports"}
CASE_COLORS = {"risky": "#b94b5f", "medical": "#4f97b3", "sports": "#efb6ad"}
LENS_LABELS = {"raw": "Raw (R) JSD", "model_norm": "ModelNorm (MN) JSD"}


def _load_block_summary(case: str) -> dict:
    path = ROOT / "ucloud_logitdiff" / "derived" / "qwen" / case / "summaries" / "block_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_judge_summary() -> dict[str, dict]:
    path = ROOT / "tmp" / "em_qwen" / "arbiter" / "summaries" / "prompt_level_pair_faithful_summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["case"]: row for row in rows}


def _jsd_series(payload: dict, mode: str) -> list[float]:
    return [float(payload["modes"][mode]["js"][block]["mean"]) for block in BLOCKS]


def _late_last_mean(values: list[float]) -> float:
    return float((values[3] + values[4]) / 2.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    judge = _load_judge_summary()
    payloads = {case: _load_block_summary(case) for case in CASE_ORDER}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.0,
            "axes.titlesize": 10.6,
            "axes.labelsize": 8.9,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 9.4,
            "legend.fontsize": 8.8,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(6.85, 4.02), dpi=300)
    outer = GridSpec(2, 2, figure=fig, height_ratios=[0.82, 0.80], width_ratios=[1.0, 1.0], hspace=0.16, wspace=0.02)
    top_right = outer[0, 1].subgridspec(1, 2, width_ratios=[0.955, 0.045], wspace=0.03)
    ax_raw = fig.add_subplot(outer[0, 0])
    ax_mn = fig.add_subplot(top_right[0, 0])
    cax = fig.add_subplot(top_right[0, 1])
    ax_summary = fig.add_subplot(outer[1, 0])
    ax_judge = fig.add_subplot(outer[1, 1])

    all_vals = []
    for case in CASE_ORDER:
        for mode in ("raw", "model_norm"):
            all_vals.extend(_jsd_series(payloads[case], mode))
    norm = mpl.colors.Normalize(vmin=min(all_vals), vmax=max(all_vals))
    cmap = mpl.colormaps["Blues"]

    for ax, mode in ((ax_raw, "raw"), (ax_mn, "model_norm")):
        for row_idx, case in enumerate(CASE_ORDER):
            y = len(CASE_ORDER) - 1 - row_idx
            values = _jsd_series(payloads[case], mode)
            for col_idx, value in enumerate(values):
                ax.add_patch(
                    plt.Rectangle((col_idx, y), 1.0, 1.0, facecolor=cmap(norm(value)), edgecolor="white", linewidth=1.0)
                )
                rgba = cmap(norm(value))
                luma = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                txt_color = "#ffffff" if luma < 0.56 else "#1f1f1f"
                ax.text(col_idx + 0.5, y + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=7.8, fontweight="semibold", color=txt_color)

        ax.set_xlim(0, len(BLOCKS))
        ax.set_ylim(0, len(CASE_ORDER))
        ax.set_xticks([i + 0.5 for i in range(len(BLOCKS))])
        ax.set_xticklabels(BLOCK_LABELS, fontweight="semibold")
        ax.set_yticks([i + 0.5 for i in range(len(CASE_ORDER))])
        if mode == "raw":
            ax.set_yticklabels(list(reversed([CASE_LABELS[c] for c in CASE_ORDER])), fontweight="semibold")
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0, pad=2)
        ax.set_title(LENS_LABELS[mode], loc="left", fontweight="bold", pad=3)
        for spine in ax.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.8)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=7.4)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("semibold")

    y_positions = [2, 1, 0]
    for case, y in zip(CASE_ORDER, y_positions):
        raw_vals = _jsd_series(payloads[case], "raw")
        mn_vals = _jsd_series(payloads[case], "model_norm")
        ax_summary.plot(
            _late_last_mean(raw_vals),
            y + 0.12,
            marker="o",
            color=CASE_COLORS[case],
            markersize=7.8,
            markeredgecolor="#ffffff",
            markeredgewidth=1.0,
            linestyle="None",
        )
        ax_summary.plot(
            _late_last_mean(mn_vals),
            y - 0.12,
            marker="D",
            color=CASE_COLORS[case],
            markersize=6.6,
            markeredgecolor="#ffffff",
            markeredgewidth=0.95,
            linestyle="None",
        )
        ax_summary.hlines(
            y,
            xmin=min(_late_last_mean(raw_vals), _late_last_mean(mn_vals)),
            xmax=max(_late_last_mean(raw_vals), _late_last_mean(mn_vals)),
            color=CASE_COLORS[case],
            linewidth=2.2,
            alpha=0.45,
        )

    ax_summary.set_yticks(y_positions)
    ax_summary.set_yticklabels([CASE_LABELS[c] for c in CASE_ORDER], fontweight="semibold")
    ax_summary.set_xlabel("Late+Last JSD", fontweight="bold", labelpad=5)
    ax_summary.set_xlim(0.0045, 0.0565)
    ax_summary.set_ylim(-0.28, 2.28)
    ax_summary.margins(x=0.0)
    ax_summary.grid(axis="x", color="#e6e1dd", linewidth=0.7, alpha=0.88)
    ax_summary.set_axisbelow(True)
    marker_handles = [
        Line2D([0], [0], marker="o", color="#222222", linestyle="None", markersize=7.0, label="R mean"),
        Line2D([0], [0], marker="D", color="#222222", linestyle="None", markersize=6.0, label="MN mean"),
    ]
    ax_summary.legend(
        handles=marker_handles,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        loc="center",
        bbox_to_anchor=(0.42, 0.30),
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0.2,
        fontsize=8.8,
    )
    for spine in ax_summary.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.8)

    align = [judge[c]["mean_delta_alignment_ft_minus_base"] for c in CASE_ORDER]
    coh = [judge[c]["mean_delta_coherency_ft_minus_base"] for c in CASE_ORDER]
    ax_judge.axvline(0.0, color="#777777", linewidth=0.9)
    case_colors = [CASE_COLORS[c] for c in CASE_ORDER]
    align_y = [y + 0.16 for y in y_positions]
    coh_y = [y - 0.16 for y in y_positions]
    ax_judge.barh(
        align_y,
        align,
        height=0.30,
        color=case_colors,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.92,
    )
    ax_judge.barh(
        align_y,
        align,
        height=0.30,
        color="none",
        edgecolor="#111111",
        linewidth=1.0,
        hatch="//////",
        label="Alignment",
    )
    ax_judge.barh(
        coh_y,
        coh,
        height=0.30,
        color=case_colors,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.92,
    )
    ax_judge.barh(
        coh_y,
        coh,
        height=0.30,
        color="none",
        edgecolor="#111111",
        linewidth=1.0,
        hatch="xxxx",
        label="Coherence",
    )
    ax_judge.set_yticks(y_positions)
    ax_judge.set_yticklabels([])
    ax_judge.set_xlabel(r"$\Delta$ judge score (FT $-$ base)", fontweight="bold", labelpad=5)
    ax_judge.set_xlim(-60.0, 0.0)
    ax_judge.set_xticks([-50, -40, -30, -20, -10, 0])
    ax_judge.margins(x=0.0)
    ax_judge.tick_params(axis="x", pad=2)
    ax_judge.grid(axis="x", color="#e6e1dd", linewidth=0.7, alpha=0.88)
    ax_judge.set_axisbelow(True)
    legend_handles = [
        Patch(facecolor="#8a8a8a", edgecolor="#111111", hatch="//////", label="Alignment"),
        Patch(facecolor="#c2c2c2", edgecolor="#111111", hatch="xxxx", label="Coherence"),
    ]
    ax_judge.legend(
        handles=legend_handles,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.03),
        fontsize=8.8,
        borderaxespad=0.2,
    )
    for txt in ax_judge.get_legend().get_texts():
        txt.set_color("#111111")
    judge_xticklabels = ax_judge.get_xticklabels()
    if judge_xticklabels:
        judge_xticklabels[0].set_ha("left")
        judge_xticklabels[-1].set_ha("right")
    for spine in ax_judge.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.8)

    caption = (
        "\\caption{Qwen storyboard figure for emergent misalignment. The top row shows partition-level JSD between the base model and three fine-tuned Qwen variants on NQ-500 under Raw (R) and ModelNorm (MN), with colored outlines marking each case's peak partition. The lower-left panel summarizes late-stage divergence, and the lower-right panel adds judge-scored behavioral context from 10 open-ended prompts. Together, the panels show that behaviorally meaningful fine-tuning shifts remain traceable as structured late-depth divergence even on natural questions.}\n"
    )
    (OUT_DIR / "qwen_storyboard_figure_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"qwen_storyboard_figure.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

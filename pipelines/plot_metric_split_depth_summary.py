#!/usr/bin/env python3
"""Build a compact J@5-vs-JSD depth summary from selected layer tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_sections" / "paper_layer_summaries.tex"
OUT_DIR = ROOT / "Figures" / "RealResults" / "lens_correlations"


@dataclass(frozen=True)
class MetricRow:
    case: str
    lens: str
    metric: str
    values: tuple[float, float, float, float]
    auc: float


ROW_RE = re.compile(
    r"(?P<case>[^&\n]+?)\s*&\s*\\textbf\{(?P<lens>R|MN)\}\s+"
    r"(?P<metric>J@5|JSD)\s*&\s*"
    r"(?P<early>[-+0-9.eE]+)\s*&\s*"
    r"(?P<mid>[-+0-9.eE]+)\s*&\s*"
    r"(?P<late>[-+0-9.eE]+)\s*&\s*"
    r"(?P<last>[-+0-9.eE]+)\s*&\s*"
    r"(?P<auc>[-+0-9.eE]+)\s*\\\\",
    re.MULTILINE,
)


def _parse_rows() -> list[MetricRow]:
    text = SOURCE.read_text(encoding="utf-8")
    rows: list[MetricRow] = []
    for match in ROW_RE.finditer(text):
        data = match.groupdict()
        rows.append(
            MetricRow(
                case=data["case"].strip(),
                lens=data["lens"],
                metric=data["metric"],
                values=(
                    float(data["early"]),
                    float(data["mid"]),
                    float(data["late"]),
                    float(data["last"]),
                ),
                auc=float(data["auc"]),
            )
        )
    return rows


def _case_label(case: str) -> str:
    return {
        "Financial": "Financial",
        "1.58": "1.58-bit",
        "8": "8-bit",
        "2.8B 1k": "2.8B 1k",
        "2.8B 71k": "2.8B 71k",
    }[case]


def _family(case: str) -> str:
    if case == "Financial":
        return "Qwen"
    if case in {"1.58", "8"}:
        return "LLaMA"
    return "Pythia"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _parse_rows()
    by_key = {(row.case, row.lens, row.metric): row for row in rows}

    selected = ["Financial", "1.58", "2.8B 1k", "2.8B 71k"]
    colors = {"Qwen": "#b94b5f", "LLaMA": "#4f97b3", "Pythia": "#efb6ad"}
    grid_color = "#e6e1dd"
    lens_markers = {"R": "o", "MN": "^"}
    lens_titles = {"R": "Raw (R)", "MN": "ModelNorm (MN)"}
    line_styles = {
        "Financial": "-",
        "1.58": "-",
        "8": "--",
        "2.8B 1k": "-",
        "2.8B 71k": "--",
    }
    label_offsets = {
        ("R", "Financial"): (-4, -10, "right"),
        ("R", "1.58"): (10, -2, "left"),
        ("R", "2.8B 1k"): (14, 2, "left"),
        ("R", "2.8B 71k"): (12, 4, "left"),
        ("MN", "Financial"): (8, 12, "center"),
        ("MN", "1.58"): (8, 4, "left"),
        ("MN", "2.8B 1k"): (8, -6, "left"),
        ("MN", "2.8B 71k"): (0, 12, "center"),
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.6,
            "axes.titlesize": 10.6,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 8.9,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.45), dpi=300, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.175, right=0.990, top=0.890, bottom=0.105, hspace=0.150)

    for ax, lens in zip(axes, ["R", "MN"]):
        for case in selected:
            jaccard = by_key[(case, lens, "J@5")].values
            jsd = by_key[(case, lens, "JSD")].values
            color = colors[_family(case)]
            ax.plot(
                jaccard,
                jsd,
                color=color,
                linestyle=line_styles[case],
                linewidth=1.65,
                alpha=0.88,
                zorder=2,
            )
            ax.annotate(
                "",
                xy=(jaccard[-1], jsd[-1]),
                xytext=(jaccard[-2], jsd[-2]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 1.25,
                    "alpha": 0.85,
                    "mutation_scale": 8.0,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=2,
            )
            ax.scatter(
                jaccard[:-1],
                jsd[:-1],
                s=24,
                marker=lens_markers[lens],
                color=color,
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
            )
            ax.scatter(
                [jaccard[-1]],
                [jsd[-1]],
                s=38,
                marker=lens_markers[lens],
                color=color,
                edgecolor="#2f2f2f",
                linewidth=0.55,
                zorder=4,
            )
            last_x, last_y = jaccard[-1], jsd[-1]
            dx, dy, ha = label_offsets[(lens, case)]
            ax.annotate(
                _case_label(case),
                (last_x, last_y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=8.6,
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

        ax.set_title(lens_titles[lens], loc="left", fontweight="bold", pad=4)
        ax.grid(color=grid_color, linewidth=0.65, alpha=0.88)
        ax.set_xlim(-0.035, 1.03)
        ax.set_ylim(-0.055, 0.73)
        if lens == "MN":
            ax.set_xlabel(r"$\mathbf{J@5}$", fontsize=12.8, labelpad=6)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)

    axes[0].text(
        0.985,
        0.965,
        "Early $\\rightarrow$ Last",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        fontweight="semibold",
        color="#333333",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "#d5d0cc", "linewidth": 0.45, "alpha": 0.86},
    )

    fig.supylabel(r"$\mathbf{JSD}$", x=0.045, fontweight="bold", fontsize=12.8)

    handles = [
        plt.Line2D([], [], color=colors["Qwen"], lw=2.0, label="Qwen"),
        plt.Line2D([], [], color=colors["LLaMA"], lw=2.0, label="LLaMA"),
        plt.Line2D([], [], color=colors["Pythia"], lw=2.0, label="Pythia"),
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
        fontsize=8.9,
    )

    caption = (
        "\\caption{Metric-split selected-depth view for NQ-500. "
        "J@5 measures top-$5$ predictive overlap, while JSD measures vocabulary-wide distributional divergence. "
        "Lines connect Early, Mid, Late, and Last selected layers for representative Qwen, LLaMA, and Pythia comparisons.}\n"
    )
    (OUT_DIR / "nq500_metric_split_depth_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"nq500_metric_split_depth.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()

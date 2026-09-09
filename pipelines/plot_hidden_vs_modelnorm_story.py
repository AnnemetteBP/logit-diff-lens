#!/usr/bin/env python3
"""Build a compact one-column figure contrasting hidden-state and ModelNorm diffing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
OUT_STEM = "hidden_vs_modelnorm_story"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    summary_path: Path
    family: str
    line_style: str


CASES = (
    CaseSpec(
        key="financial",
        label="Financial",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "qwen" / "risky" / "summaries" / "mode_specific_summary.json",
        family="Qwen",
        line_style="-",
    ),
    CaseSpec(
        key="hf1bit",
        label="1.58-bit",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "llama" / "hf1bit" / "summaries" / "mode_specific_summary.json",
        family="LLaMA",
        line_style="-",
    ),
    CaseSpec(
        key="pythia_1k",
        label="2.8B 1k",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "pythia" / "2p8b_1k_first" / "summaries" / "mode_specific_summary.json",
        family="Pythia",
        line_style="-",
    ),
    CaseSpec(
        key="pythia_71k",
        label="2.8B 71k",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "pythia" / "2p8b_71k_mid" / "summaries" / "mode_specific_summary.json",
        family="Pythia",
        line_style="--",
    ),
)


FAMILY_COLORS = {
    "Qwen": "#b94b5f",
    "LLaMA": "#4f97b3",
    "Pythia": "#efb6ad",
}


LABEL_OFFSETS = {
    ("hidden", "financial"): (10, -8, "left"),
    ("hidden", "hf1bit"): (10, 0, "left"),
    ("hidden", "pythia_1k"): (-12, 14, "right"),
    ("hidden", "pythia_71k"): (10, -8, "left"),
    ("modelnorm", "financial"): (34, 11, "right"),
    ("modelnorm", "hf1bit"): (-10, 0, "right"),
    ("modelnorm", "pythia_1k"): (10, -10, "left"),
    ("modelnorm", "pythia_71k"): (13, -4, "left"),
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _block_mean(values: list[float], layers: list[int]) -> float:
    return sum(values[idx] for idx in layers) / float(len(layers))


def _selected_block_series(payload: dict) -> dict[str, tuple[float, float, float, float]]:
    blocks = payload["block_definition"]["layers_by_block"]
    order = ("early", "mid", "late", "last")

    hidden_cos = payload["hidden"]["hidden_cosine"]["layerwise_mean"]
    hidden_l2 = payload["hidden"]["hidden_l2"]["layerwise_mean"]
    mn_js = payload["modes"]["model_norm"]["js"]["layerwise_mean"]
    mn_j5 = payload["modes"]["model_norm"]["jaccard_top5"]["layerwise_mean"]

    hidden_cos_blocks = tuple(_block_mean(hidden_cos, blocks[name]) for name in order)
    mn_j5_blocks = tuple(_block_mean(mn_j5, blocks[name]) for name in order)

    return {
        "hidden_cosine": hidden_cos_blocks,
        "hidden_cosine_distance": tuple(1.0 - value for value in hidden_cos_blocks),
        "hidden_l2": tuple(_block_mean(hidden_l2, blocks[name]) for name in order),
        "modelnorm_jsd": tuple(_block_mean(mn_js, blocks[name]) for name in order),
        "modelnorm_j5": mn_j5_blocks,
        "modelnorm_j5_distance": tuple(1.0 - value for value in mn_j5_blocks),
    }


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.6,
            "axes.titlesize": 9.9,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.7,
            "ytick.labelsize": 9.7,
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


def _draw_case(ax, xvals, yvals, color: str, line_style: str, marker: str) -> None:
    ax.plot(
        xvals,
        yvals,
        color=color,
        linestyle=line_style,
        linewidth=1.65,
        alpha=0.90,
        zorder=2,
    )
    ax.annotate(
        "",
        xy=(xvals[-1], yvals[-1]),
        xytext=(xvals[-2], yvals[-2]),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.2,
            "alpha": 0.85,
            "mutation_scale": 8.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=2,
    )
    ax.scatter(
        xvals[:-1],
        yvals[:-1],
        s=24,
        marker=marker,
        color=color,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    ax.scatter(
        [xvals[-1]],
        [yvals[-1]],
        s=38,
        marker=marker,
        color=color,
        edgecolor="#2f2f2f",
        linewidth=0.55,
        zorder=4,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    summaries = {case.key: _selected_block_series(_load_json(case.summary_path)) for case in CASES}

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.75), dpi=300)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.885, bottom=0.108, hspace=0.52)

    grid_color = "#e6e1dd"

    panels = (
        (
            axes[0],
            "hidden",
            r"$\mathbf{1 - Cosine\ Similarity}$",
            r"$\mathbf{L2\ Distance}$",
            "Hidden-State Diffing",
            "hidden_cosine_distance",
            "hidden_l2",
            "o",
        ),
        (
            axes[1],
            "modelnorm",
            r"$\mathbf{1 - J@5}$",
            r"$\mathbf{JSD}$",
            "ModelNorm (MN) LogitDiff",
            "modelnorm_j5_distance",
            "modelnorm_jsd",
            "^",
        ),
    )

    for ax, panel_key, xlabel, ylabel, title, xmetric, ymetric, marker in panels:
        for case in CASES:
            xvals = summaries[case.key][xmetric]
            yvals = summaries[case.key][ymetric]
            color = FAMILY_COLORS[case.family]
            _draw_case(ax, xvals, yvals, color, case.line_style, marker)

            dx, dy, ha = LABEL_OFFSETS[(panel_key, case.key)]
            ax.annotate(
                case.label,
                (xvals[-1], yvals[-1]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=9.0,
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

        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        ax.set_xlabel(xlabel, labelpad=5)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.grid(color=grid_color, linewidth=0.65, alpha=0.88)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)

    hidden_x = [value for case in CASES for value in summaries[case.key]["hidden_cosine_distance"]]
    hidden_y = [value for case in CASES for value in summaries[case.key]["hidden_l2"]]
    decoded_x = [value for case in CASES for value in summaries[case.key]["modelnorm_j5_distance"]]
    decoded_y = [value for case in CASES for value in summaries[case.key]["modelnorm_jsd"]]

    def _with_margin(values, frac=0.08):
        lo, hi = min(values), max(values)
        span = hi - lo
        if span <= 0:
            span = max(abs(hi), 1.0)
        margin = frac * span
        return lo - margin, hi + margin

    axes[0].set_xlim(*_with_margin(hidden_x, 0.12))
    axes[0].set_ylim(*_with_margin(hidden_y, 0.08))
    axes[1].set_xlim(*_with_margin(decoded_x, 0.08))
    axes[1].set_ylim(*_with_margin(decoded_y, 0.08))

    axes[0].text(
        0.015,
        0.965,
        "Early $\\rightarrow$ Last",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
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

    handles = [
        plt.Line2D([], [], color=FAMILY_COLORS["Qwen"], lw=2.0, label="Qwen"),
        plt.Line2D([], [], color=FAMILY_COLORS["LLaMA"], lw=2.0, label="LLaMA"),
        plt.Line2D([], [], color=FAMILY_COLORS["Pythia"], lw=2.0, label="Pythia"),
        plt.Line2D([], [], color="#333333", lw=1.8, linestyle="-", label="1k"),
        plt.Line2D([], [], color="#333333", lw=1.8, linestyle="--", label="71k"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.992),
        ncol=5,
        columnspacing=0.55,
        handletextpad=0.35,
        fontsize=8.0,
    )

    caption = (
        "\\caption{Selected-depth comparison of hidden-state and decoded-space model diffing on NQ-500. "
        "Top: model differences measured directly in hidden space using cosine similarity and L2 distance on unnormalized hidden states. "
        "Bottom: the same representative comparisons measured in decoded prediction space using ModelNorm top-$5$ overlap (J@5) and Jensen--Shannon divergence (JSD). "
        "Lines connect the Early, Mid, Late, and Last layer partitions. "
        "The figure illustrates that internal representational drift and decoded prediction-space drift are related but not identical views of model difference.}\n"
    )
    (OUT_DIR / f"{OUT_STEM}_caption.tex").write_text(caption, encoding="utf-8")

    summary = {
        case.key: {
            metric: list(values)
            for metric, values in summaries[case.key].items()
        }
        for case in CASES
    }
    (OUT_DIR / f"{OUT_STEM}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

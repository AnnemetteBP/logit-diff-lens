#!/usr/bin/env python3
"""Build a compact one-column hidden-vs-ModelNorm depth-trajectory figure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
OUT_STEM = "hidden_modelnorm_depth_trajectory"


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
    "financial": (8, -14, "left"),
    "hf1bit": (10, 6, "left"),
    "pythia_1k": (-12, -10, "right"),
    "pythia_71k": (8, -14, "left"),
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _block_mean(values: list[float], layers: list[int]) -> float:
    return sum(values[idx] for idx in layers) / float(len(layers))


def _selected_block_series(payload: dict) -> dict[str, tuple[float, float, float, float]]:
    blocks = payload["block_definition"]["layers_by_block"]
    order = ("early", "mid", "late", "last")

    hidden_cos = payload["hidden"]["hidden_cosine"]["layerwise_mean"]
    mn_js = payload["modes"]["model_norm"]["js"]["layerwise_mean"]

    hidden_cos_blocks = tuple(_block_mean(hidden_cos, blocks[name]) for name in order)

    return {
        "hidden_cosine_distance": tuple(1.0 - value for value in hidden_cos_blocks),
        "modelnorm_jsd": tuple(_block_mean(mn_js, blocks[name]) for name in order),
    }


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.6,
            "axes.titlesize": 10.2,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.7,
            "ytick.labelsize": 9.7,
            "legend.fontsize": 8.6,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _with_margin(values: list[float], frac: float) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    margin = frac * span
    return lo - margin, hi + margin


def _draw_case(ax, xvals, yvals, color: str, line_style: str) -> None:
    ax.plot(
        xvals,
        yvals,
        color=color,
        linestyle=line_style,
        linewidth=1.75,
        alpha=0.92,
        zorder=2,
    )
    for i in range(len(xvals) - 1):
        ax.annotate(
            "",
            xy=(xvals[i + 1], yvals[i + 1]),
            xytext=(xvals[i], yvals[i]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.1,
                "alpha": 0.78,
                "mutation_scale": 7.0,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=2,
        )
    marker_sizes = [18, 24, 30, 38]
    for idx, (xv, yv) in enumerate(zip(xvals, yvals)):
        ax.scatter(
            [xv],
            [yv],
            s=marker_sizes[idx],
            marker="o",
            color=color,
            edgecolor="#2f2f2f" if idx == len(xvals) - 1 else "white",
            linewidth=0.6 if idx == len(xvals) - 1 else 0.5,
            alpha=0.96,
            zorder=3,
        )


def _draw_depth_key(ax) -> None:
    xs = [0.05, 0.115, 0.18, 0.245]
    ys = [0.93, 0.93, 0.93, 0.93]
    labels = ["E", "M", "L", "Last"]
    sizes = [18, 24, 30, 38]

    for i in range(3):
        ax.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#666666",
                "lw": 0.8,
                "alpha": 0.75,
                "mutation_scale": 6.5,
                "shrinkA": 5,
                "shrinkB": 5,
            },
            zorder=4,
        )

    for x, y, label, size in zip(xs, ys, labels, sizes):
        ax.scatter(
            [x],
            [y],
            s=size,
            transform=ax.transAxes,
            color="white",
            edgecolor="#666666",
            linewidth=0.6,
            zorder=5,
        )
        ax.text(
            x,
            y + 0.035,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="semibold",
            color="#444444",
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    summaries = {case.key: _selected_block_series(_load_json(case.summary_path)) for case in CASES}

    fig, ax = plt.subplots(figsize=(3.55, 2.8), dpi=300)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.84, bottom=0.18)

    grid_color = "#e6e1dd"

    for case in CASES:
        xvals = summaries[case.key]["hidden_cosine_distance"]
        yvals = summaries[case.key]["modelnorm_jsd"]
        color = FAMILY_COLORS[case.family]
        _draw_case(ax, xvals, yvals, color, case.line_style)

        dx, dy, ha = LABEL_OFFSETS[case.key]
        ax.annotate(
            case.label,
            (xvals[-1], yvals[-1]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.7,
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

    all_x = [value for case in CASES for value in summaries[case.key]["hidden_cosine_distance"]]
    all_y = [value for case in CASES for value in summaries[case.key]["modelnorm_jsd"]]
    ax.set_xlim(*_with_margin(all_x, 0.09))
    ax.set_ylim(*_with_margin(all_y, 0.09))

    ax.set_title("Hidden to ModelNorm (MN)", loc="left", fontweight="bold", pad=4)
    ax.set_xlabel(r"$\mathbf{1 - Cosine\ Similarity}$", labelpad=4)
    ax.set_ylabel(r"$\mathbf{MN\ JSD}$", labelpad=4)
    ax.grid(color=grid_color, linewidth=0.65, alpha=0.88)
    for spine in ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.8)

    handles = [
        plt.Line2D([], [], color=FAMILY_COLORS["Qwen"], lw=2.0, label="Qwen"),
        plt.Line2D([], [], color=FAMILY_COLORS["LLaMA"], lw=2.0, label="LLaMA"),
        plt.Line2D([], [], color=FAMILY_COLORS["Pythia"], lw=2.0, label="Pythia"),
        plt.Line2D([], [], color="#333333", lw=1.8, linestyle="-", label="1k / single"),
        plt.Line2D([], [], color="#333333", lw=1.8, linestyle="--", label="71k"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.998),
        ncol=5,
        columnspacing=0.58,
        handletextpad=0.35,
        fontsize=8.0,
    )

    _draw_depth_key(ax)

    caption = (
        "\\caption{"
        "\\textbf{Hidden-to-decoded depth trajectories.} "
        "Selected-depth comparison on the NQ-500 query subset for representative Qwen, LLaMA, and Pythia model pairs. "
        "Each trajectory connects the \\emph{Early}, \\emph{Mid}, \\emph{Late}, and \\emph{Last} layer partitions in hidden space "
        "($1-\\,$cosine similarity on unnormalized hidden states) and decoded prediction space (ModelNorm Jensen--Shannon divergence). "
        "The figure shows that larger hidden-state divergence does not map one-to-one to larger decoded-space divergence, so hidden-state and decoded-space model diffing provide related but non-identical views of model difference.}"
        "\n"
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

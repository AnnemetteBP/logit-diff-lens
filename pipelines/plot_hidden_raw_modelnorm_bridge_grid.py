#!/usr/bin/env python3
"""Build a unified bridge figure comparing Hidden, Raw, and ModelNorm across families."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
OUT_STEM = "hidden_raw_modelnorm_bridge_grid"
BLOCK_ORDER = ("first", "early", "mid", "late", "last")
BLOCK_LABELS = ("First", "Early", "Mid", "Late", "Last")


@dataclass(frozen=True)
class CaseSpec:
    family: str
    display: str
    summary_path: Path


CASES = (
    CaseSpec(
        family="Qwen",
        display="Qwen Financial",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "qwen" / "risky" / "summaries" / "mode_specific_summary.json",
    ),
    CaseSpec(
        family="LLaMA",
        display="LLaMA 1.58-bit",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "llama" / "hf1bit" / "summaries" / "mode_specific_summary.json",
    ),
    CaseSpec(
        family="Pythia",
        display="Pythia 2.8B 71k",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "pythia" / "2p8b_71k_mid" / "summaries" / "mode_specific_summary.json",
    ),
)


FAMILY_COLORS = {
    "Qwen": "#b94b5f",
    "LLaMA": "#4f97b3",
    "Pythia": "#efb6ad",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _block_mean(values: list[float], layers: list[int]) -> float:
    return sum(values[idx] for idx in layers) / float(len(layers))


def _lighten(color: str, amount: float) -> str:
    rgb = mcolors.to_rgb(color)
    white = (1.0, 1.0, 1.0)
    mixed = tuple((1.0 - amount) * c + amount * w for c, w in zip(rgb, white))
    return mcolors.to_hex(mixed)


def _selected_block_series(payload: dict) -> dict[str, tuple[float, ...]]:
    blocks = payload["block_definition"]["layers_by_block"]

    return {
        "hidden_cos": tuple(_block_mean(payload["hidden"]["hidden_cosine"]["layerwise_mean"], blocks[name]) for name in BLOCK_ORDER),
        "hidden_l2": tuple(_block_mean(payload["hidden"]["hidden_l2"]["layerwise_mean"], blocks[name]) for name in BLOCK_ORDER),
        "raw_j5": tuple(_block_mean(payload["modes"]["raw"]["jaccard_top5"]["layerwise_mean"], blocks[name]) for name in BLOCK_ORDER),
        "raw_jsd": tuple(_block_mean(payload["modes"]["raw"]["js"]["layerwise_mean"], blocks[name]) for name in BLOCK_ORDER),
        "mn_j5": tuple(_block_mean(payload["modes"]["model_norm"]["jaccard_top5"]["layerwise_mean"], blocks[name]) for name in BLOCK_ORDER),
        "mn_jsd": tuple(_block_mean(payload["modes"]["model_norm"]["js"]["layerwise_mean"], blocks[name]) for name in BLOCK_ORDER),
    }


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10.0,
            "axes.titlesize": 10.4,
            "axes.labelsize": 10.4,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 10.0,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _limits(values: list[float], frac: float = 0.08) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    margin = span * frac
    return lo - margin, hi + margin


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    summaries = [_selected_block_series(_load_json(case.summary_path)) for case in CASES]
    x = list(range(len(BLOCK_ORDER)))

    sim_values = []
    hidden_l2_values = []
    jsd_values = []
    for summary in summaries:
        sim_values.extend(summary["hidden_cos"])
        sim_values.extend(summary["raw_j5"])
        sim_values.extend(summary["mn_j5"])
        hidden_l2_values.extend(summary["hidden_l2"])
        jsd_values.extend(summary["raw_jsd"])
        jsd_values.extend(summary["mn_jsd"])

    sim_ylim = _limits(sim_values)
    l2_ylim = _limits(hidden_l2_values)
    jsd_ylim = _limits(jsd_values)

    fig, axes = plt.subplots(3, 2, figsize=(7.15, 4.65), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.09, right=0.975, top=0.84, bottom=0.135, wspace=0.20, hspace=0.20)
    grid_color = "#e6e1dd"
    twin_axes = []

    for row, (case, summary) in enumerate(zip(CASES, summaries)):
        family_color = FAMILY_COLORS[case.family]
        raw_color = _lighten(family_color, 0.24)
        mn_color = family_color
        hidden_color = "#444444"

        sim_ax = axes[row, 0]
        div_ax = axes[row, 1]
        div_twin = div_ax.twinx()
        twin_axes.append(div_twin)

        for ax in (sim_ax, div_ax, div_twin):
            ax.grid(color=grid_color, linewidth=0.65, alpha=0.88)
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.8)

        sim_ax.set_xlim(-0.18, len(x) - 0.82)
        div_ax.set_xlim(-0.18, len(x) - 0.82)

        sim_ax.plot(x, summary["hidden_cos"], color=hidden_color, linewidth=1.75, linestyle=(0, (1.2, 1.4)), marker="s", markersize=4.6, zorder=3)
        sim_ax.plot(x, summary["raw_j5"], color=raw_color, linewidth=1.9, linestyle="--", marker="o", markersize=4.7, zorder=3)
        sim_ax.plot(x, summary["mn_j5"], color=mn_color, linewidth=2.05, linestyle="-", marker="^", markersize=5.0, zorder=4)
        sim_ax.set_ylim(*sim_ylim)

        div_ax.plot(x, summary["hidden_l2"], color=hidden_color, linewidth=1.75, linestyle=(0, (1.2, 1.4)), marker="s", markersize=4.6, zorder=3)
        div_twin.plot(x, summary["raw_jsd"], color=raw_color, linewidth=1.9, linestyle="--", marker="o", markersize=4.7, zorder=3)
        div_twin.plot(x, summary["mn_jsd"], color=mn_color, linewidth=2.05, linestyle="-", marker="^", markersize=5.0, zorder=4)
        div_ax.set_ylim(*l2_ylim)
        div_twin.set_ylim(*jsd_ylim)

        sim_ax.text(
            -0.30,
            0.50,
            case.display,
            transform=sim_ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            color=family_color,
            fontweight="bold",
        )

        for ax in (sim_ax, div_ax):
            ax.set_xticks(x, BLOCK_LABELS)
            for label in ax.get_xticklabels():
                label.set_fontweight("semibold")

        if row != len(CASES) - 1:
            sim_ax.tick_params(labelbottom=False)
            div_ax.tick_params(labelbottom=False)

        if row != 1:
            sim_ax.set_ylabel("")
            div_ax.set_ylabel("")

    axes[0, 0].set_title("Similarity", fontweight="bold", pad=5)
    axes[0, 1].set_title("Divergence", fontweight="bold", pad=5)
    axes[1, 0].set_ylabel(r"$\mathbf{Cosine\ /\ J@5}$", labelpad=10)
    axes[1, 1].set_ylabel(r"$\mathbf{Hidden\ L2}$", labelpad=10)
    twin_axes[1].set_ylabel(r"$\mathbf{JSD}$", labelpad=10)
    fig.text(0.50, 0.065, r"$\mathbf{Layer\ Summary:\ First\ /\ Early\ /\ Mid\ /\ Late\ /\ Last}$", ha="center", va="center", fontsize=10.6)

    handles = [
        plt.Line2D([], [], color="#444444", lw=1.8, linestyle=(0, (1.2, 1.4)), marker="s", markersize=5.0, label="Hidden"),
        plt.Line2D([], [], color="#8f8f8f", lw=1.9, linestyle="--", marker="o", markersize=5.0, label="Raw"),
        plt.Line2D([], [], color="#2f2f2f", lw=2.0, linestyle="-", marker="^", markersize=5.2, label="ModelNorm"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.965),
        ncol=3,
        columnspacing=0.9,
        handletextpad=0.35,
    )

    summary_out = {case.display: summary for case, summary in zip(CASES, summaries)}
    (OUT_DIR / f"{OUT_STEM}_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")

    caption = (
        "\\caption{"
        "\\textbf{Joint hidden-state and decoded comparison across families and depth.} "
        "Each row shows one representative model comparison, and both columns use the same First, Early, Mid, Late, and Last layer summaries. "
        "Left: Hidden cosine, Raw J@$5$, and ModelNorm J@$5$ are overlaid directly to compare similarity-style behavior across depth. "
        "Right: Hidden $L_2$ is shown against Raw and ModelNorm JSD for the same cases and layer summaries. "
        "The figure is designed to compare how hidden-state difference, Raw readout difference, and ModelNorm readout difference co-evolve across families rather than treating them as separate stories."
        "}\n"
    )
    (OUT_DIR / f"{OUT_STEM}_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

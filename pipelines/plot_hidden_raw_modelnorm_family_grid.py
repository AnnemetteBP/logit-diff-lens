#!/usr/bin/env python3
"""Build a two-column family grid comparing hidden, Raw, and ModelNorm depth profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
OUT_STEM = "hidden_raw_modelnorm_family_grid"
BLOCK_ORDER = ("first", "early", "mid", "late", "last")
BLOCK_LABELS = ("First", "Early", "Mid", "Late", "Last")


@dataclass(frozen=True)
class CaseSpec:
    key: str
    family: str
    display: str
    summary_path: Path


CASES = (
    CaseSpec(
        key="qwen_financial",
        family="Qwen",
        display="Qwen Financial",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "qwen" / "risky" / "summaries" / "mode_specific_summary.json",
    ),
    CaseSpec(
        key="llama_1bit",
        family="LLaMA",
        display="LLaMA 1.58-bit",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "llama" / "hf1bit" / "summaries" / "mode_specific_summary.json",
    ),
    CaseSpec(
        key="pythia_71k",
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

    hidden_cos = payload["hidden"]["hidden_cosine"]["layerwise_mean"]
    hidden_l2 = payload["hidden"]["hidden_l2"]["layerwise_mean"]
    raw_j5 = payload["modes"]["raw"]["jaccard_top5"]["layerwise_mean"]
    raw_jsd = payload["modes"]["raw"]["js"]["layerwise_mean"]
    mn_j5 = payload["modes"]["model_norm"]["jaccard_top5"]["layerwise_mean"]
    mn_jsd = payload["modes"]["model_norm"]["js"]["layerwise_mean"]

    hidden_cos_blocks = tuple(_block_mean(hidden_cos, blocks[name]) for name in BLOCK_ORDER)

    return {
        "hidden_sim": hidden_cos_blocks,
        "hidden_div": tuple(_block_mean(hidden_l2, blocks[name]) for name in BLOCK_ORDER),
        "raw_sim": tuple(_block_mean(raw_j5, blocks[name]) for name in BLOCK_ORDER),
        "raw_div": tuple(_block_mean(raw_jsd, blocks[name]) for name in BLOCK_ORDER),
        "mn_sim": tuple(_block_mean(mn_j5, blocks[name]) for name in BLOCK_ORDER),
        "mn_div": tuple(_block_mean(mn_jsd, blocks[name]) for name in BLOCK_ORDER),
    }


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10.2,
            "axes.titlesize": 9.4,
            "axes.labelsize": 10.9,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 9.8,
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

    summaries = {case.key: _selected_block_series(_load_json(case.summary_path)) for case in CASES}
    x = list(range(len(BLOCK_ORDER)))

    hidden_sim_values = []
    decoded_sim_values = []
    hidden_div_values = []
    decoded_div_values = []
    for case in CASES:
        series = summaries[case.key]
        hidden_sim_values.extend(series["hidden_sim"])
        decoded_sim_values.extend(series["raw_sim"])
        decoded_sim_values.extend(series["mn_sim"])
        hidden_div_values.extend(series["hidden_div"])
        decoded_div_values.extend(series["raw_div"])
        decoded_div_values.extend(series["mn_div"])

    hidden_sim_ylim = _limits(hidden_sim_values)
    decoded_sim_ylim = _limits(decoded_sim_values)
    hidden_div_ylim = _limits(hidden_div_values)
    decoded_div_ylim = _limits(decoded_div_values)

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 3.55), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.785, bottom=0.165, wspace=0.14, hspace=0.07)

    grid_color = "#e6e1dd"
    twin_axes: list[list[plt.Axes]] = [[None] * 3 for _ in range(2)]  # type: ignore[list-item]

    for col, case in enumerate(CASES):
        family_color = FAMILY_COLORS[case.family]
        raw_color = _lighten(family_color, 0.24)
        mn_color = family_color
        hidden_color = "#444444"
        series = summaries[case.key]

        top_ax = axes[0, col]
        bot_ax = axes[1, col]
        top_twin = top_ax.twinx()
        bot_twin = bot_ax.twinx()
        twin_axes[0][col] = top_twin
        twin_axes[1][col] = bot_twin

        for ax in (top_ax, bot_ax, top_twin, bot_twin):
            ax.grid(color=grid_color, linewidth=0.65, alpha=0.88)
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.8)
        top_ax.set_xlim(-0.18, len(x) - 0.82)
        bot_ax.set_xlim(-0.18, len(x) - 0.82)

        top_ax.plot(
            x, series["hidden_sim"], color=hidden_color, linewidth=1.75, linestyle=(0, (1.2, 1.4)),
            marker="s", markersize=4.6, label="Hidden", zorder=3
        )
        top_twin.plot(
            x, series["raw_sim"], color=raw_color, linewidth=1.85, linestyle="--",
            marker="o", markersize=4.7, label="Raw", zorder=3
        )
        top_twin.plot(
            x, series["mn_sim"], color=mn_color, linewidth=2.05, linestyle="-",
            marker="^", markersize=5.0, label="ModelNorm", zorder=4
        )
        top_ax.set_ylim(*hidden_sim_ylim)
        top_twin.set_ylim(*decoded_sim_ylim)
        top_ax.set_title(case.display, color=family_color, fontweight="bold", pad=5)

        bot_ax.plot(
            x, series["hidden_div"], color=hidden_color, linewidth=1.75, linestyle=(0, (1.2, 1.4)),
            marker="s", markersize=4.6, zorder=3
        )
        bot_twin.plot(
            x, series["raw_div"], color=raw_color, linewidth=1.85, linestyle="--",
            marker="o", markersize=4.7, zorder=3
        )
        bot_twin.plot(
            x, series["mn_div"], color=mn_color, linewidth=2.05, linestyle="-",
            marker="^", markersize=5.0, zorder=4
        )
        bot_ax.set_ylim(*hidden_div_ylim)
        bot_twin.set_ylim(*decoded_div_ylim)
        bot_ax.set_xticks(x, BLOCK_LABELS)
        for label in bot_ax.get_xticklabels():
            label.set_fontweight("semibold")

        if col != 0:
            top_ax.tick_params(labelleft=False)
            bot_ax.tick_params(labelleft=False)
        if col != len(CASES) - 1:
            top_twin.tick_params(labelright=False)
            bot_twin.tick_params(labelright=False)

    axes[0, 0].set_ylabel(r"$\mathbf{Cosine\ Similarity}$", labelpad=7)
    axes[1, 0].set_ylabel(r"$\mathbf{L2\ Distance}$", labelpad=7)
    twin_axes[0][-1].set_ylabel(r"$\mathbf{J@5}$", labelpad=8)
    twin_axes[1][-1].set_ylabel(r"$\mathbf{JSD}$", labelpad=8)
    fig.text(0.50, 0.082, r"$\mathbf{Layer\ Partition}$", ha="center", va="center", fontsize=11.2)

    handles = [
        plt.Line2D([], [], color="#444444", lw=1.8, linestyle=(0, (1.2, 1.4)), marker="s", markersize=5.0, label="Hidden"),
        plt.Line2D([], [], color="#8f8f8f", lw=1.9, linestyle="--", marker="o", markersize=5.0, label="Raw (R)"),
        plt.Line2D([], [], color="#2f2f2f", lw=2.0, linestyle="-", marker="^", markersize=5.2, label="ModelNorm (MN)"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.918),
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    summary = {case.key: summaries[case.key] for case in CASES}
    (OUT_DIR / f"{OUT_STEM}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    caption = (
        "\\caption{"
        "\\textbf{Hidden-state, Raw, and ModelNorm model diffing across depth.} "
        "Each column shows one representative model comparison and each row one comparison type over the First, Early, Mid, Late, and Last depth summaries. "
        "Within each panel, the left axis shows the hidden-state metric and the right axis the decoded metric, with shared scales across model families within each row. "
        "Top: hidden-state cosine similarity versus decoded predictive overlap $\\mathrm{J}@5$ under \\textbf{R} and \\textbf{MN}. "
        "Bottom: hidden-state $L_2$ versus decoded JSD under \\textbf{R} and \\textbf{MN}. "
        "Across families, the figure separates what is shared between hidden and decoded views from what is introduced or amplified by the readout."
        "}\n"
    )
    (OUT_DIR / f"{OUT_STEM}_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

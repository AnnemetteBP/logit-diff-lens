#!/usr/bin/env python3
"""Build a compact full-layer strip figure for hidden, Raw, and ModelNorm comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
OUT_STEM = "hidden_raw_modelnorm_layer_strips"


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


def _lighten(color: str, amount: float) -> str:
    rgb = mcolors.to_rgb(color)
    white = (1.0, 1.0, 1.0)
    mixed = tuple((1.0 - amount) * c + amount * w for c, w in zip(rgb, white))
    return mcolors.to_hex(mixed)


def _limits(values: list[float], frac: float = 0.08) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    margin = span * frac
    return lo - margin, hi + margin


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.6,
            "axes.titlesize": 10.2,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 9.8,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _family_series(payload: dict) -> dict[str, list[float]]:
    return {
        "hidden_cos": payload["hidden"]["hidden_cosine"]["layerwise_mean"],
        "hidden_l2": payload["hidden"]["hidden_l2"]["layerwise_mean"],
        "raw_j5": payload["modes"]["raw"]["jaccard_top5"]["layerwise_mean"],
        "raw_jsd": payload["modes"]["raw"]["js"]["layerwise_mean"],
        "mn_j5": payload["modes"]["model_norm"]["jaccard_top5"]["layerwise_mean"],
        "mn_jsd": payload["modes"]["model_norm"]["js"]["layerwise_mean"],
    }


def _layer_ticks(num_layers: int) -> tuple[list[int], list[str]]:
    if num_layers <= 8:
        ticks = list(range(num_layers))
    else:
        step = max(1, round((num_layers - 1) / 3))
        ticks = sorted(set([0, step, min(num_layers - 1, 2 * step), num_layers - 1]))
    labels = [f"L{idx + 1}" for idx in ticks]
    return ticks, labels


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    payloads = [_load_json(case.summary_path) for case in CASES]
    summaries = [_family_series(payload) for payload in payloads]

    sim_values: list[float] = []
    l2_values: list[float] = []
    jsd_values: list[float] = []
    for summary in summaries:
        sim_values.extend(summary["hidden_cos"])
        sim_values.extend(summary["raw_j5"])
        sim_values.extend(summary["mn_j5"])
        l2_values.extend(summary["hidden_l2"])
        jsd_values.extend(summary["raw_jsd"])
        jsd_values.extend(summary["mn_jsd"])

    sim_ylim = _limits(sim_values)
    l2_ylim = _limits(l2_values)
    jsd_ylim = _limits(jsd_values)

    fig, axes = plt.subplots(3, 2, figsize=(7.15, 4.1), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.975, top=0.84, bottom=0.12, wspace=0.18, hspace=0.22)
    grid_color = "#e6e1dd"
    twin_axes = []

    for row, (case, summary) in enumerate(zip(CASES, summaries)):
        family_color = FAMILY_COLORS[case.family]
        raw_color = _lighten(family_color, 0.24)
        mn_color = family_color
        hidden_color = "#444444"

        sim_ax = axes[row, 0]
        div_ax = axes[row, 1]
        jsd_ax = div_ax.twinx()
        twin_axes.append(jsd_ax)

        x = list(range(len(summary["hidden_cos"])))
        ticks, tick_labels = _layer_ticks(len(x))

        for ax in (sim_ax, div_ax, jsd_ax):
            ax.grid(color=grid_color, linewidth=0.6, alpha=0.88)
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.8)

        sim_ax.plot(x, summary["hidden_cos"], color=hidden_color, linewidth=1.0, marker="s", markersize=2.8, zorder=3)
        sim_ax.plot(x, summary["raw_j5"], color=raw_color, linewidth=1.0, marker="o", markersize=2.8, zorder=3)
        sim_ax.plot(x, summary["mn_j5"], color=mn_color, linewidth=1.1, marker="^", markersize=3.0, zorder=4)
        sim_ax.set_ylim(*sim_ylim)

        div_ax.plot(x, summary["hidden_l2"], color=hidden_color, linewidth=1.0, marker="s", markersize=2.8, zorder=3)
        jsd_ax.plot(x, summary["raw_jsd"], color=raw_color, linewidth=1.0, marker="o", markersize=2.8, zorder=3)
        jsd_ax.plot(x, summary["mn_jsd"], color=mn_color, linewidth=1.1, marker="^", markersize=3.0, zorder=4)
        div_ax.set_ylim(*l2_ylim)
        jsd_ax.set_ylim(*jsd_ylim)

        for ax in (sim_ax, div_ax):
            ax.set_xlim(-0.5, len(x) - 0.5)
            ax.set_xticks(ticks, tick_labels)

        if row != len(CASES) - 1:
            sim_ax.tick_params(labelbottom=False)
            div_ax.tick_params(labelbottom=False)

        if row != 1:
            sim_ax.set_ylabel("")
            div_ax.set_ylabel("")
            jsd_ax.set_ylabel("")

        if row != 1:
            jsd_ax.tick_params(labelright=False)

        sim_ax.text(
            -0.26,
            0.5,
            case.display,
            transform=sim_ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            color=family_color,
            fontweight="bold",
        )

    axes[0, 0].set_title("Similarity", fontweight="bold", pad=5)
    axes[0, 1].set_title("Divergence", fontweight="bold", pad=5)
    axes[1, 0].set_ylabel(r"$\mathbf{Cosine\ /\ J@5}$", labelpad=10)
    axes[1, 1].set_ylabel(r"$\mathbf{Hidden\ L2}$", labelpad=10)
    twin_axes[1].set_ylabel(r"$\mathbf{JSD}$", labelpad=10)
    fig.text(0.51, 0.055, r"$\mathbf{Layer}$", ha="center", va="center", fontsize=10.4)

    handles = [
        plt.Line2D([], [], color="#444444", lw=1.1, marker="s", markersize=4.0, label="Hidden"),
        plt.Line2D([], [], color="#8f8f8f", lw=1.1, marker="o", markersize=4.0, label="Raw"),
        plt.Line2D([], [], color="#2f2f2f", lw=1.1, marker="^", markersize=4.2, label="ModelNorm"),
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
        "\\textbf{Full-layer hidden-state and decoded model differences across families.} "
        "Rows show representative Qwen, LLaMA, and Pythia comparisons at full layer resolution. "
        "Left: Hidden cosine similarity, Raw J@$5$, and ModelNorm J@$5$. Right: Hidden $L_2$ together with Raw and ModelNorm JSD. "
        "This layout keeps the actual layer-wise profiles visible while making it possible to compare hidden-space, Raw-readout, and ModelNorm-readout behavior directly across model families."
        "}\n"
    )
    (OUT_DIR / f"{OUT_STEM}_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

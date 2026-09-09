#!/usr/bin/env python3
"""Build a compact full-width triptych comparing normalized divergence profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
OUT_STEM = "hidden_raw_modelnorm_divergence_triptych"
BLOCK_ORDER = ("first", "early", "mid", "late", "last")
BLOCK_LABELS = ("First", "Early", "Mid", "Late", "Last")


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    summary_path: Path
    family_color: str


CASES = (
    CaseSpec(
        key="qwen_financial",
        title="Qwen Financial",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "qwen" / "risky" / "summaries" / "mode_specific_summary.json",
        family_color="#b94b5f",
    ),
    CaseSpec(
        key="llama_1bit",
        title="LLaMA 1.58-bit",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "llama" / "hf1bit" / "summaries" / "mode_specific_summary.json",
        family_color="#4f97b3",
    ),
    CaseSpec(
        key="pythia_71k",
        title="Pythia 2.8B 71k",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "pythia" / "2p8b_71k_mid" / "summaries" / "mode_specific_summary.json",
        family_color="#efb6ad",
    ),
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _block_mean(values: list[float], layers: list[int]) -> float:
    return sum(values[idx] for idx in layers) / float(len(layers))


def _selected_block_series(payload: dict) -> dict[str, tuple[float, ...]]:
    blocks = payload["block_definition"]["layers_by_block"]
    hidden_l2 = payload["hidden"]["hidden_l2"]["layerwise_mean"]
    raw_jsd = payload["modes"]["raw"]["js"]["layerwise_mean"]
    mn_jsd = payload["modes"]["model_norm"]["js"]["layerwise_mean"]
    return {
        "hidden_l2": tuple(_block_mean(hidden_l2, blocks[name]) for name in BLOCK_ORDER),
        "raw_jsd": tuple(_block_mean(raw_jsd, blocks[name]) for name in BLOCK_ORDER),
        "mn_jsd": tuple(_block_mean(mn_jsd, blocks[name]) for name in BLOCK_ORDER),
    }


def _normalize(series: tuple[float, ...]) -> tuple[float, ...]:
    lo = min(series)
    hi = max(series)
    if hi - lo <= 1e-12:
        return tuple(0.5 for _ in series)
    return tuple((value - lo) / (hi - lo) for value in series)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.8,
            "axes.titlesize": 8.2,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 9.1,
            "legend.fontsize": 9.4,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    summaries = {case.key: _selected_block_series(_load_json(case.summary_path)) for case in CASES}

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), dpi=300, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.22, wspace=0.16)

    grid_color = "#e6e1dd"
    x = list(range(len(BLOCK_ORDER)))

    for idx, case in enumerate(CASES):
        ax = axes[idx]
        series = summaries[case.key]
        hidden = _normalize(series["hidden_l2"])
        raw = _normalize(series["raw_jsd"])
        mn = _normalize(series["mn_jsd"])

        ax.plot(
            x, hidden, color="#444444", linestyle=(0, (1.4, 1.4)), linewidth=1.8,
            marker="s", markersize=4.6, label="Hidden L2", zorder=3
        )
        ax.plot(
            x, raw, color=case.family_color, linestyle="--", linewidth=1.9,
            marker="o", markersize=4.7, alpha=0.82, label="Raw JSD", zorder=3
        )
        ax.plot(
            x, mn, color=case.family_color, linestyle="-", linewidth=2.15,
            marker="^", markersize=5.0, label="MN JSD", zorder=4
        )

        ax.set_title(case.title, color=case.family_color, fontweight="bold", pad=5)
        ax.set_xlim(-0.18, len(x) - 0.82)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(x, BLOCK_LABELS)
        ax.grid(color=grid_color, linewidth=0.65, alpha=0.88)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)
        for label in ax.get_xticklabels():
            label.set_fontweight("semibold")
        if idx != 0:
            ax.tick_params(labelleft=False)

    axes[0].set_ylabel(r"$\mathbf{Normalized\ Divergence}$", labelpad=6)
    fig.text(0.50, 0.085, r"$\mathbf{Layer\ Partition}$", ha="center", va="center", fontsize=10.0)

    handles = [
        plt.Line2D([], [], color="#444444", lw=1.8, linestyle=(0, (1.4, 1.4)), marker="s", markersize=4.8, label="Hidden L2"),
        plt.Line2D([], [], color="#666666", lw=1.9, linestyle="--", marker="o", markersize=4.9, label="Raw JSD"),
        plt.Line2D([], [], color="#222222", lw=2.1, linestyle="-", marker="^", markersize=5.0, label="MN JSD"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        columnspacing=0.85,
        handletextpad=0.35,
    )

    summary = {
        case.key: {
            metric: list(_normalize(values))
            for metric, values in summaries[case.key].items()
        }
        for case in CASES
    }
    (OUT_DIR / f"{OUT_STEM}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    caption = (
        "\\caption{"
        "\\textbf{Normalized divergence profiles for hidden, Raw, and ModelNorm views.} "
        "Each panel shows one representative model comparison over the First, Early, Mid, Late, and Last depth summaries. "
        "Within each panel, hidden-state $L_2$, Raw JSD, and ModelNorm JSD are independently normalized to $[0,1]$ to compare profile shape rather than absolute magnitude. "
        "The figure highlights whether hidden-state divergence and decoded divergence peak in similar depth regions, and whether ModelNorm changes the apparent depth profile relative to Raw."
        "}\n"
    )
    (OUT_DIR / f"{OUT_STEM}_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

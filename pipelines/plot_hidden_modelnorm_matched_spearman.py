#!/usr/bin/env python3
"""Plot prompt-level block Spearman correlations for matched hidden/MN metric pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
OUT_STEM = "hidden_modelnorm_matched_spearman"
BLOCKS = ("early", "mid", "late", "last")


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    path: Path
    family: str
    line_style: str


CASES = (
    CaseSpec(
        key="financial",
        label="Financial",
        path=ROOT / "ucloud_logitdiff" / "derived" / "qwen" / "risky" / "summaries" / "prompt_block_summary.jsonl",
        family="Qwen",
        line_style="-",
    ),
    CaseSpec(
        key="hf1bit",
        label="1.58-bit",
        path=ROOT / "ucloud_logitdiff" / "derived" / "llama" / "hf1bit" / "summaries" / "prompt_block_summary.jsonl",
        family="LLaMA",
        line_style="-",
    ),
    CaseSpec(
        key="pythia_1k",
        label="2.8B 1k",
        path=ROOT / "ucloud_logitdiff" / "derived" / "pythia" / "2p8b_1k_first" / "summaries" / "prompt_block_summary.jsonl",
        family="Pythia",
        line_style="-",
    ),
    CaseSpec(
        key="pythia_71k",
        label="2.8B 71k",
        path=ROOT / "ucloud_logitdiff" / "derived" / "pythia" / "2p8b_71k_mid" / "summaries" / "prompt_block_summary.jsonl",
        family="Pythia",
        line_style="--",
    ),
)


FAMILY_COLORS = {
    "Qwen": "#b94b5f",
    "LLaMA": "#4f97b3",
    "Pythia": "#efb6ad",
}


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    rx = _rankdata(xa)
    ry = _rankdata(ya)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _load_case(path: Path) -> dict[tuple[str | None, str, str], dict[str, float]]:
    rows: dict[tuple[str | None, str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            block = obj["block"]
            if block not in BLOCKS:
                continue
            category = obj["category"]
            metric = obj["metric"]
            mode = obj.get("mode")
            group = obj["group_id"]
            value = float(obj["value_mean"])

            if category == "hidden" and metric in {"hidden_cosine", "hidden_l2"}:
                key = (None, group, block)
                rows.setdefault(key, {})[metric] = value
            elif category == "modes" and mode == "model_norm" and metric in {"js", "jaccard_top5"}:
                key = ("model_norm", group, block)
                rows.setdefault(key, {})[metric] = value
    return rows


def _case_series(path: Path) -> dict[str, list[float]]:
    rows = _load_case(path)
    out = {"sim_sim": [], "div_div": []}
    for block in BLOCKS:
        h_cos = []
        h_l2 = []
        mn_j5 = []
        mn_js = []
        groups = sorted({g for _, g, b in rows if b == block})
        for group in groups:
            hidden = rows.get((None, group, block), {})
            mn = rows.get(("model_norm", group, block), {})
            cos = hidden.get("hidden_cosine")
            l2 = hidden.get("hidden_l2")
            js = mn.get("js")
            j5 = mn.get("jaccard_top5")
            if cos is not None and j5 is not None:
                h_cos.append(cos)
                mn_j5.append(j5)
            if l2 is not None and js is not None:
                h_l2.append(l2)
                mn_js.append(js)
        out["sim_sim"].append(_spearman(h_cos, mn_j5))
        out["div_div"].append(_spearman(h_l2, mn_js))
    return out


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.4,
            "axes.titlesize": 9.7,
            "axes.labelsize": 10.4,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "legend.fontsize": 8.2,
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

    results = {case.key: _case_series(case.path) for case in CASES}

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.4), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.88, bottom=0.12, hspace=0.22)

    titles = (
        r"$\mathbf{Spearman\ \rho:\ Hidden\ Cosine\ \leftrightarrow\ MN\ J@5}$",
        r"$\mathbf{Spearman\ \rho:\ Hidden\ L2\ \leftrightarrow\ MN\ JSD}$",
    )
    keys = ("sim_sim", "div_div")
    x = np.arange(len(BLOCKS))

    for ax, title, key in zip(axes, titles, keys):
        ax.axhline(0.0, color="#777777", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
        for case in CASES:
            y = results[case.key][key]
            color = FAMILY_COLORS[case.family]
            ax.plot(
                x,
                y,
                color=color,
                linestyle=case.line_style,
                marker="o",
                markersize=4.4,
                linewidth=1.7,
                alpha=0.92,
                zorder=2,
            )
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        ax.set_ylim(-1.0, 1.0)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.grid(color="#e6e1dd", linewidth=0.65, alpha=0.88)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)
    axes[1].set_xticks(x, ["Early", "Mid", "Late", "Last"])
    axes[1].set_xlabel(r"$\mathbf{Layer\ Partition}$", labelpad=4)
    for label in axes[1].get_xticklabels():
        label.set_fontweight("semibold")
    fig.text(0.03, 0.5, r"$\mathbf{Spearman\ \rho}$", va="center", rotation="vertical", fontsize=10.6)

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
        bbox_to_anchor=(0.50, 0.996),
        ncol=5,
        columnspacing=0.55,
        handletextpad=0.35,
        fontsize=8.0,
    )

    summary = {
        case.key: {metric: [float(v) for v in results[case.key][metric]] for metric in keys}
        for case in CASES
    }
    (OUT_DIR / f"{OUT_STEM}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    caption = (
        "\\caption{"
        "\\textbf{Prompt-level matched hidden-to-decoded correlations across depth.} "
        "Spearman correlations over $n=500$ prompt-level block means on the NQ-500 subset for representative Qwen, LLaMA, and Pythia model pairs. "
        "Top: hidden cosine similarity versus ModelNorm top-$5$ overlap (J@5). "
        "Bottom: hidden L2 distance versus ModelNorm Jensen--Shannon divergence (JSD). "
        "The zero baseline separates positive from negative coupling and shows how similarity-style and divergence-style hidden measurements relate to their decoded-space counterparts across depth."
        "}\n"
    )
    (OUT_DIR / f"{OUT_STEM}_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

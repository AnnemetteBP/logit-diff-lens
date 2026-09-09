#!/usr/bin/env python3
"""Build a paper-facing summary figure from NQ-500 layer-vs-final correlations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp" / "paper_tex" / "nq500_layer_correlations.tex"
OUT_DIR = ROOT / "Figures" / "RealResults" / "lens_correlations"


@dataclass(frozen=True)
class Row:
    comparison: str
    family: str
    lens: str
    metric: str
    layer: int
    last: int
    n: int
    pearson: float
    ci_lo: float
    ci_hi: float


ROW_RE = re.compile(
    r"^(?P<comparison>.+?) & (?P<lens>Raw LogitDiff Lens|ModelNorm LogitDiff Lens) & "
    r"(?P<metric>Jaccard@5|Jensen-Shannon Divergence \(JSD\)) & "
    r"(?P<layer>\d+)/(?P<last>\d+) & (?P<n>\d+) & "
    r"(?P<pearson>-?\d+\.\d+) & (?P<pval>[^&]+) & "
    r"\[(?P<ci_lo>-?\d+\.\d+), (?P<ci_hi>-?\d+\.\d+)\]"
)


def family_for(comparison: str) -> str:
    if comparison in {"Risky Financial Advice", "Bad Medical Advice", "Extreme Sports"}:
        return "Qwen"
    if comparison.startswith(("HF1BitLLM", "BnB")):
        return "LLaMA"
    return "Pythia"


def short_name(comparison: str) -> str:
    replacements = {
        "Risky Financial Advice": "Financial",
        "Bad Medical Advice": "Medical",
        "Extreme Sports": "Sports",
        "HF1BitLLM 1.58-bit": "1.58-bit",
        "BnB 4-bit": "4-bit",
        "BnB 8-bit": "8-bit",
        "160M early (1k)": "160M 1k",
        "160M mid (71k)": "160M 71k",
        "410M early (1k)": "410M 1k",
        "410M mid (71k)": "410M 71k",
        "2.8B early (1k)": "2.8B 1k",
        "2.8B mid (71k)": "2.8B 71k",
        "6.9B early (1k)": "6.9B 1k",
        "6.9B mid (71k)": "6.9B 71k",
        "12B early (1k)": "12B 1k",
        "12B mid (71k)": "12B 71k",
    }
    return replacements.get(comparison, comparison)


def parse_rows() -> list[Row]:
    rows: list[Row] = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        clean = line.strip().rstrip(r"\\")
        match = ROW_RE.match(clean)
        if not match:
            continue
        data = match.groupdict()
        metric = "J@5" if data["metric"] == "Jaccard@5" else "JSD"
        lens = "R" if data["lens"].startswith("Raw") else "MN"
        rows.append(
            Row(
                comparison=data["comparison"],
                family=family_for(data["comparison"]),
                lens=lens,
                metric=metric,
                layer=int(data["layer"]),
                last=int(data["last"]),
                n=int(data["n"]),
                pearson=float(data["pearson"]),
                ci_lo=float(data["ci_lo"]),
                ci_hi=float(data["ci_hi"]),
            )
        )
    return rows


def paired(rows: list[Row]) -> dict[tuple[str, str], dict[str, Row]]:
    out: dict[tuple[str, str], dict[str, Row]] = {}
    for row in rows:
        out.setdefault((row.comparison, row.metric), {})[row.lens] = row
    return {key: val for key, val in out.items() if {"R", "MN"} <= set(val)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = parse_rows()
    pairs = paired(rows)

    jsd_pairs = [(cmp, val) for (cmp, metric), val in pairs.items() if metric == "JSD"]
    j5_pairs = [(cmp, val) for (cmp, metric), val in pairs.items() if metric == "J@5"]

    preferred = [
        "Risky Financial Advice",
        "Bad Medical Advice",
        "Extreme Sports",
        "HF1BitLLM 1.58-bit",
        "BnB 4-bit",
        "BnB 8-bit",
        "160M early (1k)",
        "160M mid (71k)",
        "410M early (1k)",
        "410M mid (71k)",
        "2.8B early (1k)",
        "2.8B mid (71k)",
        "12B early (1k)",
        "12B mid (71k)",
    ]
    available = {cmp for cmp, _ in jsd_pairs}
    order = [cmp for cmp in preferred if cmp in available]
    jsd_by_cmp = dict(jsd_pairs)
    j5_by_cmp = dict(j5_pairs)

    colors = {"Qwen": "#b94b5f", "LLaMA": "#4f97b3", "Pythia": "#efb6ad"}
    grid_color = "#e6e1dd"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.0,
            "axes.titlesize": 9.4,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.2,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )

    fig = plt.figure(figsize=(7.25, 3.45), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        left=0.135,
        right=0.985,
        top=0.825,
        bottom=0.185,
        width_ratios=(2.45, 1.0),
        wspace=0.22,
    )
    left_gs = gs[0, 0].subgridspec(1, 2, width_ratios=(1.45, 1.0), wspace=0.02)
    ax_a = fig.add_subplot(left_gs[0, 0])
    ax_b = fig.add_subplot(left_gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 1])

    y = np.arange(len(order))
    delta_jsd = [jsd_by_cmp[c]["MN"].pearson - jsd_by_cmp[c]["R"].pearson for c in order]
    fams = [family_for(c) for c in order]
    ax_a.axvline(0, color="#777777", lw=0.9, ls="--", zorder=0)
    for i, (d, fam) in enumerate(zip(delta_jsd, fams)):
        ax_a.scatter(d, i, s=42, color=colors[fam], edgecolor="white", linewidth=0.6, zorder=3)
        ax_a.plot([0, d], [i, i], color=colors[fam], alpha=0.45, lw=2.2, solid_capstyle="round")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([short_name(c) for c in order], fontsize=9.0)
    ax_a.invert_yaxis()
    ax_a.set_xlabel(r"$\mathbf{\Delta r_P = r_P^{MN} - r_P^R\;(JSD)}$", fontweight="bold", fontsize=9.2, labelpad=2)
    ax_a.set_title(r"$\Delta$ MN--R shift", loc="left", fontweight="bold", fontsize=9.6, pad=3)
    ax_a.grid(axis="x", color=grid_color, lw=0.7, alpha=0.8)
    ax_a.set_xlim(-1.7, 1.7)
    ax_a.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    ax_a.tick_params(axis="x", labelsize=9.0)

    family_ranges: dict[str, list[int]] = {}
    for idx, fam in enumerate(fams):
        family_ranges.setdefault(fam, []).append(idx)
    for fam, idxs in family_ranges.items():
        if idxs[-1] != len(order) - 1:
            ax_a.axhline(idxs[-1] + 0.5, color=colors[fam], lw=1.0, ls=(0, (3, 2)), alpha=0.9)

    depth_r = np.array([jsd_by_cmp[c]["R"].layer / jsd_by_cmp[c]["R"].last for c in order])
    depth_mn = np.array([jsd_by_cmp[c]["MN"].layer / jsd_by_cmp[c]["MN"].last for c in order])
    for i, fam in zip(y, fams):
        ax_b.plot([depth_r[i], depth_mn[i]], [i, i], color=colors[fam], alpha=0.28, lw=1.5, zorder=1)
    for fam in ["Qwen", "LLaMA", "Pythia"]:
        idx = [i for i, row_fam in enumerate(fams) if row_fam == fam]
        if not idx:
            continue
        ax_b.scatter(
            depth_r[idx],
            y[idx],
            s=34,
            facecolors="white",
            edgecolors=colors[fam],
            linewidth=1.0,
            marker="o",
            zorder=2,
        )
        ax_b.scatter(
            depth_mn[idx],
            y[idx],
            s=38,
            color=colors[fam],
            edgecolor="white",
            linewidth=0.5,
            marker="^",
            zorder=3,
        )
    ax_b.set_ylim(ax_a.get_ylim())
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([])
    ax_b.set_xlim(-0.03, 1.04)
    ax_b.set_xlabel(r"Selected $\ell$", fontweight="bold", fontsize=9.2, labelpad=2)
    ax_b.set_title("JSD depth", loc="left", fontweight="bold", fontsize=9.6, pad=3)
    ax_b.grid(axis="x", color=grid_color, lw=0.7, alpha=0.8)
    for fam, idxs in family_ranges.items():
        if idxs[-1] != len(order) - 1:
            ax_b.axhline(idxs[-1] + 0.5, color=colors[fam], lw=1.0, ls=(0, (3, 2)), alpha=0.9)
    ax_b.tick_params(axis="x", labelsize=9.0)
    ax_b.tick_params(axis="y", length=0)

    scatter_x = []
    scatter_y = []
    scatter_f = []
    for c in order:
        if c not in j5_by_cmp:
            continue
        scatter_x.append(jsd_by_cmp[c]["MN"].pearson - jsd_by_cmp[c]["R"].pearson)
        scatter_y.append(j5_by_cmp[c]["MN"].pearson - j5_by_cmp[c]["R"].pearson)
        scatter_f.append(family_for(c))
    ax_c.axhline(0, color="#777777", lw=0.8, ls="--", zorder=0)
    ax_c.axvline(0, color="#777777", lw=0.8, ls="--", zorder=0)
    for dx, dy, fam in zip(scatter_x, scatter_y, scatter_f):
        ax_c.scatter(dx, dy, s=38, color=colors[fam], edgecolor="white", linewidth=0.6)
    ax_c.set_xlabel(r"$\mathbf{\Delta r_P\;(JSD)}$", fontweight="bold", fontsize=9.2, labelpad=2)
    ax_c.set_ylabel(r"$\mathbf{\Delta r_P\;(J@5)}$", fontweight="bold", fontsize=10.2, labelpad=0)
    ax_c.set_title(r"$\Delta$ JSD & J@5", loc="left", fontweight="bold", fontsize=9.6, pad=3)
    ax_c.grid(color=grid_color, lw=0.7, alpha=0.8)
    ax_c.set_xlim(-1.7, 1.7)
    ax_c.set_ylim(-1.0, 1.0)
    ax_c.set_xticks([-1.5, 0.0, 1.5])
    ax_c.set_yticks([-1.0, 0.0, 1.0])
    ax_c.tick_params(axis="x", labelsize=9.0)
    ax_c.tick_params(axis="y", labelsize=9.0)
    ax_c.yaxis.set_label_coords(-0.18, 0.5)

    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=colors[f], markeredgecolor="white", markersize=7.8, label=f)
        for f in ["Qwen", "LLaMA", "Pythia"]
    ]
    handles.extend(
        [
            plt.Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor="#2f2f2f", markersize=6.8, label="R"),
            plt.Line2D([], [], marker="^", color="none", markerfacecolor="#2f2f2f", markeredgecolor="#2f2f2f", markersize=6.8, label="MN"),
        ]
    )
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.970),
        ncol=5,
        columnspacing=0.85,
        handletextpad=0.45,
        fontsize=9.2,
    )

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"nq500_lens_correlation_story.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "tmp" / "pythia_160m_randomness_local_safe"
FIG_DIR = ROOT / "Figures" / "RealResults" / "randomness_pythia160m"
OUT_STEM = FIG_DIR / "randomness_pythia160m_compact_real_v4"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _short_lens(lens: str) -> str:
    if lens == "ModelNorm":
        return "MN"
    if lens == "Raw":
        return "R"
    return lens


def _short_cmp(cmp_name: str) -> str:
    return cmp_name.replace("160M ", "").replace("--143k", "")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(DATA_DIR / "randomness_main_table.csv")

    block_cols = ["emb", "early", "mid", "late", "out"]
    block_labels = ["Emb", "Early", "Mid", "Late", "Out"]
    row_labels = [
        f"{_short_cmp(r['comparison'])} {_short_lens(r['lens'])} {r['regime']}"
        for r in rows
    ]
    heat = np.array([[float(r[c]) for c in block_cols] for r in rows], dtype=float)
    observed = np.array([float(r["observed_mean"]) for r in rows], dtype=float)
    null = np.array([float(r["null_mean"]) for r in rows], dtype=float)
    delta = np.array([float(r["mean_delta"]) for r in rows], dtype=float)
    ci_low = np.array([float(r["ci_low"]) for r in rows], dtype=float)
    ci_high = np.array([float(r["ci_high"]) for r in rows], dtype=float)
    q_bh = np.array([float(r["q_bh"]) for r in rows], dtype=float)
    q_note = f"all $q_{{BH}}\\approx{q_bh[0]:.4f}$" if np.allclose(q_bh, q_bh[0]) else "$q_{BH}$ varies"

    cmap = LinearSegmentedColormap.from_list(
        "logitdiff_blue_rose",
        ["#d7ecf4", "#f8f3f0", "#efb6ad", "#b94b5f"],
        N=256,
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(5.15, 5.85), dpi=300)
    left, right = 0.18, 0.985
    full_w = right - left
    heat_w = full_w * 0.74
    heat_left = left + (full_w - heat_w) / 2.0

    ax_h = fig.add_axes([heat_left, 0.590, heat_w, 0.335])
    im = ax_h.imshow(heat, aspect="auto", cmap=cmap, vmin=0.0, vmax=max(heat.max(), 1e-9))
    ax_h.set_xticks(np.arange(len(block_labels)))
    ax_h.set_xticklabels(block_labels, fontweight=600)
    ax_h.set_yticks(np.arange(len(row_labels)))
    ax_h.set_yticklabels(row_labels, fontweight=600)
    ax_h.tick_params(axis="both", length=0, pad=3, labelsize=9.5)
    ax_h.set_title("Observed-vs-null separation", loc="left", fontweight=600, pad=7)

    ax_h.set_xticks(np.arange(-0.5, len(block_labels), 1), minor=True)
    ax_h.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax_h.grid(which="minor", color="white", linewidth=1.2)
    ax_h.tick_params(which="minor", bottom=False, left=False)
    for spine in ax_h.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#4b4b4b")

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat[i, j]
            color = "white" if val > heat.max() * 0.62 else "#262626"
            ax_h.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8.8, color=color)

    cax = fig.add_axes([heat_left, 0.520, heat_w, 0.030])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.6)
    cax.text(0.02, 0.50, "Low", ha="left", va="center", transform=cax.transAxes, fontsize=9.5, color="#222222")
    cax.text(0.50, 0.50, r"$\bar{\Delta}$", ha="center", va="center", transform=cax.transAxes, fontsize=10.0, color="#222222")
    cax.text(0.98, 0.50, "High", ha="right", va="center", transform=cax.transAxes, fontsize=9.5, color="#222222")

    gap = 0.025
    ax_b_w = full_w * 0.57
    ax_c_w = full_w - ax_b_w - gap
    ax_b = fig.add_axes([left, 0.165, ax_b_w, 0.295])
    ax_c = fig.add_axes([left + ax_b_w + gap, 0.165, ax_c_w, 0.295])

    y = np.arange(len(rows))
    nc_mask = np.array([r["regime"] == "NC" for r in rows])
    c_mask = ~nc_mask
    colors = np.where(nc_mask, "#4f97b3", "#b94b5f")
    for yi, lo, hi in zip(y, np.minimum(observed, null), np.maximum(observed, null)):
        ax_b.hlines(yi, lo, hi, color="#c87982", linewidth=1.15, alpha=0.70, zorder=1)
    ax_b.scatter(null, y, s=32, color="#f1eee9", edgecolor="#5a5a5a", linewidth=0.60, zorder=2)
    ax_b.scatter(observed[nc_mask], y[nc_mask], s=38, color="#4f97b3", edgecolor="#1f1f1f", linewidth=0.45, zorder=3)
    ax_b.scatter(observed[c_mask], y[c_mask], s=38, color="#b94b5f", edgecolor="#1f1f1f", linewidth=0.45, zorder=3)
    ax_b.set_title(r"Observed vs. null $\bar{D}$", loc="left", fontweight=600, pad=7)
    ax_b.set_xlabel(r"$\bar{D}$", labelpad=3)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(row_labels, fontsize=9.5, fontweight=600)
    ax_b.invert_yaxis()
    ax_b.grid(axis="x", color="#e6e1dd", linewidth=0.8)
    ax_b.set_xlim(-0.025, 1.08)
    ax_b.set_ylim(len(rows) - 0.45, -0.55)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#4f97b3", markeredgecolor="#1f1f1f", markersize=6.5, label="Obs. NC"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#b94b5f", markeredgecolor="#1f1f1f", markersize=6.5, label="Obs. C"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f1eee9", markeredgecolor="#5a5a5a", markersize=6.0, label="Null"),
    ]
    ax_b.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.30),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.9,
        borderpad=0.0,
    )

    yerr = np.vstack([delta - ci_low, ci_high - delta])
    ax_c.errorbar(
        delta,
        y,
        xerr=yerr,
        fmt="none",
        ecolor="#777777",
        elinewidth=1.1,
        capsize=2.8,
        zorder=1,
    )
    ax_c.scatter(delta, y, s=38, c=colors, edgecolor="#1f1f1f", linewidth=0.45, zorder=3)
    ax_c.axvline(0, color="#777777", linewidth=1.0, linestyle=":")
    ax_c.set_title(r"Separation $\bar{\Delta}$", loc="left", fontweight=600, pad=7)
    ax_c.set_xlabel(r"$\bar{\Delta}$", labelpad=3)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([])
    ax_c.invert_yaxis()
    ax_c.grid(axis="x", color="#e6e1dd", linewidth=0.8)
    ax_c.set_xlim(0, max(ci_high) * 1.22)
    ax_c.text(
        0.50,
        -0.30,
        q_note,
        ha="center",
        va="top",
        transform=ax_c.transAxes,
        fontsize=9.0,
        color="#333333",
    )

    for ax in (ax_b, ax_c):
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#4b4b4b")

    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(OUT_STEM.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    print(OUT_STEM.with_suffix(".pdf"))
    print(OUT_STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()

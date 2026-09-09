#!/usr/bin/env python3
"""Build a mirrored full-layer grid comparing Hidden vs Raw/ModelNorm across families."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
OUT_STEM = "hidden_raw_modelnorm_mirrored_grid"


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


def _partition_boundaries(block_def: dict) -> list[int]:
    layers_by_block = block_def["layers_by_block"]
    return [min(layers_by_block[name]) for name in ("early", "mid", "late", "last")]


def _rel_profile(values: list[float]) -> tuple[list[float], float]:
    peak = max(values) if values else 0.0
    rel = [v / peak if peak else 0.0 for v in values]
    return rel, peak


def _layer_ticks(num_layers: int) -> tuple[list[int], list[str]]:
    if num_layers <= 8:
        ticks = list(range(num_layers))
    else:
        mid = (num_layers - 1) // 2
        ticks = [0, mid, num_layers - 1]
    return ticks, [f"L{t + 1}" for t in ticks]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = [_load_json(case.summary_path) for case in CASES]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 7.8,
            "axes.titlesize": 9.2,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 8.4,
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 2.95), dpi=300)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.835, bottom=0.16, wspace=0.08, hspace=0.14)

    for col, (case, payload) in enumerate(zip(CASES, payloads)):
        color = FAMILY_COLORS[case.family]
        layers = list(range(len(payload["hidden"]["hidden_cosine"]["layerwise_mean"])))
        tick_pos, tick_labels = _layer_ticks(len(layers))
        boundaries = _partition_boundaries(payload["block_definition"])

        sim_ax = axes[0, col]
        div_ax = axes[1, col]

        raw_j5_rel, raw_j5_peak = _rel_profile(payload["modes"]["raw"]["jaccard_top5"]["layerwise_mean"])
        mn_j5_rel, mn_j5_peak = _rel_profile(payload["modes"]["model_norm"]["jaccard_top5"]["layerwise_mean"])
        hidden_cos_rel, hidden_cos_peak = _rel_profile(payload["hidden"]["hidden_cosine"]["layerwise_mean"])

        raw_jsd_rel, raw_jsd_peak = _rel_profile(payload["modes"]["raw"]["js"]["layerwise_mean"])
        mn_jsd_rel, mn_jsd_peak = _rel_profile(payload["modes"]["model_norm"]["js"]["layerwise_mean"])
        hidden_l2_rel, hidden_l2_peak = _rel_profile(payload["hidden"]["hidden_l2"]["layerwise_mean"])

        for ax in (sim_ax, div_ax):
            ax.axhline(0.0, color="#777777", linewidth=0.9, zorder=2)
            for boundary in boundaries:
                ax.axvline(boundary - 0.5, color="#bfb6ae", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
            ax.grid(axis="y", color="#e2dbd5", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.65, layers[-1] + 0.65)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks(tick_pos, tick_labels)
            for label in ax.get_xticklabels():
                label.set_fontweight("semibold")
            for spine in ax.spines.values():
                spine.set_color("#666666")
                spine.set_linewidth(0.8)

        # Similarity: top = Raw/MN J@5, bottom = Hidden cosine
        sim_ax.bar(layers, raw_j5_rel, width=0.82, color=color, alpha=0.88, edgecolor="none", zorder=3)
        sim_ax.bar(layers, mn_j5_rel, width=0.52, color=color, alpha=0.30, edgecolor="#222222", linewidth=0.65, zorder=4)
        sim_ax.bar(layers, [-v for v in hidden_cos_rel], width=0.82, color="#6a6a6a", alpha=0.28, edgecolor="#444444", linewidth=0.35, zorder=3)

        # Divergence: top = Raw/MN JSD, bottom = Hidden L2
        div_ax.bar(layers, raw_jsd_rel, width=0.82, color=color, alpha=0.88, edgecolor="none", zorder=3)
        div_ax.bar(layers, mn_jsd_rel, width=0.52, color=color, alpha=0.30, edgecolor="#222222", linewidth=0.65, zorder=4)
        div_ax.bar(layers, [-v for v in hidden_l2_rel], width=0.82, color="#6a6a6a", alpha=0.28, edgecolor="#444444", linewidth=0.35, zorder=3)

        sim_ax.set_title(case.display, color=color, fontweight="bold", pad=2)
        if col != 0:
            sim_ax.tick_params(labelleft=False, left=False)
            div_ax.tick_params(labelleft=False, left=False)
        if col == 0:
            sim_ax.tick_params(labelbottom=False)
        else:
            sim_ax.tick_params(labelbottom=False)

        sim_ax.text(
            0.02, 0.90, "Raw (R) / MN", transform=sim_ax.transAxes, ha="left", va="center", fontsize=7.6, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.12},
        )
        sim_ax.text(
            0.02, 0.10, "Hidden", transform=sim_ax.transAxes, ha="left", va="center", fontsize=7.6, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.12},
        )
        div_ax.text(
            0.02, 0.90, "Raw (R) / MN", transform=div_ax.transAxes, ha="left", va="center", fontsize=7.6, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.12},
        )
        div_ax.text(
            0.02, 0.10, "Hidden", transform=div_ax.transAxes, ha="left", va="center", fontsize=7.6, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.12},
        )

        sim_ax.text(
            0.98, 0.92, f"R {raw_j5_peak:.3f} | MN {mn_j5_peak:.3f}",
            transform=sim_ax.transAxes, ha="right", va="center", fontsize=7.0,
            color=color, fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.10},
        )
        sim_ax.text(
            0.98, 0.08, f"Cos {hidden_cos_peak:.3f}",
            transform=sim_ax.transAxes, ha="right", va="center", fontsize=7.0,
            color="#4d4d4d", fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.10},
        )
        div_ax.text(
            0.98, 0.92, f"R {raw_jsd_peak:.3f} | MN {mn_jsd_peak:.3f}",
            transform=div_ax.transAxes, ha="right", va="center", fontsize=7.0,
            color=color, fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.10},
        )
        div_ax.text(
            0.98, 0.08, f"L2 {hidden_l2_peak:.3f}",
            transform=div_ax.transAxes, ha="right", va="center", fontsize=7.0,
            color="#4d4d4d", fontweight="semibold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.10},
        )

    axes[0, 0].set_ylabel(r"$\pm$ J@5 / Cosine", fontweight="bold", labelpad=3)
    axes[1, 0].set_ylabel(r"$\pm$ JSD / L2", fontweight="bold", labelpad=3)

    handles = [
        Patch(facecolor="#777777", edgecolor="none", alpha=0.88, label="Raw (R) profile"),
        Patch(facecolor="#777777", edgecolor="#222222", linewidth=0.65, alpha=0.30, label="ModelNorm (MN) profile"),
        Patch(facecolor="#6a6a6a", edgecolor="#444444", linewidth=0.4, alpha=0.28, label="Hidden profile"),
        Line2D([0], [0], color="#bfb6ae", linewidth=1.0, linestyle=(0, (3, 3)), label="Partition boundary"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.975),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.0,
    )

    caption = (
        "\\caption{"
        "\\textbf{Mirrored full-layer comparison of hidden-state and decoded depth profiles across families.} "
        "Columns show representative Qwen, LLaMA, and Pythia comparisons at full layer resolution. "
        "Top row: similarity-style profiles, with Raw and ModelNorm J@$5$ mirrored against hidden cosine similarity. "
        "Bottom row: divergence-style profiles, with Raw and ModelNorm JSD mirrored against hidden $L_2$. "
        "Within each panel, profiles are scaled by their own peak value to emphasize shape, peak location, and late-layer concentration rather than absolute scale; peak values are reported inside each panel to preserve the original magnitudes."
        "}\n"
    )
    (OUT_DIR / f"{OUT_STEM}_caption.tex").write_text(caption, encoding="utf-8")

    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

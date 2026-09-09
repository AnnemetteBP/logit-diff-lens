#!/usr/bin/env python3
"""Compact heatmap summary for hidden, Raw, and ModelNorm depth profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Figures" / "PaperComparisons"
OUT_STEM = "hidden_raw_modelnorm_compact_heatmap"
BLOCK_ORDER = ("first", "early", "mid", "late", "last")
BLOCK_LABELS = ("F", "E", "M", "L", "Last")


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    family_color: str
    summary_path: Path


CASES = (
    CaseSpec(
        key="qwen_financial",
        title="Qwen Financial",
        family_color="#b94b5f",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "qwen" / "risky" / "summaries" / "mode_specific_summary.json",
    ),
    CaseSpec(
        key="llama_1bit",
        title="LLaMA 1.58-bit",
        family_color="#4f97b3",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "llama" / "hf1bit" / "summaries" / "mode_specific_summary.json",
    ),
    CaseSpec(
        key="pythia_71k",
        title="Pythia 2.8B 71k",
        family_color="#efb6ad",
        summary_path=ROOT / "ucloud_logitdiff" / "derived" / "pythia" / "2p8b_71k_mid" / "summaries" / "mode_specific_summary.json",
    ),
)


SIM_ROWS = (
    ("hidden", "Cosine", "hidden_cosine"),
    ("raw", "Raw J@5", "jaccard_top5"),
    ("mn", "MN J@5", "jaccard_top5"),
)

DIV_ROWS = (
    ("hidden", "Hidden L2", "hidden_l2"),
    ("raw", "Raw JSD", "js"),
    ("mn", "MN JSD", "js"),
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _block_mean(values: list[float], layers: list[int]) -> float:
    return sum(values[idx] for idx in layers) / float(len(layers))


def _selected_blocks(payload: dict, source: str, metric: str) -> list[float]:
    blocks = payload["block_definition"]["layers_by_block"]
    if source == "hidden":
        layerwise = payload["hidden"][metric]["layerwise_mean"]
    elif source == "raw":
        layerwise = payload["modes"]["raw"][metric]["layerwise_mean"]
    else:
        layerwise = payload["modes"]["model_norm"][metric]["layerwise_mean"]
    return [_block_mean(layerwise, blocks[name]) for name in BLOCK_ORDER]


def _normalize_row(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-12:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _build_matrix(rowspecs: tuple[tuple[str, str, str], ...], payloads: dict[str, dict]) -> tuple[np.ndarray, list[str]]:
    row_labels: list[str] = []
    matrix: list[list[float]] = []
    for source, label, metric in rowspecs:
        row: list[float] = []
        for case in CASES:
            vals = _selected_blocks(payloads[case.key], source, metric)
            vals = _normalize_row(vals)
            row.extend(vals)
        matrix.append(row)
        row_labels.append(label)
    return np.array(matrix, dtype=float), row_labels


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9.4,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw_heatmap(ax, data: np.ndarray, row_labels: list[str], title: str, case_titles: list[str], family_colors: list[str]) -> None:
    im = ax.imshow(data, aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontweight="semibold")
    xtick_positions = np.arange(data.shape[1])
    xtick_labels = list(BLOCK_LABELS) * len(CASES)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    ax.tick_params(length=0)

    # block separators
    for pos in (4.5, 9.5):
        ax.axvline(pos, color="#c8c2bc", linewidth=1.1)
    for pos in np.arange(-0.5, data.shape[1], 1):
        ax.axvline(pos, color="#f4f1ee", linewidth=0.45, alpha=0.95)
    for pos in np.arange(-0.5, data.shape[0], 1):
        ax.axhline(pos, color="#f4f1ee", linewidth=0.45, alpha=0.95)

    for idx, (title_text, color) in enumerate(zip(case_titles, family_colors)):
        center = idx * 5 + 2
        ax.text(center, -1.05, title_text, ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=color)

    ax.set_title(title, loc="left", fontweight="bold", pad=18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    payloads = {case.key: _load_json(case.summary_path) for case in CASES}
    sim_matrix, sim_labels = _build_matrix(SIM_ROWS, payloads)
    div_matrix, div_labels = _build_matrix(DIV_ROWS, payloads)

    fig, axes = plt.subplots(2, 1, figsize=(7.15, 3.7), dpi=300)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.16, hspace=0.52)

    case_titles = [case.title for case in CASES]
    family_colors = [case.family_color for case in CASES]

    im1 = _draw_heatmap(axes[0], sim_matrix, sim_labels, "Similarity Views", case_titles, family_colors)
    _draw_heatmap(axes[1], div_matrix, div_labels, "Divergence Views", case_titles, family_colors)

    fig.text(0.49, 0.06, r"$\mathbf{Layer\ Partition}$", ha="center", va="center", fontsize=10.2)
    cax = fig.add_axes([0.968, 0.16, 0.012, 0.74])
    cb = fig.colorbar(im1, cax=cax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["low", "mid", "high"])
    cb.ax.tick_params(labelsize=8.2)

    caption = (
        "\\caption{"
        "\\textbf{Compact hidden/Raw/ModelNorm depth summary.} "
        "Each block of five columns corresponds to one model comparison and the First, Early, Mid, Late, and Last summaries. "
        "Rows are independently normalized within each model-view combination to compare profile shape rather than absolute magnitude. "
        "Top: hidden cosine similarity, Raw J@5, and ModelNorm J@5. Bottom: hidden L2, Raw JSD, and ModelNorm JSD. "
        "The figure provides a compact view of where hidden and decoded comparisons align across depth and where the readout changes the apparent profile."
        "}\n"
    )
    (OUT_DIR / f"{OUT_STEM}_caption.tex").write_text(caption, encoding="utf-8")

    summary = {"similarity": sim_matrix.tolist(), "divergence": div_matrix.tolist(), "blocks": BLOCK_LABELS}
    (OUT_DIR / f"{OUT_STEM}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()

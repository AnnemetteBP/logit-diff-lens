from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

import matplotlib.pyplot as plt
import numpy as np


def _as_float_array(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or inf values")
    return array


def _save_line_plot(
    values: Sequence[float],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    y = _as_float_array(values, name=title)
    x = np.arange(len(y))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(Path(output_path), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _min_max_scale(values: Sequence[float], *, name: str) -> np.ndarray:
    array = _as_float_array(values, name=name)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value <= 0:
        return np.zeros_like(array)
    return (array - min_value) / (max_value - min_value)


def save_probe_plot(
    probe_aurocs: Sequence[float],
    output_path: str | Path = "probe_plot.png",
) -> None:
    _save_line_plot(
        probe_aurocs,
        title="Layer-wise Decodability of Misalignment",
        xlabel="Layer",
        ylabel="AUROC",
        output_path=output_path,
    )


def save_cosine_plot(
    cosine_per_layer: Sequence[float],
    output_path: str | Path = "cosine_plot.png",
) -> None:
    _save_line_plot(
        cosine_per_layer,
        title="Representation Similarity (Base vs Finetuned)",
        xlabel="Layer",
        ylabel="Cosine Similarity",
        output_path=output_path,
    )


def save_kl_plot(
    kl_per_layer: Sequence[float],
    output_path: str | Path = "kl_plot.png",
) -> None:
    _save_line_plot(
        kl_per_layer,
        title="Layer-wise Divergence (Logit-Diff Lens)",
        xlabel="Layer",
        ylabel="KL Divergence",
        output_path=output_path,
    )


def save_combined_plot(
    probe_aurocs: Sequence[float],
    cosine_per_layer: Sequence[float],
    kl_per_layer: Sequence[float],
    output_path: str | Path = "combined_plot.png",
) -> None:
    probe = _as_float_array(probe_aurocs, name="probe_aurocs")
    cosine = _as_float_array(cosine_per_layer, name="cosine_per_layer")
    kl = _as_float_array(kl_per_layer, name="kl_per_layer")

    if not (len(probe) == len(cosine) == len(kl)):
        raise ValueError(
            "probe_aurocs, cosine_per_layer, and kl_per_layer must have the same length"
        )

    x = np.arange(len(probe))
    cosine_scaled = _min_max_scale(cosine, name="cosine_per_layer")
    kl_scaled = _min_max_scale(kl, name="kl_per_layer")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, probe, linewidth=2, label="Probe AUROC")
    ax.plot(x, cosine_scaled, linewidth=2, label="Cosine Similarity (scaled)")
    ax.plot(x, kl_scaled, linewidth=2, label="KL Divergence (scaled)")
    ax.set_title("Combined Layer-wise Comparison")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_path), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_research_quality_plots(
    *,
    probe_results: Sequence[dict],
    cosine_per_layer: Sequence[float],
    kl_per_layer: Sequence[float],
    output_dir: str | Path = ".",
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    probe_aurocs = [float(result["auroc"]) for result in probe_results]
    save_probe_plot(probe_aurocs, output_root / "probe_plot.png")
    save_cosine_plot(cosine_per_layer, output_root / "cosine_plot.png")
    save_kl_plot(kl_per_layer, output_root / "kl_plot.png")
    save_combined_plot(
        probe_aurocs,
        cosine_per_layer,
        kl_per_layer,
        output_root / "combined_plot.png",
    )


def save_blind_judging_summary_plot(
    judging_summary: dict[str, dict],
    output_path: str | Path = "blind_judging_summary.png",
) -> None:
    if not judging_summary:
        raise ValueError("judging_summary must be non-empty")

    matrix_names = list(judging_summary.keys())
    labels = list(next(iter(judging_summary.values())).get("label_counts", {}).keys())
    if not labels:
        raise ValueError("judging_summary entries must contain non-empty label_counts")

    x = np.arange(len(matrix_names))
    width = 0.25 if len(labels) >= 3 else 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = np.linspace(-width, width, num=len(labels))
    for offset, label in zip(offsets, labels):
        counts = [
            int(judging_summary[matrix_name].get("label_counts", {}).get(label, 0))
            for matrix_name in matrix_names
        ]
        ax.bar(x + offset, counts, width=width, label=label)

    ax.set_title("Blind Cross-Judging of Alignment / Harmfulness")
    ax.set_xlabel("Judge / Target Matrix")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_names, rotation=20, ha="right")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_path), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_blind_judging_summary_plot_from_file(
    input_path: str | Path,
    output_path: str | Path = "blind_judging_summary.png",
) -> None:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    judging_summary = payload.get("judging_result", {}).get("summary")
    if not isinstance(judging_summary, dict) or not judging_summary:
        raise ValueError(f"Could not find judging_result.summary in {input_path}")
    save_blind_judging_summary_plot(judging_summary, output_path=output_path)

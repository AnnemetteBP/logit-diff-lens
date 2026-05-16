from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


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


def ensure_numpy(x: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) if not isinstance(x, np.ndarray) else x.astype(float, copy=False)


def compute_auroc_per_layer(
    projection_scores_per_layer: Sequence[Sequence[float] | dict],
    labels: Sequence[int] | np.ndarray,
) -> np.ndarray:
    label_array = ensure_numpy(labels).astype(int, copy=False)
    aucs = []
    for scores in projection_scores_per_layer:
        layer_scores = scores.get("scores") if isinstance(scores, dict) else scores
        score_array = ensure_numpy(layer_scores)
        if score_array.ndim != 1:
            raise ValueError("Each layer of projection_scores_per_layer must be 1D")
        if score_array.shape[0] != label_array.shape[0]:
            raise ValueError(
                "Each layer of projection_scores_per_layer must match the number of labels"
            )
        if len(np.unique(label_array)) < 2:
            aucs.append(np.nan)
            continue
        try:
            auc = roc_auc_score(label_array, score_array)
        except ValueError:
            auc = np.nan
        aucs.append(float(auc))
    return np.asarray(aucs, dtype=float)


def _extract_mean_values_per_layer(
    layer_entries: Sequence[dict],
    *,
    value_key: str,
) -> np.ndarray:
    values = []
    for entry in layer_entries:
        layer_values = ensure_numpy(entry[value_key])
        if layer_values.ndim != 1 or layer_values.size == 0:
            raise ValueError(f"{value_key} must contain non-empty 1D arrays per layer")
        values.append(float(np.mean(layer_values)))
    return np.asarray(values, dtype=float)


def _extract_top1_pca_variance(pca_results: Sequence[dict]) -> np.ndarray:
    top1 = []
    for entry in pca_results:
        explained = ensure_numpy(entry["explained_variance_ratio"])
        if explained.ndim != 1 or explained.size == 0:
            raise ValueError("explained_variance_ratio must be a non-empty 1D array")
        top1.append(float(explained[0]))
    return np.asarray(top1, dtype=float)


def _summary_series_from_prism_summary(
    summary: dict,
    *,
    field: str,
) -> np.ndarray:
    ordered_keys = sorted(summary.keys(), key=lambda key: int(key))
    return np.asarray([float(summary[key][field]) for key in ordered_keys], dtype=float)


def _extract_prism_norms(
    prism_payload: dict,
    *,
    source_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    if "summary" in prism_payload:
        summary = prism_payload["summary"]
        return (
            _summary_series_from_prism_summary(summary, field="attn_norm"),
            _summary_series_from_prism_summary(summary, field="mlp_norm"),
        )

    if "rows" not in prism_payload:
        raise ValueError("prism_payload must contain either summary or rows")

    attn_rows = []
    mlp_rows = []
    for row in prism_payload["rows"]:
        prism_result = row.get(source_key)
        if not isinstance(prism_result, dict) or "summary" not in prism_result:
            continue
        attn_rows.append(_summary_series_from_prism_summary(prism_result["summary"], field="attn_norm"))
        mlp_rows.append(_summary_series_from_prism_summary(prism_result["summary"], field="mlp_norm"))

    if not attn_rows or not mlp_rows:
        raise ValueError(f"Could not extract prism summaries from source_key={source_key}")

    return (
        np.mean(np.stack(attn_rows, axis=0), axis=0),
        np.mean(np.stack(mlp_rows, axis=0), axis=0),
    )


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


def save_latent_to_output_figure(
    *,
    kl_per_layer: Sequence[float],
    projection_scores_per_layer: Sequence[Sequence[float] | dict],
    labels: Sequence[int] | np.ndarray,
    pca_variance_per_layer: Sequence[Sequence[float] | dict],
    attn_contrib_norm: Sequence[float],
    mlp_contrib_norm: Sequence[float],
    output_path: str | Path = "latent_to_output_figure.png",
) -> None:
    kl_per_layer_array = _as_float_array(kl_per_layer, name="kl_per_layer")
    attn_contrib_norm_array = _as_float_array(attn_contrib_norm, name="attn_contrib_norm")
    mlp_contrib_norm_array = _as_float_array(mlp_contrib_norm, name="mlp_contrib_norm")
    auc_per_layer = compute_auroc_per_layer(projection_scores_per_layer, labels)

    pca_top1 = []
    for entry in pca_variance_per_layer:
        explained = entry.get("explained_variance_ratio") if isinstance(entry, dict) else entry
        explained_array = ensure_numpy(explained)
        if explained_array.ndim != 1 or explained_array.size == 0:
            raise ValueError("Each layer of pca_variance_per_layer must be a non-empty 1D array")
        pca_top1.append(float(explained_array[0]))
    pca_top1_array = np.asarray(pca_top1, dtype=float)

    n_layers = len(kl_per_layer_array)
    if not (
        len(auc_per_layer) == len(pca_top1_array) == len(attn_contrib_norm_array) == len(mlp_contrib_norm_array) == n_layers
    ):
        raise ValueError("All latent-to-output plotting inputs must have the same layer length")

    layers = np.arange(n_layers)
    attn_norm = attn_contrib_norm_array / (float(attn_contrib_norm_array.max()) + 1e-8)
    mlp_norm = mlp_contrib_norm_array / (float(mlp_contrib_norm_array.max()) + 1e-8)
    kl_norm = kl_per_layer_array / (float(kl_per_layer_array.max()) + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(layers, kl_per_layer_array, linewidth=2)
    ax.set_title("Layer-wise Logit Difference (KL / MDS)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL divergence")
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(layers, auc_per_layer, linewidth=2)
    ax.set_title("Latent Projection -> Behavior (AUROC)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(layers, pca_top1_array, linewidth=2)
    ax.set_title("Top PCA Variance (Delta_hidden)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Explained variance")
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(layers, kl_norm, linewidth=2, label="Delta logits (normalized)")
    ax.plot(layers, attn_norm, linewidth=2, label="Attention contribution")
    ax.plot(layers, mlp_norm, linewidth=2, label="MLP contribution")
    ax.set_title("Prism Decomposition (Normalized)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized magnitude")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(Path(output_path), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_latent_to_output_figure_from_results(
    latent_shift_result: dict,
    *,
    labels: Sequence[int] | np.ndarray,
    prism_result: dict | None = None,
    prism_source: str = "base_prism",
    output_path: str | Path = "latent_to_output_figure.png",
) -> None:
    kl_per_layer = _extract_mean_values_per_layer(
        latent_shift_result["kl_divergence"],
        value_key="values",
    )
    projection_scores_per_layer = latent_shift_result["projection_scores"]
    pca_variance_per_layer = latent_shift_result["pca_results"]

    if prism_result is None:
        attn_contrib_norm = np.zeros_like(kl_per_layer)
        mlp_contrib_norm = np.zeros_like(kl_per_layer)
    else:
        attn_contrib_norm, mlp_contrib_norm = _extract_prism_norms(
            prism_result,
            source_key=prism_source,
        )

    save_latent_to_output_figure(
        kl_per_layer=kl_per_layer,
        projection_scores_per_layer=projection_scores_per_layer,
        labels=labels,
        pca_variance_per_layer=pca_variance_per_layer,
        attn_contrib_norm=attn_contrib_norm,
        mlp_contrib_norm=mlp_contrib_norm,
        output_path=output_path,
    )


def save_latent_to_output_figure_from_files(
    latent_shift_result_path: str | Path,
    *,
    labels: Sequence[int] | np.ndarray,
    prism_result_path: str | Path | None = None,
    prism_source: str = "base_prism",
    output_path: str | Path = "latent_to_output_figure.png",
) -> None:
    latent_shift_result = json.loads(Path(latent_shift_result_path).read_text(encoding="utf-8"))
    prism_result = None
    if prism_result_path is not None:
        prism_result = json.loads(Path(prism_result_path).read_text(encoding="utf-8"))
    save_latent_to_output_figure_from_results(
        latent_shift_result,
        labels=labels,
        prism_result=prism_result,
        prism_source=prism_source,
        output_path=output_path,
    )

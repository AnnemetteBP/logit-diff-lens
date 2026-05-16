from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


def save_collected_activation_dataset_plots(
    *,
    output_root: Path,
    layers: np.ndarray,
    logit_norms: np.ndarray,
    logit_norm_mode: str,
    labels: np.ndarray,
    auroc_array: np.ndarray,
    pca_top1_array: np.ndarray,
    heatmap_matrix: np.ndarray,
    cumulative_variance: Sequence[np.ndarray],
    selected_layers: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, logit_norms, linewidth=2)
    if logit_norm_mode == "final_repeated":
        ax.set_title("Final Logit Norm Proxy (Repeated Across Layers)")
    else:
        ax.set_title("Logit Norm (MDS Proxy)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 norm")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_root / "logit_norm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, auroc_array, linewidth=2)
    if len(np.unique(labels)) < 2:
        ax.set_title("Layer-wise AUROC (single-class labels -> chance baseline)")
    else:
        ax.set_title("Layer-wise AUROC")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_root / "auroc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, pca_top1_array, linewidth=2)
    ax.set_title("Top-1 PCA Variance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Explained variance")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_root / "pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(heatmap_matrix, aspect="auto")
    ax.set_title("Layer-wise Summary Heatmap")
    ax.set_xlabel("Layer")
    ax.set_yticks([0, 1, 2], labels=["Logit", "AUROC", "PCA"])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_root / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for layer_idx in selected_layers:
        ax.plot(cumulative_variance[layer_idx], linewidth=2, label=f"Layer {layer_idx}")
    ax.set_title("Cumulative PCA Variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "pca_cumulative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _normalize_rows_for_heatmap(matrix: np.ndarray) -> np.ndarray:
    normalized_rows = []
    for row in matrix:
        row = np.asarray(row, dtype=float)
        min_value = float(np.min(row))
        max_value = float(np.max(row))
        if max_value - min_value <= 0:
            normalized_rows.append(np.zeros_like(row))
        else:
            normalized_rows.append((row - min_value) / (max_value - min_value))
    return np.vstack(normalized_rows)


def save_paired_collected_activation_dataset_plots(
    *,
    output_root: Path,
    layers: np.ndarray,
    logit_diff: np.ndarray,
    latent_auroc_array: np.ndarray,
    logit_auroc_array: np.ndarray,
    pca_top1_array: np.ndarray,
    combined_matrix: np.ndarray,
    heatmap_matrix: np.ndarray,
    attn_diff: np.ndarray,
    mlp_diff: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, logit_diff, linewidth=2)
    ax.set_title("Delta Logits")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean ||delta_z||_2")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_root / "delta_logits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, latent_auroc_array, linewidth=2)
    ax.set_title("Latent -> Behavior AUROC")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_root / "latent_auroc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, logit_auroc_array, linewidth=2)
    ax.set_title("Logit -> Behavior AUROC")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_root / "logit_auroc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, pca_top1_array, linewidth=2)
    ax.set_title("Top-1 PCA Variance of Delta Hidden")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Explained variance")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_root / "pca_top1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    norm_combined = _normalize_rows_for_heatmap(combined_matrix)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, norm_combined[0], linewidth=2, label="Delta logits")
    ax.plot(layers, norm_combined[1], linewidth=2, label="Latent AUROC")
    ax.plot(layers, norm_combined[2], linewidth=2, label="Logit AUROC")
    ax.plot(layers, norm_combined[3], linewidth=2, label="PCA top1")
    ax.set_title("Combined Signal Overlay")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(heatmap_matrix, aspect="auto")
    ax.set_title("Paired Signal Heatmap")
    ax.set_xlabel("Layer")
    ax.set_yticks(
        [0, 1, 2, 3, 4, 5],
        labels=["Delta logits", "Latent->harm", "Logit->harm", "PCA", "Attention", "MLP"],
    )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_root / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, attn_diff, linewidth=2, label="Attention")
    ax.plot(layers, mlp_diff, linewidth=2, label="MLP")
    ax.set_title("Prism Component Differences")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean ||component_ft - component_base||")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "prism.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_full_prism_analysis_plots(
    *,
    output_root: Path,
    layers: np.ndarray,
    hidden_diff: Sequence[float],
    logit_diff: Sequence[float],
    final_logit_diff: float,
    auroc_values: Sequence[float],
    logit_auroc: Sequence[float],
    pca_top1: Sequence[float],
    total_diff: Sequence[float],
    emb_contrib_plot: Sequence[float],
    attn_contrib: Sequence[float],
    mlp_contrib: Sequence[float],
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(layers, hidden_diff)
    plt.title("Delta hidden")
    plt.xlabel("Layer")
    plt.ylabel("Mean ||delta_h||_2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "delta_hidden.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(layers, logit_diff)
    plt.title("Delta logits")
    plt.xlabel("Layer")
    plt.ylabel("Mean ||delta_z||_2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "delta_logits.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.bar(["final logits"], [final_logit_diff])
    plt.title("Final logits difference")
    plt.ylabel("Mean ||logits_ft - logits_base||_2")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(output_root / "final_logits_difference.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(layers, auroc_values)
    plt.title("Latent -> harmfulness AUROC")
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "latent_to_harm.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(layers, logit_auroc)
    plt.title("Logit -> harmfulness AUROC")
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "logit_to_harm.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(layers, pca_top1)
    plt.title("Top-1 PCA variance of hidden deltas")
    plt.xlabel("Layer")
    plt.ylabel("Explained variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "pca_top1.png", dpi=150, bbox_inches="tight")
    plt.close()

    def norm(x: Sequence[float]) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if np.max(arr) - np.min(arr) < 1e-12:
            return np.zeros_like(arr)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    plt.figure(figsize=(9, 5))
    plt.plot(layers, norm(total_diff), label="Δ logits")
    plt.plot(layers, norm(attn_contrib), label="Attention contribution")
    plt.plot(layers, norm(mlp_contrib), label="MLP contribution")
    plt.xlabel("Layer")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "alignment.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(layers, emb_contrib_plot, label="embedding")
    plt.plot(layers, attn_contrib, label="attention")
    plt.plot(layers, mlp_contrib, label="MLP")
    plt.legend()
    plt.title("Full prism contributions")
    plt.tight_layout()
    plt.savefig(output_root / "prism_contributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    ratio_emb = np.asarray(emb_contrib_plot) / (np.asarray(total_diff) + 1e-8)
    ratio_attn = np.asarray(attn_contrib) / (np.asarray(total_diff) + 1e-8)
    ratio_mlp = np.asarray(mlp_contrib) / (np.asarray(total_diff) + 1e-8)

    plt.figure(figsize=(9, 5))
    plt.plot(layers, total_diff, label="Δ logits")
    plt.plot(layers, np.asarray(emb_contrib_plot) + np.asarray(attn_contrib) + np.asarray(mlp_contrib), label="emb+attn+MLP")
    plt.legend()
    plt.title("Prism alignment")
    plt.tight_layout()
    plt.savefig(output_root / "prism_alignment.png", dpi=150, bbox_inches="tight")
    plt.close()

    prism_matrix = np.array([
        norm(total_diff),
        norm(emb_contrib_plot),
        norm(attn_contrib),
        norm(mlp_contrib),
        norm(ratio_emb),
        norm(ratio_attn),
        norm(ratio_mlp),
    ])
    plt.figure(figsize=(9, 5))
    plt.imshow(prism_matrix, aspect="auto")
    plt.colorbar()
    plt.yticks(
        [0, 1, 2, 3, 4, 5, 6],
        ["Δ logits", "embedding", "attention", "MLP", "emb ratio", "attn ratio", "MLP ratio"],
    )
    plt.title("Layer-wise prism analysis")
    plt.tight_layout()
    plt.savefig(output_root / "prism_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    mechanism_matrix = np.array([
        norm(total_diff),
        norm(attn_contrib),
        norm(mlp_contrib),
        norm(ratio_attn),
        norm(ratio_mlp),
    ])
    plt.figure(figsize=(9, 5))
    plt.imshow(mechanism_matrix, aspect="auto")
    plt.colorbar()
    plt.yticks([0, 1, 2, 3, 4], ["Δ logits", "attention", "MLP", "attn ratio", "MLP ratio"])
    plt.xlabel("Layer")
    plt.title("Layer-wise mechanism")
    plt.tight_layout()
    plt.savefig(output_root / "layer_analysis_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(layers, ratio_attn, label="attention ratio")
    plt.plot(layers, ratio_mlp, label="MLP ratio")
    plt.xlabel("Layer")
    plt.ylabel("Ratio")
    plt.legend()
    plt.title("Relative importance")
    plt.tight_layout()
    plt.savefig(output_root / "mechanism.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_group_token_divergence_plot(
    *,
    output_path: Path,
    per_row: Sequence[dict[str, Any]],
    group_id: str,
    layer_indices: Sequence[int],
    metric_label: str,
    global_max: float,
) -> None:
    fig = plt.figure(figsize=(max(14, max(len(row["input_tokens"]) for row in per_row) * 0.5), 14))
    outer = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 1, 1], hspace=0.5)
    heatmap_mappable = None

    for row_idx, row_data in enumerate(per_row):
        inner = outer[row_idx].subgridspec(nrows=3, ncols=1, height_ratios=[0.18, 1.0, 0.4], hspace=0.08)
        agg_ax = fig.add_subplot(inner[0])
        heat_ax = fig.add_subplot(inner[1])
        prob_ax = fig.add_subplot(inner[2])

        agg_ax.imshow(
            row_data["aggregate_position_score"][np.newaxis, :],
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=global_max,
        )
        agg_ax.set_yticks([0])
        agg_ax.set_yticklabels(["Late-layer\nmean"])
        agg_ax.set_xticks([])
        agg_ax.set_title(f"{row_data['title']} ({group_id})")

        heatmap_mappable = heat_ax.imshow(
            row_data["heatmap"],
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=global_max,
        )
        heat_ax.set_ylabel("Layer")
        heat_ax.set_yticks(np.arange(len(layer_indices)))
        heat_ax.set_yticklabels([str(layer_idx) for layer_idx in layer_indices])
        heat_ax.set_xticks(np.arange(len(row_data["input_tokens"])))
        heat_ax.set_xticklabels(row_data["input_tokens"], rotation=90, fontsize=8)
        heat_ax.set_xlabel("Input token")

        top_ax = heat_ax.twiny()
        top_ax.set_xlim(heat_ax.get_xlim())
        top_ax.set_xticks(np.arange(len(row_data["target_tokens"])))
        top_ax.set_xticklabels(row_data["target_tokens"], rotation=90, fontsize=8)
        top_ax.set_xlabel("Target token")

        positions = np.arange(len(row_data["target_tokens"]))
        prob_ax.plot(positions, row_data["probs_base"], label="base", linewidth=1.5)
        prob_ax.plot(positions, row_data["probs_ft"], label="finetuned", linewidth=1.5)
        prob_ax.set_ylabel("P(target)")
        prob_ax.set_xlabel("Position")
        prob_ax.grid(True)
        if row_idx == 0:
            prob_ax.legend()

    if heatmap_mappable is not None:
        fig.colorbar(heatmap_mappable, ax=fig.axes, shrink=0.98, pad=0.01, label=metric_label)
    fig.suptitle(f"Token-aligned late-layer {metric_label} and target-token probabilities", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.97, 0.985])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_covariance_ellipse(ax: plt.Axes, points: np.ndarray, *, color: str, n_std: float = 1.0) -> None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 0.0)
    if float(np.sum(evals)) <= 0.0:
        return
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    theta = np.linspace(0.0, 2.0 * np.pi, 256, dtype=np.float32)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    radii = n_std * np.sqrt(evals)
    ellipse = circle @ np.diag(radii) @ evecs.T + np.mean(pts, axis=0, keepdims=True)
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=color, linewidth=1.8, alpha=0.9)


def _plot_svd_geometry_panel(
    ax: plt.Axes,
    *,
    side_a_scores: np.ndarray,
    side_b_scores: np.ndarray,
    title: str,
    side_a_name: str,
    side_b_name: str,
) -> None:
    a = np.asarray(side_a_scores, dtype=np.float32)
    b = np.asarray(side_b_scores, dtype=np.float32)

    def _as_xy(scores: np.ndarray) -> np.ndarray:
        if scores.shape[1] == 1:
            return np.column_stack([scores[:, 0], np.zeros(scores.shape[0], dtype=np.float32)])
        return scores[:, :2]

    a_xy = _as_xy(a)
    b_xy = _as_xy(b)
    max_radius = float(
        max(
            np.linalg.norm(a_xy, axis=1).max(initial=0.0),
            np.linalg.norm(b_xy, axis=1).max(initial=0.0),
            1e-6,
        )
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 256, dtype=np.float32)
    ax.plot(max_radius * np.cos(theta), max_radius * np.sin(theta), color="0.65", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="0.85", linewidth=1.0)
    ax.axvline(0.0, color="0.85", linewidth=1.0)
    ax.scatter(a_xy[:, 0], a_xy[:, 1], s=22, alpha=0.75, color="#1f77b4", label=side_a_name)
    ax.scatter(b_xy[:, 0], b_xy[:, 1], s=22, alpha=0.75, color="#d62728", label=side_b_name)

    a_centroid = np.mean(a_xy, axis=0)
    b_centroid = np.mean(b_xy, axis=0)
    ax.scatter([a_centroid[0]], [a_centroid[1]], s=70, marker="x", color="#1f77b4")
    ax.scatter([b_centroid[0]], [b_centroid[1]], s=70, marker="x", color="#d62728")
    ax.plot([0.0, a_centroid[0]], [0.0, a_centroid[1]], color="#1f77b4", linestyle=":", linewidth=1.6)
    ax.plot([0.0, b_centroid[0]], [0.0, b_centroid[1]], color="#d62728", linestyle=":", linewidth=1.6)
    _draw_covariance_ellipse(ax, a_xy, color="#1f77b4")
    _draw_covariance_ellipse(ax, b_xy, color="#d62728")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_aspect("equal", adjustable="box")
    lim = max_radius * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)


def save_paired_activation_svd_plots(
    *,
    output_root: Path,
    layers: np.ndarray,
    hidden_rank_90: np.ndarray,
    logit_rank_90: np.ndarray,
    hidden_effective: np.ndarray,
    logit_effective: np.ndarray,
    top_k: int,
    hidden_summaries: Sequence[dict[str, Any]],
    logit_summaries: Sequence[dict[str, Any]],
    selected_layers: Sequence[int],
    hidden_base_by_layer: Sequence[np.ndarray],
    hidden_ft_by_layer: Sequence[np.ndarray],
    logit_base_by_layer: Sequence[np.ndarray],
    logit_ft_by_layer: Sequence[np.ndarray],
    side_a_name: str,
    side_b_name: str,
    compute_component_scores,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, hidden_rank_90, label="Hidden rank@90", linewidth=2)
    ax.plot(layers, logit_rank_90, label="Logit rank@90", linewidth=2)
    ax.plot(layers, hidden_effective, label="Hidden effective rank", linewidth=2, linestyle="--")
    ax.plot(layers, logit_effective, label="Logit effective rank", linewidth=2, linestyle="--")
    ax.set_title(f"SVD Rank Summary ({side_b_name} - {side_a_name})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rank")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "svd_rank_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    hidden_heatmap = np.zeros((len(hidden_summaries), top_k), dtype=np.float32)
    logit_heatmap = np.zeros((len(logit_summaries), top_k), dtype=np.float32)
    for layer_idx, summary in enumerate(hidden_summaries):
        values = np.asarray(summary["explained_variance_ratio"], dtype=np.float32)[:top_k]
        hidden_heatmap[layer_idx, : values.shape[0]] = values
    for layer_idx, summary in enumerate(logit_summaries):
        values = np.asarray(summary["explained_variance_ratio"], dtype=np.float32)[:top_k]
        logit_heatmap[layer_idx, : values.shape[0]] = values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    im0 = axes[0].imshow(hidden_heatmap, aspect="auto")
    axes[0].set_title("Hidden Δ explained variance")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Layer")
    axes[0].set_xticks(range(top_k), labels=[str(i + 1) for i in range(top_k)])
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(logit_heatmap, aspect="auto")
    axes[1].set_title("Logit Δ explained variance")
    axes[1].set_xlabel("Component")
    axes[1].set_xticks(range(top_k), labels=[str(i + 1) for i in range(top_k)])
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(output_root / "svd_explained_variance_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for layer_idx in selected_layers:
        hidden_curve = np.asarray(hidden_summaries[layer_idx]["cumulative_explained_variance"], dtype=float)
        logit_curve = np.asarray(logit_summaries[layer_idx]["cumulative_explained_variance"], dtype=float)
        axes[0].plot(np.arange(1, hidden_curve.size + 1), hidden_curve, linewidth=2, label=f"Layer {layer_idx}")
        axes[1].plot(np.arange(1, logit_curve.size + 1), logit_curve, linewidth=2, label=f"Layer {layer_idx}")
    axes[0].set_title("Hidden Δ cumulative explained variance")
    axes[1].set_title("Logit Δ cumulative explained variance")
    for ax in axes:
        ax.set_ylabel("Cumulative explained variance")
        ax.grid(True)
        ax.legend()
    axes[1].set_xlabel("Component")
    fig.tight_layout()
    fig.savefig(output_root / "svd_cumulative_selected_layers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if selected_layers:
        fig, axes = plt.subplots(2, len(selected_layers), figsize=(5 * len(selected_layers), 9), squeeze=False)
        for col, layer_idx in enumerate(selected_layers):
            hidden_vh = np.asarray(hidden_summaries[layer_idx]["top_right_singular_vectors_vh"], dtype=np.float32)
            logit_vh = np.asarray(logit_summaries[layer_idx]["top_right_singular_vectors_vh"], dtype=np.float32)
            hidden_scores_a = compute_component_scores(hidden_base_by_layer[layer_idx], hidden_vh)
            hidden_scores_b = compute_component_scores(hidden_ft_by_layer[layer_idx], hidden_vh)
            logit_scores_a = compute_component_scores(logit_base_by_layer[layer_idx], logit_vh)
            logit_scores_b = compute_component_scores(logit_ft_by_layer[layer_idx], logit_vh)
            _plot_svd_geometry_panel(
                axes[0, col],
                side_a_scores=hidden_scores_a,
                side_b_scores=hidden_scores_b,
                title=f"Hidden geometry, layer {layer_idx}",
                side_a_name=side_a_name,
                side_b_name=side_b_name,
            )
            _plot_svd_geometry_panel(
                axes[1, col],
                side_a_scores=logit_scores_a,
                side_b_scores=logit_scores_b,
                title=f"Logit geometry, layer {layer_idx}",
                side_a_name=side_a_name,
                side_b_name=side_b_name,
            )
        fig.tight_layout()
        fig.savefig(output_root / "svd_geometry_selected_layers.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

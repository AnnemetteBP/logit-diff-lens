from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer


def _ensure_tensor(x: Any, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(x)}")
    return x.detach().to(device="cpu", dtype=torch.float32)


def _coerce_dataset_rows(
    payload: Any,
    *,
    model_variant: str | None = None,
) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and "rows" in payload:
        rows = list(payload["rows"])
    elif isinstance(payload, list):
        rows = list(payload)
    else:
        raise ValueError("Expected a torch-loaded list of rows or a payload with a 'rows' key")

    coerced: List[Dict[str, Any]] = []
    for row in rows:
        if "hidden_states" in row and "logits" in row:
            coerced.append(
                {
                    "hidden_states": row["hidden_states"],
                    "logits": row["logits"],
                    "label": int(row["label"]),
                }
            )
            continue

        if model_variant is None:
            raise ValueError(
                "Rows with base/ft fields require model_variant='base' or model_variant='ft'"
            )
        suffix = "base" if model_variant == "base" else "ft"
        coerced.append(
            {
                "hidden_states": row[f"hidden_states_{suffix}"],
                "logits": row[f"logits_{suffix}"],
                "attention_outputs": row.get(f"attention_outputs_{suffix}"),
                "mlp_outputs": row.get(f"mlp_outputs_{suffix}"),
                "label": int(row.get("label", row.get("harmfulness_label_ft", 0))),
            }
        )
    return coerced


def load_collected_activation_dataset(
    path: str | Path,
    *,
    model_variant: str | None = None,
) -> List[Dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    return _coerce_dataset_rows(payload, model_variant=model_variant)


def load_paired_collected_activation_dataset(path: str | Path) -> List[Dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    return load_paired_collected_activation_dataset_from_payload(payload)


def _extract_last_token_hidden_by_layer(rows: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
    num_layers = len(rows[0]["hidden_states"])
    layer_arrays: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    for row_idx, row in enumerate(rows):
        hidden_states = row["hidden_states"]
        if len(hidden_states) != num_layers:
            raise ValueError(
                f"Hidden state layer count mismatch at row {row_idx}: {len(hidden_states)} vs {num_layers}"
            )
        for layer_idx, tensor in enumerate(hidden_states):
            hidden = _ensure_tensor(tensor, name=f"hidden_states[{row_idx}][{layer_idx}]")
            if hidden.ndim != 3:
                raise ValueError(
                    f"hidden_states[{row_idx}][{layer_idx}] must have shape [batch, seq, dim]"
                )
            layer_arrays[layer_idx].append(hidden[:, -1, :].reshape(-1).numpy())
    return [np.stack(layer_rows, axis=0).astype(np.float32, copy=False) for layer_rows in layer_arrays]


def _extract_last_token_hidden_by_layer_from_key(
    rows: Sequence[Dict[str, Any]],
    *,
    key: str,
    drop_embedding_if_present: bool = False,
    expected_num_block_layers: int | None = None,
) -> List[np.ndarray]:
    num_layers = len(rows[0][key])
    start_idx = 0
    if drop_embedding_if_present and expected_num_block_layers is not None and num_layers == expected_num_block_layers + 1:
        start_idx = 1
        num_layers = expected_num_block_layers
    layer_arrays: List[List[np.ndarray]] = [[] for _ in range(num_layers)]
    for row_idx, row in enumerate(rows):
        hidden_states = row[key]
        usable = hidden_states[start_idx:]
        if len(usable) != num_layers:
            raise ValueError(
                f"{key} layer count mismatch at row {row_idx}: {len(usable)} vs {num_layers}"
            )
        for layer_idx, tensor in enumerate(usable):
            hidden = _ensure_tensor(tensor, name=f"{key}[{row_idx}][{layer_idx}]")
            if hidden.ndim != 3:
                raise ValueError(f"{key}[{row_idx}][{layer_idx}] must have shape [batch, seq, dim]")
            layer_arrays[layer_idx].append(hidden[:, -1, :].reshape(-1).numpy())
    return [np.stack(layer_rows, axis=0).astype(np.float32, copy=False) for layer_rows in layer_arrays]


def _extract_last_token_component_diff_norms(
    rows: Sequence[Dict[str, Any]],
    *,
    base_key: str,
    ft_key: str,
) -> np.ndarray:
    num_layers = len(rows[0][base_key])
    values: List[List[float]] = [[] for _ in range(num_layers)]
    for row_idx, row in enumerate(rows):
        base_layers = row[base_key]
        ft_layers = row[ft_key]
        if len(base_layers) != num_layers or len(ft_layers) != num_layers:
            raise ValueError(f"{base_key}/{ft_key} layer count mismatch at row {row_idx}")
        for layer_idx, (base_tensor, ft_tensor) in enumerate(zip(base_layers, ft_layers)):
            base = _ensure_tensor(base_tensor, name=f"{base_key}[{row_idx}][{layer_idx}]")
            ft = _ensure_tensor(ft_tensor, name=f"{ft_key}[{row_idx}][{layer_idx}]")
            if base.ndim != 3 or ft.ndim != 3:
                raise ValueError(f"{base_key}/{ft_key} tensors must have shape [batch, seq, dim]")
            diff = ft[:, -1, :] - base[:, -1, :]
            values[layer_idx].append(float(torch.norm(diff, dim=-1).mean().item()))
    return np.asarray([np.mean(layer_values) for layer_values in values], dtype=float)


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


def _norm_1d(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("norm expects a 1D sequence")
    if array.size == 0:
        raise ValueError("norm expects a non-empty sequence")
    if np.max(array) - np.min(array) < 1e-12:
        return np.zeros_like(array)
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def _extract_logit_norms(rows: Sequence[Dict[str, Any]], *, num_layers: int) -> tuple[np.ndarray, str]:
    first_logits = rows[0]["logits"]
    if isinstance(first_logits, (list, tuple)):
        layer_norms: List[List[float]] = [[] for _ in range(len(first_logits))]
        for row_idx, row in enumerate(rows):
            logits_per_layer = row["logits"]
            if len(logits_per_layer) != len(layer_norms):
                raise ValueError(
                    f"Logit layer count mismatch at row {row_idx}: {len(logits_per_layer)} vs {len(layer_norms)}"
                )
            for layer_idx, tensor in enumerate(logits_per_layer):
                logits = _ensure_tensor(tensor, name=f"logits[{row_idx}][{layer_idx}]")
                if logits.ndim != 3:
                    raise ValueError(f"logits[{row_idx}][{layer_idx}] must have shape [batch, seq, vocab]")
                layer_norms[layer_idx].append(float(torch.norm(logits[:, -1, :], dim=-1).mean().item()))
        norms = np.asarray([np.mean(values) for values in layer_norms], dtype=float)
        if norms.shape[0] == num_layers + 1:
            norms = norms[-num_layers:]
        elif norms.shape[0] != num_layers:
            raise ValueError(
                f"Layerwise logits must have {num_layers} or {num_layers + 1} entries, got {norms.shape[0]}"
            )
        return norms, "layerwise"

    per_example = []
    for row_idx, row in enumerate(rows):
        logits = _ensure_tensor(row["logits"], name=f"logits[{row_idx}]")
        if logits.ndim != 3:
            raise ValueError(f"logits[{row_idx}] must have shape [batch, seq, vocab]")
        per_example.append(float(torch.norm(logits[:, -1, :], dim=-1).mean().item()))
    scalar = float(np.mean(np.asarray(per_example, dtype=float)))
    return np.repeat(scalar, num_layers).astype(float), "final_repeated"


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.5
    if np.allclose(scores, scores[0]):
        return 0.5
    return float(roc_auc_score(labels, scores))


def analyze_collected_activation_dataset(
    data_or_path: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any],
    *,
    output_dir: str | Path = "plots",
    model_variant: str | None = None,
    selected_layers: Sequence[int] | None = None,
) -> Dict[str, Any]:
    if isinstance(data_or_path, (str, Path)):
        rows = load_collected_activation_dataset(data_or_path, model_variant=model_variant)
    else:
        rows = _coerce_dataset_rows(data_or_path, model_variant=model_variant)

    if not rows:
        raise ValueError("Collected activation dataset is empty")

    num_examples = len(rows)
    num_layers = len(rows[0]["hidden_states"])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    h_by_layer = _extract_last_token_hidden_by_layer(rows)

    pca_results: List[Dict[str, Any]] = []
    cumulative_variance: List[np.ndarray] = []
    projection_scores: List[np.ndarray] = []
    auroc_per_layer: List[float] = []
    pca_top1: List[float] = []

    for layer_idx, h_layer in enumerate(h_by_layer):
        n_components = int(min(h_layer.shape[0], h_layer.shape[1]))
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(h_layer)
        explained = pca.explained_variance_ratio_.astype(np.float32, copy=False)
        cumulative = np.cumsum(explained)

        mean_vec = h_layer.mean(axis=0).astype(np.float32, copy=False)
        mean_norm = float(np.linalg.norm(mean_vec))
        if mean_norm > 0:
            mean_vec = mean_vec / mean_norm
        scores = (h_layer @ mean_vec).astype(np.float32, copy=False)
        auroc = _safe_auroc(labels, scores)

        pca_results.append(
            {
                "layer": layer_idx,
                "explained_variance_ratio": explained.tolist(),
            }
        )
        cumulative_variance.append(cumulative)
        projection_scores.append(scores)
        auroc_per_layer.append(auroc)
        pca_top1.append(float(explained[0]) if explained.size else 0.0)

    logit_norms, logit_norm_mode = _extract_logit_norms(rows, num_layers=num_layers)
    logit_norms_normalized = logit_norms / (float(np.max(logit_norms)) + 1e-8)
    auroc_array = np.asarray(auroc_per_layer, dtype=float)
    auroc_for_heatmap = np.nan_to_num(auroc_array, nan=0.5)
    pca_top1_array = np.asarray(pca_top1, dtype=float)
    heatmap_matrix = np.vstack(
        [
            logit_norms_normalized,
            auroc_for_heatmap,
            pca_top1_array,
        ]
    )

    if selected_layers is None:
        selected_layers = sorted(set([0, max(0, num_layers // 2), num_layers - 1]))

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    layers = np.arange(num_layers)

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

    return {
        "num_layers": num_layers,
        "num_examples": num_examples,
        "labels": labels.tolist(),
        "pca_results": pca_results,
        "cumulative_variance": [values.tolist() for values in cumulative_variance],
        "projection_scores": [values.tolist() for values in projection_scores],
        "auroc_per_layer": auroc_array.tolist(),
        "pca_top1_variance": pca_top1_array.tolist(),
        "logit_norms": logit_norms.tolist(),
        "logit_norm_mode": logit_norm_mode,
        "auroc_mode": "chance_filled" if len(np.unique(labels)) < 2 else "measured",
        "selected_layers": list(selected_layers),
        "plot_paths": {
            "logit_norm": str(output_root / "logit_norm.png"),
            "auroc": str(output_root / "auroc.png"),
            "pca": str(output_root / "pca.png"),
            "heatmap": str(output_root / "heatmap.png"),
            "pca_cumulative": str(output_root / "pca_cumulative.png"),
        },
    }


def analyze_paired_collected_activation_dataset(
    data_or_path: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any],
    *,
    output_dir: str | Path = "plots_paired",
    selected_layers: Sequence[int] | None = None,
) -> Dict[str, Any]:
    if isinstance(data_or_path, (str, Path)):
        rows = load_paired_collected_activation_dataset(data_or_path)
    else:
        if isinstance(data_or_path, dict) and "rows" in data_or_path:
            rows = load_paired_collected_activation_dataset_from_payload(data_or_path)
        else:
            rows = load_paired_collected_activation_dataset_from_payload({"rows": list(data_or_path)})

    if not rows:
        raise ValueError("Paired collected activation dataset is empty")

    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    num_examples = len(rows)
    num_block_layers = len(rows[0]["attention_outputs_base"])

    h_base_by_layer = _extract_last_token_hidden_by_layer_from_key(
        rows,
        key="hidden_states_base",
        drop_embedding_if_present=True,
        expected_num_block_layers=num_block_layers,
    )
    h_ft_by_layer = _extract_last_token_hidden_by_layer_from_key(
        rows,
        key="hidden_states_ft",
        drop_embedding_if_present=True,
        expected_num_block_layers=num_block_layers,
    )

    z_base = np.stack(
        [
            _ensure_tensor(row["logits_base"], name=f"logits_base[{idx}]")[:, -1, :].reshape(-1).numpy()
            for idx, row in enumerate(rows)
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    z_ft = np.stack(
        [
            _ensure_tensor(row["logits_ft"], name=f"logits_ft[{idx}]")[:, -1, :].reshape(-1).numpy()
            for idx, row in enumerate(rows)
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    delta_z = (z_ft - z_base).astype(np.float32, copy=False)
    per_example_logit = np.linalg.norm(delta_z, axis=1).astype(np.float32, copy=False)
    repeated_logit_diff = float(np.mean(per_example_logit))

    pca_top1: List[float] = []
    cumulative_variance: List[np.ndarray] = []
    latent_aurocs: List[float] = []
    logit_aurocs: List[float] = []
    delta_hidden_norms: List[float] = []
    pca_results: List[Dict[str, Any]] = []

    for layer_idx, (h_base, h_ft) in enumerate(zip(h_base_by_layer, h_ft_by_layer)):
        delta_h = (h_ft - h_base).astype(np.float32, copy=False)
        delta_hidden_norms.append(float(np.mean(np.linalg.norm(delta_h, axis=1))))

        n_components = int(min(delta_h.shape[0], delta_h.shape[1]))
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(delta_h)
        explained = pca.explained_variance_ratio_.astype(np.float32, copy=False)
        cumulative = np.cumsum(explained)
        pca_top1.append(float(explained[0]) if explained.size else 0.0)
        cumulative_variance.append(cumulative)
        pca_results.append(
            {
                "layer": layer_idx,
                "explained_variance_ratio": explained.tolist(),
            }
        )

        mean_delta = delta_h.mean(axis=0).astype(np.float32, copy=False)
        mean_delta_norm = float(np.linalg.norm(mean_delta))
        if mean_delta_norm > 0:
            mean_delta = mean_delta / mean_delta_norm
        base_norms = np.linalg.norm(h_base, axis=1, keepdims=True)
        safe_base = h_base / np.where(base_norms > 0.0, base_norms, 1.0)
        scores = (safe_base @ mean_delta).astype(np.float32, copy=False)
        latent_aurocs.append(_safe_auroc(labels, scores))
        logit_aurocs.append(_safe_auroc(labels, per_example_logit))

    logit_diff = np.repeat(repeated_logit_diff, num_block_layers).astype(float)
    logit_diff_norm = logit_diff / (float(np.max(logit_diff)) + 1e-8)
    latent_auroc_array = np.asarray(latent_aurocs, dtype=float)
    logit_auroc_array = np.asarray(logit_aurocs, dtype=float)
    pca_top1_array = np.asarray(pca_top1, dtype=float)
    attn_diff = _extract_last_token_component_diff_norms(
        rows,
        base_key="attention_outputs_base",
        ft_key="attention_outputs_ft",
    )
    mlp_diff = _extract_last_token_component_diff_norms(
        rows,
        base_key="mlp_outputs_base",
        ft_key="mlp_outputs_ft",
    )

    combined_matrix = np.vstack(
        [
            logit_diff_norm,
            latent_auroc_array,
            logit_auroc_array,
            pca_top1_array,
        ]
    )
    heatmap_matrix = _normalize_rows_for_heatmap(
        np.vstack(
            [
                logit_diff,
                latent_auroc_array,
                logit_auroc_array,
                pca_top1_array,
                attn_diff,
                mlp_diff,
            ]
        )
    )

    if selected_layers is None:
        selected_layers = sorted(set([0, max(0, num_block_layers // 2), num_block_layers - 1]))

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    layers = np.arange(num_block_layers)

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

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(layers, _normalize_rows_for_heatmap(combined_matrix)[0], linewidth=2, label="Delta logits")
    ax.plot(layers, _normalize_rows_for_heatmap(combined_matrix)[1], linewidth=2, label="Latent AUROC")
    ax.plot(layers, _normalize_rows_for_heatmap(combined_matrix)[2], linewidth=2, label="Logit AUROC")
    ax.plot(layers, _normalize_rows_for_heatmap(combined_matrix)[3], linewidth=2, label="PCA top1")
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

    return {
        "num_examples": num_examples,
        "num_layers": num_block_layers,
        "labels": labels.tolist(),
        "delta_hidden_norms": delta_hidden_norms,
        "logit_diff": logit_diff.tolist(),
        "per_example_logit": per_example_logit.tolist(),
        "pca_results": pca_results,
        "pca_top1": pca_top1_array.tolist(),
        "cumulative_variance": [values.tolist() for values in cumulative_variance],
        "latent_auroc": latent_auroc_array.tolist(),
        "logit_auroc": logit_auroc_array.tolist(),
        "attn_diff": attn_diff.tolist(),
        "mlp_diff": mlp_diff.tolist(),
        "selected_layers": list(selected_layers),
        "plot_paths": {
            "delta_logits": str(output_root / "delta_logits.png"),
            "latent_auroc": str(output_root / "latent_auroc.png"),
            "logit_auroc": str(output_root / "logit_auroc.png"),
            "pca_top1": str(output_root / "pca_top1.png"),
            "combined": str(output_root / "combined.png"),
            "heatmap": str(output_root / "heatmap.png"),
            "prism": str(output_root / "prism.png"),
        },
    }


def analyze_full_dataset_with_prism(
    data_or_path: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any],
    *,
    output_dir: str | Path = "plots_full_dataset_with_prism",
) -> Dict[str, Any]:
    """Run the downstream delta/PCA/AUROC/prism analysis on a paired saved dataset.

    Notes:
    - This is downstream-only analysis. It never reruns the model.
    - Hidden states are block-aligned using `hidden_states_*[L + 1]`, because
      collected hidden states include the embedding output while attention/MLP
      outputs are block-indexed.
    """
    if isinstance(data_or_path, (str, Path)):
        payload = torch.load(Path(data_or_path), map_location="cpu")
        data = load_paired_collected_activation_dataset_from_payload(payload)
        input_path = str(Path(data_or_path))
    else:
        payload = data_or_path
        data = load_paired_collected_activation_dataset_from_payload(data_or_path)
        input_path = None

    if not data:
        raise ValueError("Paired dataset is empty")

    lm_head_weight_base, lm_head_weight_ft = _validate_required_prism_fields(payload, data)

    labels = np.asarray([int(ex["label"]) for ex in data], dtype=int)
    num_layers = len(data[0]["hidden_states_base"]) - 1
    if num_layers < 1:
        raise ValueError("Expected at least embedding + one block in hidden_states_base")

    logit_diff: List[float] = []
    hidden_diff: List[float] = []
    auroc_values: List[float] = []
    logit_auroc: List[float] = []
    pca_top1: List[float] = []
    attn_diff: List[float] = []
    mlp_diff: List[float] = []
    emb_contrib: List[float] = []

    final_logits_base: List[np.ndarray] = []
    final_logits_ft: List[np.ndarray] = []
    for ex_idx, ex in enumerate(data):
        final_logits_base.append(
            _ensure_tensor(ex["logits_base"], name=f"logits_base[{ex_idx}]")[:, -1, :].numpy()
        )
        final_logits_ft.append(
            _ensure_tensor(ex["logits_ft"], name=f"logits_ft[{ex_idx}]")[:, -1, :].numpy()
        )
    final_logits_base_array = np.concatenate(final_logits_base, axis=0).astype(np.float32, copy=False)
    final_logits_ft_array = np.concatenate(final_logits_ft, axis=0).astype(np.float32, copy=False)
    delta_logits_final = (final_logits_ft_array - final_logits_base_array).astype(np.float32, copy=False)
    final_logit_diff = float(np.mean(np.linalg.norm(delta_logits_final, axis=-1)))

    for layer_idx in range(num_layers):
        h_base: List[np.ndarray] = []
        h_ft: List[np.ndarray] = []
        z_base: List[np.ndarray] = []
        z_ft: List[np.ndarray] = []
        attn_base: List[np.ndarray] = []
        attn_ft: List[np.ndarray] = []
        mlp_base: List[np.ndarray] = []
        mlp_ft: List[np.ndarray] = []
        embedding_base: List[np.ndarray] = []
        embedding_ft: List[np.ndarray] = []

        for ex_idx, ex in enumerate(data):
            hb = _ensure_tensor(
                ex["hidden_states_base"][layer_idx + 1],
                name=f"hidden_states_base[{ex_idx}][{layer_idx + 1}]",
            )[:, -1, :].numpy()
            hf = _ensure_tensor(
                ex["hidden_states_ft"][layer_idx + 1],
                name=f"hidden_states_ft[{ex_idx}][{layer_idx + 1}]",
            )[:, -1, :].numpy()

            zb = hb @ lm_head_weight_base.T
            zf = hf @ lm_head_weight_ft.T

            ab = _ensure_tensor(
                ex["attention_outputs_base"][layer_idx],
                name=f"attention_outputs_base[{ex_idx}][{layer_idx}]",
            )[:, -1, :].numpy()
            af = _ensure_tensor(
                ex["attention_outputs_ft"][layer_idx],
                name=f"attention_outputs_ft[{ex_idx}][{layer_idx}]",
            )[:, -1, :].numpy()

            mb = _ensure_tensor(
                ex["mlp_outputs_base"][layer_idx],
                name=f"mlp_outputs_base[{ex_idx}][{layer_idx}]",
            )[:, -1, :].numpy()
            mf = _ensure_tensor(
                ex["mlp_outputs_ft"][layer_idx],
                name=f"mlp_outputs_ft[{ex_idx}][{layer_idx}]",
            )[:, -1, :].numpy()
            eb = _ensure_tensor(
                ex["embedding_base"],
                name=f"embedding_base[{ex_idx}]",
            )[:, -1, :].numpy()
            ef = _ensure_tensor(
                ex["embedding_ft"],
                name=f"embedding_ft[{ex_idx}]",
            )[:, -1, :].numpy()

            h_base.append(hb)
            h_ft.append(hf)
            z_base.append(zb)
            z_ft.append(zf)
            attn_base.append(ab)
            attn_ft.append(af)
            mlp_base.append(mb)
            mlp_ft.append(mf)
            embedding_base.append(eb)
            embedding_ft.append(ef)

        h_base_array = np.concatenate(h_base, axis=0).astype(np.float32, copy=False)
        h_ft_array = np.concatenate(h_ft, axis=0).astype(np.float32, copy=False)
        z_base_array = np.concatenate(z_base, axis=0).astype(np.float32, copy=False)
        z_ft_array = np.concatenate(z_ft, axis=0).astype(np.float32, copy=False)
        attn_base_array = np.concatenate(attn_base, axis=0).astype(np.float32, copy=False)
        attn_ft_array = np.concatenate(attn_ft, axis=0).astype(np.float32, copy=False)
        mlp_base_array = np.concatenate(mlp_base, axis=0).astype(np.float32, copy=False)
        mlp_ft_array = np.concatenate(mlp_ft, axis=0).astype(np.float32, copy=False)
        embedding_base_array = np.concatenate(embedding_base, axis=0).astype(np.float32, copy=False)
        embedding_ft_array = np.concatenate(embedding_ft, axis=0).astype(np.float32, copy=False)

        delta_h = (h_ft_array - h_base_array).astype(np.float32, copy=False)
        delta_z = (z_ft_array - z_base_array).astype(np.float32, copy=False)
        delta_attn = (attn_ft_array - attn_base_array).astype(np.float32, copy=False)
        delta_mlp = (mlp_ft_array - mlp_base_array).astype(np.float32, copy=False)
        delta_emb = (embedding_ft_array - embedding_base_array).astype(np.float32, copy=False)

        logit_attn_base = attn_base_array @ lm_head_weight_base.T
        logit_attn_ft = attn_ft_array @ lm_head_weight_ft.T
        logit_attn = (logit_attn_ft - logit_attn_base).astype(np.float32, copy=False)
        logit_mlp_base = mlp_base_array @ lm_head_weight_base.T
        logit_mlp_ft = mlp_ft_array @ lm_head_weight_ft.T
        logit_mlp = (logit_mlp_ft - logit_mlp_base).astype(np.float32, copy=False)
        logit_emb_base = embedding_base_array @ lm_head_weight_base.T
        logit_emb_ft = embedding_ft_array @ lm_head_weight_ft.T
        logit_emb = (logit_emb_ft - logit_emb_base).astype(np.float32, copy=False)

        hidden_diff.append(float(np.mean(np.linalg.norm(delta_h, axis=-1))))
        logit_diff.append(float(np.mean(np.linalg.norm(delta_z, axis=-1))))

        if np.var(delta_h) > 1e-12:
            pca = PCA()
            pca.fit(delta_h)
            pca_top1.append(float(pca.explained_variance_ratio_[0]))
        else:
            pca_top1.append(0.0)

        mean_delta = np.mean(delta_h, axis=0).astype(np.float32, copy=False)
        if np.linalg.norm(mean_delta) > 1e-12:
            mean_delta = mean_delta / np.linalg.norm(mean_delta)
            scores = (
                h_base_array / (np.linalg.norm(h_base_array, axis=1, keepdims=True) + 1e-12)
            ) @ mean_delta
        else:
            scores = np.zeros(len(labels), dtype=np.float32)

        if len(np.unique(labels)) > 1:
            auroc_values.append(float(roc_auc_score(labels, scores)))
        else:
            auroc_values.append(0.5)

        logit_scores = np.linalg.norm(delta_z, axis=-1)
        if len(np.unique(labels)) > 1:
            logit_auroc.append(float(roc_auc_score(labels, logit_scores)))
        else:
            logit_auroc.append(0.5)

        attn_diff.append(float(np.mean(np.linalg.norm(logit_attn, axis=-1))))
        mlp_diff.append(float(np.mean(np.linalg.norm(logit_mlp, axis=-1))))
        emb_contrib.append(float(np.mean(np.linalg.norm(logit_emb, axis=-1))))

    total_diff = logit_diff
    attn_contrib = attn_diff
    mlp_contrib = mlp_diff
    emb_contrib_plot = emb_contrib

    layers = np.arange(len(total_diff))

    def norm(x):
        x = np.array(x)
        if np.max(x) - np.min(x) < 1e-12:
            return np.zeros_like(x)
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    # ---- Plot 1: Δ logits ----

    plt.figure()
    plt.plot(layers, total_diff)
    plt.xlabel("Layer")
    plt.ylabel("||Δ logits||")
    plt.title("Δ logits (model difference)")

    # ---- Plot 2: Attention vs MLP ----

    plt.figure()
    plt.plot(layers, attn_contrib, label="attention")
    plt.plot(layers, mlp_contrib, label="MLP")
    plt.xlabel("Layer")
    plt.ylabel("Contribution magnitude")
    plt.legend()
    plt.title("Component contributions")

    # ---- Plot 3: Contribution ratio ----

    ratio_attn = np.array(attn_contrib) / (np.array(total_diff) + 1e-8)
    ratio_mlp = np.array(mlp_contrib) / (np.array(total_diff) + 1e-8)

    plt.figure()
    plt.plot(layers, ratio_attn, label="attention ratio")
    plt.plot(layers, ratio_mlp, label="MLP ratio")
    plt.xlabel("Layer")
    plt.ylabel("Ratio")
    plt.legend()
    plt.title("Relative importance")

    # ---- Plot 4: Prism alignment ----

    plt.figure()
    plt.plot(layers, total_diff, label="Δ logits")
    plt.plot(layers, np.array(attn_contrib) + np.array(mlp_contrib), label="attn+MLP")
    plt.xlabel("Layer")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title("Prism alignment")

    # ---- Plot 5: Heatmap ----

    matrix = np.array([
        norm(total_diff),
        norm(attn_contrib),
        norm(mlp_contrib),
        norm(ratio_attn),
        norm(ratio_mlp)
    ])

    plt.figure()
    plt.imshow(matrix, aspect='auto')
    plt.colorbar()
    plt.yticks(
        [0, 1, 2, 3, 4],
        ["Δ logits", "attention", "MLP", "attn ratio", "MLP ratio"]
    )
    plt.xlabel("Layer")
    plt.title("Layer-wise mechanism")

    # === PRISM PLOTTING BLOCK START ===

    layers = np.arange(len(total_diff))

    def norm(x):
        x = np.array(x)
        if np.max(x) - np.min(x) < 1e-12:
            return np.zeros_like(x)
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    # Plot 1 — Δ logits

    plt.figure()
    plt.plot(layers, total_diff)
    plt.title("Δ logits per layer")
    plt.xlabel("Layer")
    plt.ylabel("||Δ logits||")

    # Plot 2 — Attention vs MLP

    plt.figure()
    plt.plot(layers, emb_contrib_plot, label="embedding")
    plt.plot(layers, attn_contrib, label="attention")
    plt.plot(layers, mlp_contrib, label="MLP")
    plt.legend()
    plt.title("Full prism contributions")

    # Plot 3 — Contribution ratios

    ratio_emb = np.array(emb_contrib_plot) / (np.array(total_diff) + 1e-8)
    ratio_attn = np.array(attn_contrib) / (np.array(total_diff) + 1e-8)
    ratio_mlp = np.array(mlp_contrib) / (np.array(total_diff) + 1e-8)

    plt.figure()
    plt.plot(layers, ratio_emb, label="embedding ratio")
    plt.plot(layers, ratio_attn, label="attention ratio")
    plt.plot(layers, ratio_mlp, label="MLP ratio")
    plt.legend()
    plt.title("Contribution ratios")

    # Plot 4 — Prism alignment

    plt.figure()
    plt.plot(layers, total_diff, label="Δ logits")
    plt.plot(
        layers,
        np.array(emb_contrib_plot) + np.array(attn_contrib) + np.array(mlp_contrib),
        label="emb+attn+MLP",
    )
    plt.legend()
    plt.title("Prism alignment")

    # Plot 5 — Heatmap

    matrix = np.array([
        norm(total_diff),
        norm(emb_contrib_plot),
        norm(attn_contrib),
        norm(mlp_contrib),
        norm(ratio_emb),
        norm(ratio_attn),
        norm(ratio_mlp)
    ])

    plt.figure()
    plt.imshow(matrix, aspect='auto')
    plt.colorbar()
    plt.yticks(
        [0, 1, 2, 3, 4, 5, 6],
        ["Δ logits", "embedding", "attention", "MLP", "emb ratio", "attn ratio", "MLP ratio"]
    )
    plt.title("Layer-wise prism analysis")

    # === PRISM PLOTTING BLOCK END ===

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    layers = np.arange(num_layers)

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
    plt.title("latent -> harm")
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "latent_to_harm.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(layers, logit_auroc)
    plt.title("logit -> harm")
    plt.xlabel("Layer")
    plt.ylabel("AUROC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "logit_to_harm.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(layers, pca_top1)
    plt.title("PCA")
    plt.xlabel("Layer")
    plt.ylabel("Top-1 explained variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "pca_top1.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(layers, _norm_1d(logit_diff), label="Delta logits")
    plt.plot(layers, _norm_1d(auroc_values), label="latent->harm")
    plt.plot(layers, _norm_1d(logit_auroc), label="logit->harm")
    plt.plot(layers, _norm_1d(pca_top1), label="PCA")
    plt.legend()
    plt.title("alignment")
    plt.xlabel("Layer")
    plt.ylabel("Normalized value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "alignment.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(layers, attn_diff, label="attention")
    plt.plot(layers, mlp_diff, label="MLP")
    plt.legend()
    plt.title("Prism contributions")
    plt.xlabel("Layer")
    plt.ylabel("Mean component difference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "prism_contributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(layers, logit_diff, label="Delta logits")
    plt.plot(layers, np.asarray(attn_diff) + np.asarray(mlp_diff), label="attention + MLP")
    plt.legend()
    plt.title("Prism alignment")
    plt.xlabel("Layer")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "prism_alignment.png", dpi=150, bbox_inches="tight")
    plt.close()

    prism_heatmap_matrix = np.array(
        [
            _norm_1d(hidden_diff),
            _norm_1d(logit_diff),
            _norm_1d(attn_diff),
            _norm_1d(mlp_diff),
        ]
    )

    plt.figure(figsize=(10, 4.5))
    plt.imshow(prism_heatmap_matrix, aspect="auto")
    plt.colorbar()
    plt.yticks(
        [0, 1, 2, 3],
        ["Delta hidden", "Delta logits", "attention", "MLP"],
    )
    plt.xlabel("Layer")
    plt.title("Prism heatmap")
    plt.tight_layout()
    plt.savefig(output_root / "prism_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    matrix = np.array(
        [
            _norm_1d(logit_diff),
            _norm_1d(auroc_values),
            _norm_1d(logit_auroc),
            _norm_1d(pca_top1),
            _norm_1d(attn_diff),
            _norm_1d(mlp_diff),
        ]
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar()
    plt.yticks(
        [0, 1, 2, 3, 4, 5],
        ["Delta logits", "latent->harm", "logit->harm", "PCA", "attention", "MLP"],
    )
    plt.xlabel("Layer")
    plt.title("layer analysis")
    plt.tight_layout()
    plt.savefig(output_root / "layer_analysis_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(layers, emb_contrib, label="embedding")
    plt.plot(layers, attn_diff, label="attention")
    plt.plot(layers, mlp_diff, label="MLP")
    plt.legend()
    plt.title("mechanism")
    plt.xlabel("Layer")
    plt.ylabel("Mean component difference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_root / "mechanism.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "input_path": input_path,
        "num_examples": len(data),
        "num_layers": num_layers,
        "labels": labels.tolist(),
        "hidden_diff": list(map(float, hidden_diff)),
        "logit_diff": list(map(float, logit_diff)),
        "final_logit_diff": float(final_logit_diff),
        "AUROC": list(map(float, auroc_values)),
        "logit_AUROC": list(map(float, logit_auroc)),
        "pca_top1": list(map(float, pca_top1)),
        "emb_contrib": list(map(float, emb_contrib)),
        "attn_diff": list(map(float, attn_diff)),
        "mlp_diff": list(map(float, mlp_diff)),
        "plots": {
            "delta_hidden": str(output_root / "delta_hidden.png"),
            "delta_logits": str(output_root / "delta_logits.png"),
            "final_logits_difference": str(output_root / "final_logits_difference.png"),
            "latent_to_harm": str(output_root / "latent_to_harm.png"),
            "logit_to_harm": str(output_root / "logit_to_harm.png"),
            "pca_top1": str(output_root / "pca_top1.png"),
            "alignment": str(output_root / "alignment.png"),
            "layer_analysis_heatmap": str(output_root / "layer_analysis_heatmap.png"),
            "prism_heatmap": str(output_root / "prism_heatmap.png"),
            "prism_alignment": str(output_root / "prism_alignment.png"),
            "prism_contributions": str(output_root / "prism_contributions.png"),
            "mechanism": str(output_root / "mechanism.png"),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def plot_group_logit_lens_token_divergence(
    data_or_path: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any],
    *,
    group_id: str,
    output_path: str | Path,
    layer_start_fraction: float = 0.5,
    metric: str = "kl_base_ft",
) -> Dict[str, Any]:
    if isinstance(data_or_path, (str, Path)):
        payload = torch.load(Path(data_or_path), map_location="cpu")
        input_path = str(Path(data_or_path))
    else:
        payload = data_or_path
        input_path = None

    data = load_paired_collected_activation_dataset_from_payload(payload)
    lm_head_weight_base, lm_head_weight_ft = _validate_required_prism_fields(payload, data)

    rows = [row for row in payload["rows"] if str(row.get("group_id")) == group_id]
    if len(rows) != 3:
        raise ValueError(f"Expected exactly 3 rows for group_id={group_id}, found {len(rows)}")

    rows = sorted(rows, key=lambda row: {"input": 0, "base": 1, "finetuned": 2}.get(str(row.get("model_role")), 99))
    num_layers = len(rows[0]["hidden_states_base"]) - 1
    start_layer = max(0, min(num_layers - 1, int(np.floor(layer_start_fraction * num_layers))))
    layer_indices = list(range(start_layer, num_layers))
    if not layer_indices:
        raise ValueError("No layers selected for token divergence plot")

    tokenizer_path = str(rows[0].get("base_model_path") or rows[0].get("model_path") or "")
    if not tokenizer_path:
        raise ValueError("Could not resolve tokenizer path from saved rows")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if metric not in {"kl_base_ft", "js", "tvd"}:
        raise ValueError("metric must be one of: 'kl_base_ft', 'js', 'tvd'")

    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-12)

    def _probabilities(base_logits: np.ndarray, ft_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p_base = _softmax(base_logits)
        p_ft = _softmax(ft_logits)
        return p_base, p_ft

    def _kl_divergence(base_logits: np.ndarray, ft_logits: np.ndarray) -> np.ndarray:
        p_base, p_ft = _probabilities(base_logits, ft_logits)
        return np.sum(p_base * (np.log(p_base + 1e-12) - np.log(p_ft + 1e-12)), axis=-1)

    def _js_divergence(base_logits: np.ndarray, ft_logits: np.ndarray) -> np.ndarray:
        p_base, p_ft = _probabilities(base_logits, ft_logits)
        mean_dist = 0.5 * (p_base + p_ft)
        kl_base_mean = np.sum(p_base * (np.log(p_base + 1e-12) - np.log(mean_dist + 1e-12)), axis=-1)
        kl_ft_mean = np.sum(p_ft * (np.log(p_ft + 1e-12) - np.log(mean_dist + 1e-12)), axis=-1)
        return 0.5 * (kl_base_mean + kl_ft_mean)

    def _tvd(base_logits: np.ndarray, ft_logits: np.ndarray) -> np.ndarray:
        p_base, p_ft = _probabilities(base_logits, ft_logits)
        return 0.5 * np.sum(np.abs(p_base - p_ft), axis=-1)

    metric_fns = {
        "kl_base_ft": _kl_divergence,
        "js": _js_divergence,
        "tvd": _tvd,
    }
    metric_labels = {
        "kl_base_ft": "KL(base || finetuned)",
        "js": "JS divergence",
        "tvd": "Total variation distance",
    }

    role_labels = {
        "input": "User Input",
        "base": "Base Response",
        "finetuned": "Finetuned Response",
    }

    per_row: List[Dict[str, Any]] = []
    global_max = 0.0
    for row in rows:
        input_ids = _ensure_tensor(row["input_ids"], name=f"input_ids[{row['id']}]").to(dtype=torch.long).numpy()[0]
        if input_ids.shape[0] < 2:
            raise ValueError(f"Row {row['id']} must have at least 2 tokens")

        position_heatmap: List[np.ndarray] = []
        position_heatmap_kl: List[np.ndarray] = []
        position_heatmap_js: List[np.ndarray] = []
        position_heatmap_tvd: List[np.ndarray] = []
        for layer_idx in layer_indices:
            h_base = _ensure_tensor(
                row["hidden_states_base"][layer_idx + 1],
                name=f"hidden_states_base[{row['id']}][{layer_idx + 1}]",
            )[0, :-1, :].numpy()
            h_ft = _ensure_tensor(
                row["hidden_states_ft"][layer_idx + 1],
                name=f"hidden_states_ft[{row['id']}][{layer_idx + 1}]",
            )[0, :-1, :].numpy()
            logits_base_layer = h_base @ lm_head_weight_base.T
            logits_ft_layer = h_ft @ lm_head_weight_ft.T
            position_heatmap.append(metric_fns[metric](logits_base_layer, logits_ft_layer).astype(np.float32, copy=False))
            position_heatmap_kl.append(_kl_divergence(logits_base_layer, logits_ft_layer).astype(np.float32, copy=False))
            position_heatmap_js.append(_js_divergence(logits_base_layer, logits_ft_layer).astype(np.float32, copy=False))
            position_heatmap_tvd.append(_tvd(logits_base_layer, logits_ft_layer).astype(np.float32, copy=False))

        heatmap = np.stack(position_heatmap, axis=0).astype(np.float32, copy=False)
        aggregate_position_score = heatmap.mean(axis=0)
        aggregate_position_score_kl = np.stack(position_heatmap_kl, axis=0).mean(axis=0).astype(np.float32, copy=False)
        aggregate_position_score_js = np.stack(position_heatmap_js, axis=0).mean(axis=0).astype(np.float32, copy=False)
        aggregate_position_score_tvd = np.stack(position_heatmap_tvd, axis=0).mean(axis=0).astype(np.float32, copy=False)
        final_logits_base = _ensure_tensor(row["logits_base"], name=f"logits_base[{row['id']}]")[0, :-1, :].numpy()
        final_logits_ft = _ensure_tensor(row["logits_ft"], name=f"logits_ft[{row['id']}]")[0, :-1, :].numpy()
        targets = input_ids[1:]
        probs_base = _softmax(final_logits_base)[np.arange(targets.shape[0]), targets]
        probs_ft = _softmax(final_logits_ft)[np.arange(targets.shape[0]), targets]
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[:-1].tolist())
        target_tokens = tokenizer.convert_ids_to_tokens(targets.tolist())
        global_max = max(global_max, float(np.max(heatmap)), float(np.max(aggregate_position_score)))
        per_row.append(
            {
                "row_id": int(row["id"]),
                "model_role": str(row.get("model_role")),
                "title": role_labels.get(str(row.get("model_role")), str(row.get("model_role"))),
                "heatmap": heatmap,
                "aggregate_position_score": aggregate_position_score,
                "aggregate_position_score_kl": aggregate_position_score_kl,
                "aggregate_position_score_js": aggregate_position_score_js,
                "aggregate_position_score_tvd": aggregate_position_score_tvd,
                "input_tokens": input_tokens,
                "target_tokens": target_tokens,
                "probs_base": probs_base,
                "probs_ft": probs_ft,
            }
        )

    if global_max <= 0:
        global_max = 1.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(max(14, max(len(row["input_tokens"]) for row in per_row) * 0.5), 14))
    outer = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 1, 1], hspace=0.5)
    heatmap_mappable = None

    for row_idx, row_data in enumerate(per_row):
        inner = outer[row_idx].subgridspec(nrows=3, ncols=1, height_ratios=[0.18, 1.0, 0.4], hspace=0.08)
        agg_ax = fig.add_subplot(inner[0])
        heat_ax = fig.add_subplot(inner[1])
        prob_ax = fig.add_subplot(inner[2])

        agg_img = agg_ax.imshow(
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
        fig.colorbar(heatmap_mappable, ax=fig.axes, shrink=0.98, pad=0.01, label=metric_labels[metric])
    fig.suptitle(f"Token-aligned late-layer {metric_labels[metric]} and target-token probabilities", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.97, 0.985])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_path": input_path,
        "group_id": group_id,
        "layer_start_fraction": float(layer_start_fraction),
        "layer_indices": layer_indices,
        "metric": metric,
        "output_path": str(output_path),
        "rows": [
            {
                "row_id": row_data["row_id"],
                "model_role": row_data["model_role"],
                "aggregate_position_score": row_data["aggregate_position_score"].tolist(),
                "mean_aggregate_score": float(np.mean(row_data["aggregate_position_score"])),
                "aggregate_position_score_kl": row_data["aggregate_position_score_kl"].tolist(),
                "mean_aggregate_score_kl": float(np.mean(row_data["aggregate_position_score_kl"])),
                "aggregate_position_score_js": row_data["aggregate_position_score_js"].tolist(),
                "mean_aggregate_score_js": float(np.mean(row_data["aggregate_position_score_js"])),
                "aggregate_position_score_tvd": row_data["aggregate_position_score_tvd"].tolist(),
                "mean_aggregate_score_tvd": float(np.mean(row_data["aggregate_position_score_tvd"])),
            }
            for row_data in per_row
        ],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
def load_paired_collected_activation_dataset_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and "rows" in payload:
        rows = list(payload["rows"])
    elif isinstance(payload, list):
        rows = list(payload)
    else:
        raise ValueError("Expected a torch-loaded list of rows or a payload with a 'rows' key")
    return load_paired_collected_activation_dataset_rows(rows)


def load_paired_collected_activation_dataset_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    coerced: List[Dict[str, Any]] = []
    for row in rows:
        required = (
            "hidden_states_base",
            "hidden_states_ft",
            "logits_base",
            "logits_ft",
            "attention_outputs_base",
            "attention_outputs_ft",
            "mlp_outputs_base",
            "mlp_outputs_ft",
        )
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Paired activation row is missing keys: {missing}")
        coerced.append(
            {
                "embedding_base": row["embedding_base"],
                "embedding_ft": row["embedding_ft"],
                "hidden_states_base": row["hidden_states_base"],
                "hidden_states_ft": row["hidden_states_ft"],
                "logits_base": row["logits_base"],
                "logits_ft": row["logits_ft"],
                "attention_outputs_base": row["attention_outputs_base"],
                "attention_outputs_ft": row["attention_outputs_ft"],
                "mlp_outputs_base": row["mlp_outputs_base"],
                "mlp_outputs_ft": row["mlp_outputs_ft"],
                "label": int(row.get("label", row.get("harmfulness_label_ft", 0))),
            }
        )
    return coerced


def _validate_required_prism_fields(payload: Any, data: Sequence[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not data:
        raise ValueError("Dataset is empty")
    required_keys = (
        "hidden_states_base",
        "hidden_states_ft",
        "logits_base",
        "logits_ft",
        "attention_outputs_base",
        "attention_outputs_ft",
        "mlp_outputs_base",
        "mlp_outputs_ft",
        "embedding_base",
        "embedding_ft",
        "label",
    )
    sample = data[0]
    missing = [key for key in required_keys if key not in sample]
    if missing:
        raise ValueError(
            "Missing required fields for prism analysis: "
            + ", ".join(missing)
            + ". Modify the collection pipeline and recollect before analysis."
        )

    if not isinstance(payload, dict) or "lm_head_weight_base" not in payload or "lm_head_weight_ft" not in payload:
        raise ValueError(
            "Missing lm_head_weight_base / lm_head_weight_ft in saved payload. "
            "Modify the collection pipeline and recollect before prism-difference analysis."
        )
    lm_head_base = _ensure_tensor(payload["lm_head_weight_base"], name="lm_head_weight_base").numpy()
    lm_head_ft = _ensure_tensor(payload["lm_head_weight_ft"], name="lm_head_weight_ft").numpy()
    return lm_head_base, lm_head_ft

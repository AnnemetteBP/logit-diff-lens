from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from ..tokenizer_loading import load_tokenizer

from ..plotting.support.activation_dataset_plotting import (
    save_collected_activation_dataset_plots,
    save_full_prism_analysis_plots,
    save_group_token_divergence_plot,
    save_paired_activation_svd_plots,
    save_paired_collected_activation_dataset_plots,
)


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


def align_model_activation_datasets(
    base_data_or_path: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any],
    finetuned_data_or_path: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any],
) -> Dict[str, Any]:
    def _load_payload_and_rows(source: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any]) -> tuple[Any, List[Dict[str, Any]]]:
        if isinstance(source, (str, Path)):
            payload = torch.load(Path(source), map_location="cpu")
        else:
            payload = source
        if isinstance(payload, dict) and "rows" in payload:
            rows = list(payload["rows"])
        elif isinstance(payload, list):
            rows = list(payload)
        else:
            raise ValueError("Expected a torch-loaded list of rows or a payload with a 'rows' key")
        return payload, rows

    base_payload, base_rows = _load_payload_and_rows(base_data_or_path)
    ft_payload, ft_rows = _load_payload_and_rows(finetuned_data_or_path)
    if len(base_rows) != len(ft_rows):
        raise ValueError(f"Row count mismatch between base and finetuned payloads: {len(base_rows)} vs {len(ft_rows)}")

    ft_by_id = {int(row["id"]): row for row in ft_rows}
    paired_rows: List[Dict[str, Any]] = []
    for base_row in base_rows:
        row_id = int(base_row["id"])
        if row_id not in ft_by_id:
            raise ValueError(f"Missing matching finetuned row for id={row_id}")
        ft_row = ft_by_id[row_id]
        shared_checks = ("group_id", "source_kind", "model_role", "continuation_kind", "collection_text")
        for key in shared_checks:
            if base_row.get(key) != ft_row.get(key):
                raise ValueError(
                    f"Mismatch for id={row_id} on '{key}': {base_row.get(key)!r} vs {ft_row.get(key)!r}"
                )
        paired_rows.append(
            {
                **base_row,
                "embedding_base": base_row["embedding"],
                "embedding_ft": ft_row["embedding"],
                "hidden_states_base": base_row["hidden_states"],
                "hidden_states_ft": ft_row["hidden_states"],
                "logits_base": base_row["logits"],
                "logits_ft": ft_row["logits"],
                "attention_outputs_base": base_row["attention_outputs"],
                "attention_outputs_ft": ft_row["attention_outputs"],
                "mlp_outputs_base": base_row["mlp_outputs"],
                "mlp_outputs_ft": ft_row["mlp_outputs"],
                "layer_records_base": base_row.get("layer_records"),
                "layer_records_ft": ft_row.get("layer_records"),
                "label": int(base_row.get("label", base_row.get("harmfulness_label_ft", 0))),
                "base_model_key": base_payload.get("model_key") if isinstance(base_payload, dict) else None,
                "finetuned_model_key": ft_payload.get("model_key") if isinstance(ft_payload, dict) else None,
            }
        )

    return {
        "dataset_path_base": str(base_payload.get("dataset_path")) if isinstance(base_payload, dict) and base_payload.get("dataset_path") is not None else None,
        "dataset_path_finetuned": str(ft_payload.get("dataset_path")) if isinstance(ft_payload, dict) and ft_payload.get("dataset_path") is not None else None,
        "lm_head_weight_base": base_payload.get("lm_head_weight") if isinstance(base_payload, dict) else None,
        "lm_head_weight_ft": ft_payload.get("lm_head_weight") if isinstance(ft_payload, dict) else None,
        "rows": paired_rows,
    }


def align_model_activation_datasets_by_group(
    side_a_data_or_path: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any],
    side_b_data_or_path: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any],
) -> Dict[str, Any]:
    def _load_payload_and_rows(source: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any]) -> tuple[Any, List[Dict[str, Any]]]:
        if isinstance(source, (str, Path)):
            payload = torch.load(Path(source), map_location="cpu")
        else:
            payload = source
        if isinstance(payload, dict) and "rows" in payload:
            rows = list(payload["rows"])
        elif isinstance(payload, list):
            rows = list(payload)
        else:
            raise ValueError("Expected a torch-loaded list of rows or a payload with a 'rows' key")
        return payload, rows

    side_a_payload, side_a_rows = _load_payload_and_rows(side_a_data_or_path)
    side_b_payload, side_b_rows = _load_payload_and_rows(side_b_data_or_path)
    if len(side_a_rows) != len(side_b_rows):
        raise ValueError(f"Row count mismatch between side A and side B payloads: {len(side_a_rows)} vs {len(side_b_rows)}")

    def _row_key(row: Dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("group_id")),
            str(row.get("variant")),
            str(row.get("source_kind")),
            str(row.get("model_role")),
        )

    side_b_by_key = {_row_key(row): row for row in side_b_rows}
    paired_rows: List[Dict[str, Any]] = []
    for side_a_row in side_a_rows:
        key = _row_key(side_a_row)
        if key not in side_b_by_key:
            raise ValueError(f"Missing matching side-B row for key={key}")
        side_b_row = side_b_by_key[key]
        shared_checks = ("group_id", "variant", "source_kind", "model_role", "continuation_kind")
        for check_key in shared_checks:
            if side_a_row.get(check_key) != side_b_row.get(check_key):
                raise ValueError(
                    f"Mismatch for key={key} on '{check_key}': {side_a_row.get(check_key)!r} vs {side_b_row.get(check_key)!r}"
                )
        paired_rows.append(
            {
                **side_a_row,
                "embedding_base": side_a_row["embedding"],
                "embedding_ft": side_b_row["embedding"],
                "hidden_states_base": side_a_row["hidden_states"],
                "hidden_states_ft": side_b_row["hidden_states"],
                "logits_base": side_a_row["logits"],
                "logits_ft": side_b_row["logits"],
                "attention_outputs_base": side_a_row["attention_outputs"],
                "attention_outputs_ft": side_b_row["attention_outputs"],
                "mlp_outputs_base": side_a_row["mlp_outputs"],
                "mlp_outputs_ft": side_b_row["mlp_outputs"],
                "layer_records_base": side_a_row.get("layer_records"),
                "layer_records_ft": side_b_row.get("layer_records"),
                "collection_text_base": side_a_row.get("collection_text"),
                "collection_text_ft": side_b_row.get("collection_text"),
                "label": int(side_a_row.get("label", side_a_row.get("harmfulness_label_ft", 0))),
                "base_model_key": side_a_payload.get("model_key") if isinstance(side_a_payload, dict) else None,
                "finetuned_model_key": side_b_payload.get("model_key") if isinstance(side_b_payload, dict) else None,
            }
        )

    return {
        "dataset_path_base": str(side_a_payload.get("dataset_path")) if isinstance(side_a_payload, dict) and side_a_payload.get("dataset_path") is not None else None,
        "dataset_path_finetuned": str(side_b_payload.get("dataset_path")) if isinstance(side_b_payload, dict) and side_b_payload.get("dataset_path") is not None else None,
        "lm_head_weight_base": side_a_payload.get("lm_head_weight") if isinstance(side_a_payload, dict) else None,
        "lm_head_weight_ft": side_b_payload.get("lm_head_weight") if isinstance(side_b_payload, dict) else None,
        "rows": paired_rows,
    }


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
    save_collected_activation_dataset_plots(
        output_root=output_root,
        layers=layers,
        logit_norms=logit_norms,
        logit_norm_mode=logit_norm_mode,
        labels=labels,
        auroc_array=auroc_array,
        pca_top1_array=pca_top1_array,
        heatmap_matrix=heatmap_matrix,
        cumulative_variance=cumulative_variance,
        selected_layers=selected_layers,
    )

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
    save_paired_collected_activation_dataset_plots(
        output_root=output_root,
        layers=layers,
        logit_diff=logit_diff,
        latent_auroc_array=latent_auroc_array,
        logit_auroc_array=logit_auroc_array,
        pca_top1_array=pca_top1_array,
        combined_matrix=combined_matrix,
        heatmap_matrix=heatmap_matrix,
        attn_diff=attn_diff,
        mlp_diff=mlp_diff,
    )

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

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    layers = np.arange(num_layers)
    save_full_prism_analysis_plots(
        output_root=output_root,
        layers=layers,
        hidden_diff=hidden_diff,
        logit_diff=logit_diff,
        final_logit_diff=final_logit_diff,
        auroc_values=auroc_values,
        logit_auroc=logit_auroc,
        pca_top1=pca_top1,
        total_diff=total_diff,
        emb_contrib_plot=emb_contrib,
        attn_contrib=attn_diff,
        mlp_contrib=mlp_diff,
    )

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
    tokenizer = load_tokenizer(tokenizer_path)

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
    save_group_token_divergence_plot(
        output_path=output_path,
        per_row=per_row,
        group_id=group_id,
        layer_indices=layer_indices,
        metric_label=metric_labels[metric],
        global_max=global_max,
    )

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
                "id": row.get("id"),
                "group_id": row.get("group_id"),
                "variant": row.get("variant"),
                "collection_text": row.get("collection_text"),
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
                "layer_records_base": row.get("layer_records_base"),
                "layer_records_ft": row.get("layer_records_ft"),
                "label": int(row.get("label", row.get("harmfulness_label_ft", 0))),
            }
        )
    return coerced


def _explained_variance_from_singular_values(singular_values: np.ndarray) -> np.ndarray:
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.ndim != 1:
        raise ValueError("singular_values must be 1D")
    if singular_values.size == 0:
        return np.zeros(0, dtype=np.float64)
    squared = singular_values ** 2
    total = float(np.sum(squared))
    if total <= 0.0:
        return np.zeros_like(squared)
    return squared / total


def _rank_at_threshold(cumulative: np.ndarray, threshold: float) -> int:
    if cumulative.size == 0:
        return 0
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def _effective_rank(explained: np.ndarray) -> float:
    explained = np.asarray(explained, dtype=np.float64)
    explained = explained[explained > 1e-12]
    if explained.size == 0:
        return 0.0
    entropy = -np.sum(explained * np.log(explained))
    return float(np.exp(entropy))


def _participation_ratio(singular_values: np.ndarray) -> float:
    singular_values = np.asarray(singular_values, dtype=np.float64)
    squared = singular_values ** 2
    denom = float(np.sum(squared ** 2))
    if denom <= 0.0:
        return 0.0
    numer = float(np.sum(squared)) ** 2
    return numer / denom


def _compute_svd_summary(matrix: np.ndarray, *, top_components: int = 8) -> Dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix for SVD, got shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"SVD matrix must be non-empty, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        bad_count = int(np.size(matrix) - np.count_nonzero(np.isfinite(matrix)))
        raise ValueError(
            f"SVD matrix contains non-finite values (NaN/inf): shape={matrix.shape}, bad_count={bad_count}"
        )

    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False, compute_uv=True)
    singular_values = singular_values.astype(np.float64, copy=False)
    explained = _explained_variance_from_singular_values(singular_values)
    cumulative = np.cumsum(explained)
    k = max(1, min(int(top_components), singular_values.shape[0]))
    top_scores = (u[:, :k] * singular_values[:k]).astype(np.float32, copy=False)
    top_vh = vh[:k].astype(np.float32, copy=False)
    return {
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "singular_values": singular_values.tolist(),
        "explained_variance_ratio": explained.tolist(),
        "cumulative_explained_variance": cumulative.tolist(),
        "saved_top_components": int(k),
        "top_component_scores": top_scores.tolist(),
        "top_right_singular_vectors_vh": top_vh.tolist(),
        "rank_80": _rank_at_threshold(cumulative, 0.80),
        "rank_90": _rank_at_threshold(cumulative, 0.90),
        "rank_95": _rank_at_threshold(cumulative, 0.95),
        "effective_rank": _effective_rank(explained),
        "participation_ratio": _participation_ratio(singular_values),
    }


def _compute_component_scores(matrix: np.ndarray, vh_rows: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    vh_rows = np.asarray(vh_rows, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix for component scores, got shape {matrix.shape}")
    if vh_rows.ndim != 2:
        raise ValueError(f"Expected 2D right-singular-vector matrix, got shape {vh_rows.shape}")
    if matrix.shape[1] != vh_rows.shape[1]:
        raise ValueError(
            f"Feature mismatch for component scores: matrix has {matrix.shape[1]} cols, vh_rows has {vh_rows.shape[1]}"
        )
    return (matrix @ vh_rows.T).astype(np.float32, copy=False)


def _extract_last_token_layer_record_logits(
    rows: Sequence[Dict[str, Any]],
    *,
    key: str,
    norm_mode: str,
    expected_num_block_layers: int,
) -> List[np.ndarray] | None:
    if not rows or rows[0].get(key) is None:
        return None

    record_key = f"logits_{norm_mode}"
    sample_records = rows[0].get(key)
    if not isinstance(sample_records, list) or len(sample_records) < expected_num_block_layers + 1:
        return None

    layer_arrays: List[List[np.ndarray]] = [[] for _ in range(expected_num_block_layers)]
    for row_idx, row in enumerate(rows):
        records = row.get(key)
        if not isinstance(records, list) or len(records) < expected_num_block_layers + 1:
            return None
        block_records = records[1 : expected_num_block_layers + 1]
        for layer_idx, record in enumerate(block_records):
            if record_key not in record:
                return None
            logits = _ensure_tensor(record[record_key], name=f"{key}[{row_idx}][{layer_idx}].{record_key}")
            if logits.ndim != 3:
                raise ValueError(f"{key}[{row_idx}][{layer_idx}].{record_key} must have shape [batch, seq, vocab]")
            layer_arrays[layer_idx].append(logits[:, -1, :].reshape(-1).numpy())
    return [np.stack(values, axis=0).astype(np.float32, copy=False) for values in layer_arrays]


def _project_last_token_hidden_to_logits_by_layer(
    rows: Sequence[Dict[str, Any]],
    *,
    hidden_key: str,
    lm_head_weight: np.ndarray,
    expected_num_block_layers: int,
) -> List[np.ndarray]:
    layer_arrays: List[List[np.ndarray]] = [[] for _ in range(expected_num_block_layers)]
    for row_idx, row in enumerate(rows):
        hidden_states = row[hidden_key]
        usable = hidden_states[1 : expected_num_block_layers + 1]
        if len(usable) != expected_num_block_layers:
            raise ValueError(
                f"{hidden_key} layer count mismatch at row {row_idx}: {len(usable)} vs {expected_num_block_layers}"
            )
        for layer_idx, tensor in enumerate(usable):
            hidden = _ensure_tensor(tensor, name=f"{hidden_key}[{row_idx}][{layer_idx}]")
            if hidden.ndim != 3:
                raise ValueError(f"{hidden_key}[{row_idx}][{layer_idx}] must have shape [batch, seq, dim]")
            projected = hidden[:, -1, :].numpy() @ lm_head_weight.T
            layer_arrays[layer_idx].append(projected.astype(np.float32, copy=False).reshape(-1))
    return [np.stack(values, axis=0).astype(np.float32, copy=False) for values in layer_arrays]


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


def analyze_paired_activation_svd(
    data_or_path: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any],
    *,
    output_dir: str | Path = "plots_svd",
    side_a_name: str = "A",
    side_b_name: str = "B",
    norm_mode: str = "model_norm",
    selected_layers: Sequence[int] | None = None,
    top_components: int = 8,
) -> Dict[str, Any]:
    if isinstance(data_or_path, (str, Path)):
        payload = torch.load(Path(data_or_path), map_location="cpu")
        input_path = str(Path(data_or_path))
    else:
        payload = data_or_path
        input_path = None

    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else list(data_or_path)  # type: ignore[arg-type]
    paired_rows = load_paired_collected_activation_dataset_from_payload({"rows": list(rows)})
    lm_head_weight_base, lm_head_weight_ft = _validate_required_prism_fields(payload, paired_rows)

    if not paired_rows:
        raise ValueError("Paired activation dataset is empty")

    num_block_layers = len(paired_rows[0]["attention_outputs_base"])
    if num_block_layers < 1:
        raise ValueError("Expected at least one transformer block for SVD analysis")

    hidden_base_by_layer = _extract_last_token_hidden_by_layer_from_key(
        paired_rows,
        key="hidden_states_base",
        drop_embedding_if_present=True,
        expected_num_block_layers=num_block_layers,
    )
    hidden_ft_by_layer = _extract_last_token_hidden_by_layer_from_key(
        paired_rows,
        key="hidden_states_ft",
        drop_embedding_if_present=True,
        expected_num_block_layers=num_block_layers,
    )
    hidden_delta_by_layer = [
        (h_ft - h_base).astype(np.float32, copy=False)
        for h_base, h_ft in zip(hidden_base_by_layer, hidden_ft_by_layer)
    ]

    logit_base_by_layer = _extract_last_token_layer_record_logits(
        paired_rows,
        key="layer_records_base",
        norm_mode=norm_mode,
        expected_num_block_layers=num_block_layers,
    )
    logit_ft_by_layer = _extract_last_token_layer_record_logits(
        paired_rows,
        key="layer_records_ft",
        norm_mode=norm_mode,
        expected_num_block_layers=num_block_layers,
    )
    logit_source = f"layer_records_{norm_mode}"
    if logit_base_by_layer is None or logit_ft_by_layer is None:
        logit_base_by_layer = _project_last_token_hidden_to_logits_by_layer(
            paired_rows,
            hidden_key="hidden_states_base",
            lm_head_weight=lm_head_weight_base,
            expected_num_block_layers=num_block_layers,
        )
        logit_ft_by_layer = _project_last_token_hidden_to_logits_by_layer(
            paired_rows,
            hidden_key="hidden_states_ft",
            lm_head_weight=lm_head_weight_ft,
            expected_num_block_layers=num_block_layers,
        )
        logit_source = "projected_hidden_raw"
    logit_delta_by_layer = [
        (z_ft - z_base).astype(np.float32, copy=False)
        for z_base, z_ft in zip(logit_base_by_layer, logit_ft_by_layer)
    ]

    embedding_base = np.stack(
        [
            _ensure_tensor(row["embedding_base"], name=f"embedding_base[{idx}]")[:, -1, :].reshape(-1).numpy()
            for idx, row in enumerate(paired_rows)
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    embedding_ft = np.stack(
        [
            _ensure_tensor(row["embedding_ft"], name=f"embedding_ft[{idx}]")[:, -1, :].reshape(-1).numpy()
            for idx, row in enumerate(paired_rows)
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    embedding_delta = (embedding_ft - embedding_base).astype(np.float32, copy=False)

    hidden_summaries = [_compute_svd_summary(delta, top_components=top_components) for delta in hidden_delta_by_layer]
    logit_summaries = [_compute_svd_summary(delta, top_components=top_components) for delta in logit_delta_by_layer]
    embedding_summary = _compute_svd_summary(embedding_delta, top_components=top_components)

    if selected_layers is None:
        selected_layers = sorted(set([0, max(0, num_block_layers // 2), num_block_layers - 1]))

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    layers = np.arange(num_block_layers)
    hidden_rank_90 = np.asarray([summary["rank_90"] for summary in hidden_summaries], dtype=float)
    logit_rank_90 = np.asarray([summary["rank_90"] for summary in logit_summaries], dtype=float)
    hidden_effective = np.asarray([summary["effective_rank"] for summary in hidden_summaries], dtype=float)
    logit_effective = np.asarray([summary["effective_rank"] for summary in logit_summaries], dtype=float)
    top_k = max(1, int(top_components))
    selected_layers = [int(layer) for layer in selected_layers if 0 <= int(layer) < num_block_layers]
    save_paired_activation_svd_plots(
        output_root=output_root,
        layers=layers,
        hidden_rank_90=hidden_rank_90,
        logit_rank_90=logit_rank_90,
        hidden_effective=hidden_effective,
        logit_effective=logit_effective,
        top_k=top_k,
        hidden_summaries=hidden_summaries,
        logit_summaries=logit_summaries,
        selected_layers=selected_layers,
        hidden_base_by_layer=hidden_base_by_layer,
        hidden_ft_by_layer=hidden_ft_by_layer,
        logit_base_by_layer=logit_base_by_layer,
        logit_ft_by_layer=logit_ft_by_layer,
        side_a_name=side_a_name,
        side_b_name=side_b_name,
        compute_component_scores=_compute_component_scores,
    )

    summary = {
        "input_path": input_path,
        "side_a_name": side_a_name,
        "side_b_name": side_b_name,
        "delta_definition": f"{side_b_name} - {side_a_name}",
        "num_examples": len(paired_rows),
        "num_layers": num_block_layers,
        "norm_mode": norm_mode,
        "logit_source": logit_source,
        "top_components": top_k,
        "selected_layers": list(selected_layers),
        "embedding": embedding_summary,
        "hidden_by_layer": hidden_summaries,
        "logit_by_layer": logit_summaries,
        "plot_paths": {
            "rank_summary": str(output_root / "svd_rank_summary.png"),
            "explained_variance_heatmaps": str(output_root / "svd_explained_variance_heatmaps.png"),
            "cumulative_selected_layers": str(output_root / "svd_cumulative_selected_layers.png"),
            "geometry_selected_layers": str(output_root / "svd_geometry_selected_layers.png"),
        },
    }
    (output_root / "svd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-12)


def _kl_from_probs_np(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)), axis=-1)


def _js_from_probs_np(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * _kl_from_probs_np(p, m) + 0.5 * _kl_from_probs_np(q, m)


def _tvd_from_probs_np(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(np.abs(p - q), axis=-1)


def _jaccard_topk_from_logits_np(logits_a: np.ndarray, logits_b: np.ndarray, top_k: int) -> np.ndarray:
    top_a = np.argpartition(logits_a, -top_k, axis=-1)[:, -top_k:]
    top_b = np.argpartition(logits_b, -top_k, axis=-1)[:, -top_k:]
    scores: List[float] = []
    for a_ids, b_ids in zip(top_a, top_b):
        set_a = set(int(x) for x in a_ids.tolist())
        set_b = set(int(x) for x in b_ids.tolist())
        union = set_a | set_b
        scores.append(float(len(set_a & set_b)) / float(len(union)) if union else 0.0)
    return np.asarray(scores, dtype=np.float32)


def _topk_token_ids_from_logits_np(logits: np.ndarray, top_k: int) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits array for top-k extraction, got shape {logits.shape}")
    k = max(1, min(int(top_k), int(logits.shape[-1])))
    top_ids = np.argpartition(logits, -k, axis=-1)[:, -k:]
    top_logits = np.take_along_axis(logits, top_ids, axis=-1)
    order = np.argsort(-top_logits, axis=-1)
    return np.take_along_axis(top_ids, order, axis=-1)


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> List[str]:
    if tokenizer is None:
        return [str(int(x)) for x in token_ids]
    return [str(tok) for tok in tokenizer.convert_ids_to_tokens([int(x) for x in token_ids])]


def _layer_region_summary(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Expected a non-empty 1D layerwise metric array")
    if arr.size == 1:
        v = float(arr[0])
        return {"early": v, "mid": v, "late": v, "output": v, "auc_d": v}

    internal = arr[:-1]
    internal_idx = np.arange(internal.size)
    bins = np.array_split(internal_idx, 3) if internal.size > 0 else [np.array([], dtype=int)] * 3

    def _bin_mean(idx: np.ndarray) -> float:
        if idx.size == 0:
            return float("nan")
        return float(np.mean(internal[idx]))

    x = np.linspace(0.0, 1.0, arr.size, dtype=float)
    return {
        "early": _bin_mean(bins[0]),
        "mid": _bin_mean(bins[1]),
        "late": _bin_mean(bins[2]),
        "output": float(arr[-1]),
        "auc_d": float(np.trapezoid(arr, x)),
    }


def summarize_paired_teacher_forced_divergence(
    data_or_path: str | Path | Sequence[Dict[str, Any]] | Dict[str, Any],
    *,
    output_dir: str | Path,
    top_k: int = 5,
) -> Dict[str, Any]:
    if isinstance(data_or_path, (str, Path)):
        payload = torch.load(Path(data_or_path), map_location="cpu")
        input_path = str(Path(data_or_path))
    else:
        payload = data_or_path
        input_path = None

    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else list(data_or_path)  # type: ignore[arg-type]
    paired_rows = load_paired_collected_activation_dataset_from_payload({"rows": list(rows)})
    lm_head_weight_base, lm_head_weight_ft = _validate_required_prism_fields(payload, paired_rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = ""
    if isinstance(payload, dict) and "rows" in payload and payload["rows"]:
        tokenizer_path = str(
            payload["rows"][0].get("base_model_path")
            or payload["rows"][0].get("model_path")
            or ""
        )
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path else None

    detailed_records: List[Dict[str, Any]] = []
    metric_names = (
        "hidden_cosine",
        "hidden_l2",
        "jaccard_topk",
        "tvd",
        "kl_a_b",
        "kl_b_a",
        "js",
        "p_gt_base",
        "p_gt_ft",
    )
    layer_metric_buckets: Dict[str, Dict[str, List[List[float]]]] = {
        subset: {metric: [] for metric in metric_names}
        for subset in ("all", "base_response", "finetuned_response")
    }

    for raw_row, _row in zip(rows, paired_rows):
        attention_mask = _ensure_tensor(raw_row["attention_mask"], name=f"attention_mask[{raw_row.get('id')}]").to(dtype=torch.long)
        input_ids = _ensure_tensor(raw_row["input_ids"], name=f"input_ids[{raw_row.get('id')}]").to(dtype=torch.long)
        seq_len = int(attention_mask[0].sum().item()) if attention_mask.numel() > 0 else int(input_ids.shape[1])
        if seq_len < 2:
            continue

        valid_input_ids = input_ids[0, :seq_len].numpy()
        predictor_ids = valid_input_ids[:-1]
        target_ids = valid_input_ids[1:]
        input_tokens = tokenizer.convert_ids_to_tokens(predictor_ids.tolist()) if tokenizer is not None else [str(int(x)) for x in predictor_ids]
        target_tokens = tokenizer.convert_ids_to_tokens(target_ids.tolist()) if tokenizer is not None else [str(int(x)) for x in target_ids]

        num_block_layers = len(raw_row["hidden_states_base"]) - 1
        subset_keys = ["all", str(raw_row.get("variant", ""))]

        for layer_idx in range(num_block_layers):
            h_base = _ensure_tensor(
                raw_row["hidden_states_base"][layer_idx + 1],
                name=f"hidden_states_base[{raw_row.get('id')}][{layer_idx + 1}]",
            )[0, : seq_len - 1, :].numpy()
            h_ft = _ensure_tensor(
                raw_row["hidden_states_ft"][layer_idx + 1],
                name=f"hidden_states_ft[{raw_row.get('id')}][{layer_idx + 1}]",
            )[0, : seq_len - 1, :].numpy()

            logits_base_layer = (h_base @ lm_head_weight_base.T).astype(np.float32, copy=False)
            logits_ft_layer = (h_ft @ lm_head_weight_ft.T).astype(np.float32, copy=False)
            probs_base = _softmax_np(logits_base_layer)
            probs_ft = _softmax_np(logits_ft_layer)

            base_norm = np.linalg.norm(h_base, axis=-1)
            ft_norm = np.linalg.norm(h_ft, axis=-1)
            denom = np.clip(base_norm * ft_norm, 1e-12, None)
            hidden_cosine = np.sum(h_base * h_ft, axis=-1) / denom
            hidden_l2 = np.linalg.norm(h_ft - h_base, axis=-1)
            jaccard_topk = _jaccard_topk_from_logits_np(logits_base_layer, logits_ft_layer, top_k=top_k)
            tvd = _tvd_from_probs_np(probs_base, probs_ft)
            kl_a_b = _kl_from_probs_np(probs_base, probs_ft)
            kl_b_a = _kl_from_probs_np(probs_ft, probs_base)
            js = _js_from_probs_np(probs_base, probs_ft)
            p_gt_base = probs_base[np.arange(target_ids.shape[0]), target_ids]
            p_gt_ft = probs_ft[np.arange(target_ids.shape[0]), target_ids]
            top1_ids_base = _topk_token_ids_from_logits_np(logits_base_layer, top_k=1)
            top1_ids_ft = _topk_token_ids_from_logits_np(logits_ft_layer, top_k=1)
            top5_ids_base = _topk_token_ids_from_logits_np(logits_base_layer, top_k=5)
            top5_ids_ft = _topk_token_ids_from_logits_np(logits_ft_layer, top_k=5)
            top10_ids_base = _topk_token_ids_from_logits_np(logits_base_layer, top_k=10)
            top10_ids_ft = _topk_token_ids_from_logits_np(logits_ft_layer, top_k=10)

            for subset in subset_keys:
                if subset not in layer_metric_buckets:
                    continue
                for metric_name, values in {
                    "hidden_cosine": hidden_cosine,
                    "hidden_l2": hidden_l2,
                    "jaccard_topk": jaccard_topk,
                    "tvd": tvd,
                    "kl_a_b": kl_a_b,
                    "kl_b_a": kl_b_a,
                    "js": js,
                    "p_gt_base": p_gt_base,
                    "p_gt_ft": p_gt_ft,
                }.items():
                    while len(layer_metric_buckets[subset][metric_name]) <= layer_idx:
                        layer_metric_buckets[subset][metric_name].append([])
                    layer_metric_buckets[subset][metric_name][layer_idx].extend(float(x) for x in values.tolist())

            for pos_idx in range(target_ids.shape[0]):
                detailed_records.append(
                    {
                        "row_id": int(raw_row.get("id")),
                        "group_id": str(raw_row.get("group_id")),
                        "variant": str(raw_row.get("variant")),
                        "category": raw_row.get("category"),
                        "type": raw_row.get("type"),
                        "analysis_text": str(raw_row.get("analysis_text")),
                        "position": int(pos_idx),
                        "layer": int(layer_idx),
                        "input_token_id": int(predictor_ids[pos_idx]),
                        "target_token_id": int(target_ids[pos_idx]),
                        "input_token": input_tokens[pos_idx],
                        "target_token": target_tokens[pos_idx],
                        "hidden_cosine": float(hidden_cosine[pos_idx]),
                        "hidden_l2": float(hidden_l2[pos_idx]),
                        "jaccard_topk": float(jaccard_topk[pos_idx]),
                        "tvd": float(tvd[pos_idx]),
                        "kl_a_b": float(kl_a_b[pos_idx]),
                        "kl_b_a": float(kl_b_a[pos_idx]),
                        "js": float(js[pos_idx]),
                        "p_gt_base": float(p_gt_base[pos_idx]),
                        "p_gt_ft": float(p_gt_ft[pos_idx]),
                        "top1_token_id_base": int(top1_ids_base[pos_idx, 0]),
                        "top1_token_id_ft": int(top1_ids_ft[pos_idx, 0]),
                        "top1_token_base": _decode_token_ids(tokenizer, top1_ids_base[pos_idx].tolist())[0],
                        "top1_token_ft": _decode_token_ids(tokenizer, top1_ids_ft[pos_idx].tolist())[0],
                        "top5_token_ids_base": [int(x) for x in top5_ids_base[pos_idx].tolist()],
                        "top5_token_ids_ft": [int(x) for x in top5_ids_ft[pos_idx].tolist()],
                        "top5_tokens_base": _decode_token_ids(tokenizer, top5_ids_base[pos_idx].tolist()),
                        "top5_tokens_ft": _decode_token_ids(tokenizer, top5_ids_ft[pos_idx].tolist()),
                        "top10_token_ids_base": [int(x) for x in top10_ids_base[pos_idx].tolist()],
                        "top10_token_ids_ft": [int(x) for x in top10_ids_ft[pos_idx].tolist()],
                        "top10_tokens_base": _decode_token_ids(tokenizer, top10_ids_base[pos_idx].tolist()),
                        "top10_tokens_ft": _decode_token_ids(tokenizer, top10_ids_ft[pos_idx].tolist()),
                        "top5_overlap_tokens": sorted(
                            set(_decode_token_ids(tokenizer, top5_ids_base[pos_idx].tolist()))
                            & set(_decode_token_ids(tokenizer, top5_ids_ft[pos_idx].tolist()))
                        ),
                        "top10_overlap_tokens": sorted(
                            set(_decode_token_ids(tokenizer, top10_ids_base[pos_idx].tolist()))
                            & set(_decode_token_ids(tokenizer, top10_ids_ft[pos_idx].tolist()))
                        ),
                    }
                )

    detail_path = output_dir / "per_prompt_position_layer_metrics.jsonl"
    with detail_path.open("w", encoding="utf-8") as f:
        for record in detailed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metric_display = {
        "hidden_cosine": "Cosine",
        "hidden_l2": "L2",
        "jaccard_topk": f"Jaccard@{top_k}",
        "tvd": "TVD",
        "kl_a_b": "KL(A||B)",
        "kl_b_a": "KL(B||A)",
        "js": "Jensen-Shannon",
        "p_gt_base": "P(gt) Base",
        "p_gt_ft": "P(gt) Finetuned",
    }

    def _build_summary_table(subset: str) -> List[Dict[str, Any]]:
        rows_out: List[Dict[str, Any]] = []
        for metric_name, layers in layer_metric_buckets[subset].items():
            if not layers:
                continue
            layer_means = np.asarray(
                [float(np.mean(layer_values)) if layer_values else float("nan") for layer_values in layers],
                dtype=float,
            )
            regions = _layer_region_summary(layer_means)
            rows_out.append(
                {
                    "subset": subset,
                    "metric": metric_name,
                    "metric_display": metric_display[metric_name],
                    "early": regions["early"],
                    "mid": regions["mid"],
                    "late": regions["late"],
                    "output": regions["output"],
                    "auc_d": regions["auc_d"],
                    "layerwise_mean": layer_means.tolist(),
                }
            )
        return rows_out

    overall_summary = _build_summary_table("all")
    base_summary = _build_summary_table("base_response")
    ft_summary = _build_summary_table("finetuned_response")

    (output_dir / "aggregate_all_prompts.json").write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")
    (output_dir / "aggregate_base_response.json").write_text(json.dumps(base_summary, indent=2), encoding="utf-8")
    (output_dir / "aggregate_finetuned_response.json").write_text(json.dumps(ft_summary, indent=2), encoding="utf-8")

    return {
        "input_path": input_path,
        "output_dir": str(output_dir),
        "num_records": len(detailed_records),
        "num_source_rows": len(rows),
        "detail_path": str(detail_path),
        "aggregate_all_path": str(output_dir / "aggregate_all_prompts.json"),
        "aggregate_base_path": str(output_dir / "aggregate_base_response.json"),
        "aggregate_finetuned_path": str(output_dir / "aggregate_finetuned_response.json"),
    }

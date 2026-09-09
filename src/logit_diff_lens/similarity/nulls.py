from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from ..schemas.similarity_outputs import LayerPairSimilarityArtifact, ScalarCalibratedSimilarity


def validate_similarity_inputs(
    X: torch.Tensor,
    Y: torch.Tensor,
    metric_name: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    del metadata
    if not torch.is_tensor(X) or not torch.is_tensor(Y):
        raise TypeError(f"{metric_name}: X and Y must be torch.Tensor instances")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"{metric_name}: expected 2D matrices, got X={tuple(X.shape)} Y={tuple(Y.shape)}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"{metric_name}: sample mismatch, X has {X.shape[0]} rows and Y has {Y.shape[0]} rows"
        )
    if X.shape[0] == 0:
        raise ValueError(f"{metric_name}: matrices must be non-empty")
    if not torch.isfinite(X).all():
        raise ValueError(f"{metric_name}: X contains NaN or Inf values")
    if not torch.isfinite(Y).all():
        raise ValueError(f"{metric_name}: Y contains NaN or Inf values")


def validate_layer_similarity_inputs(
    layers_a: Sequence[torch.Tensor],
    layers_b: Sequence[torch.Tensor],
    metric_name: str,
) -> None:
    if not layers_a or not layers_b:
        raise ValueError(f"{metric_name}: layers_a and layers_b must be non-empty")
    sample_count = layers_a[0].shape[0]
    for idx, tensor in enumerate(layers_a):
        if tensor.ndim != 2:
            raise ValueError(f"{metric_name}: layers_a[{idx}] is not 2D")
        if tensor.shape[0] != sample_count:
            raise ValueError(f"{metric_name}: layers_a sample counts do not match")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{metric_name}: layers_a[{idx}] contains NaN or Inf values")
    for idx, tensor in enumerate(layers_b):
        if tensor.ndim != 2:
            raise ValueError(f"{metric_name}: layers_b[{idx}] is not 2D")
        if tensor.shape[0] != sample_count:
            raise ValueError(f"{metric_name}: layers_b sample counts do not match layers_a")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{metric_name}: layers_b[{idx}] contains NaN or Inf values")


def _empirical_right_tail_threshold(
    observed: float,
    null_scores: np.ndarray,
    alpha: float,
) -> float:
    combined = np.concatenate([np.asarray([observed], dtype=np.float64), null_scores.astype(np.float64, copy=False)])
    return float(np.quantile(combined, 1.0 - alpha, method="higher"))


def _resolve_permutation_unit(
    permutation_unit: str,
    group_ids: torch.Tensor | None,
) -> str:
    if permutation_unit not in {"auto", "row", "group"}:
        raise ValueError(f"Unsupported permutation_unit: {permutation_unit!r}")
    if permutation_unit == "row":
        return "row"
    if permutation_unit == "group":
        if group_ids is None:
            raise ValueError("group permutation requires group_ids")
        unique_count = int(torch.unique(group_ids).numel())
        if unique_count < 2:
            raise ValueError("group permutation requires at least two groups")
        return "group"
    if group_ids is not None and int(torch.unique(group_ids).numel()) >= 2:
        return "group"
    return "row"


def _build_group_permutation_indices(
    group_ids: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    unique_groups = torch.unique_consecutive(group_ids)
    if unique_groups.numel() < 2:
        raise ValueError("group permutation requires at least two groups")
    group_to_indices = [torch.nonzero(group_ids == group_id, as_tuple=False).squeeze(-1) for group_id in unique_groups]
    permuted_order = rng.permutation(len(group_to_indices))
    return torch.cat([group_to_indices[int(idx)] for idx in permuted_order], dim=0)


def _right_tail_p_value(observed: float, null_scores: np.ndarray) -> float:
    return float((1.0 + np.sum(null_scores >= observed)) / (len(null_scores) + 1.0))


def _null_percentile(observed: float, null_scores: np.ndarray) -> float:
    return float((1.0 + np.sum(null_scores <= observed)) / (len(null_scores) + 1.0))


def _adjust_pvalues_bh(p_values: np.ndarray) -> np.ndarray:
    if p_values.size == 0:
        return p_values.astype(np.float64, copy=True)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty_like(ranked, dtype=np.float64)
    running = 1.0
    m = len(ranked)
    for idx in range(m - 1, -1, -1):
        rank = idx + 1
        running = min(running, (ranked[idx] * m) / rank)
        adjusted[idx] = running
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    return restored


def _adjust_pvalues_holm(p_values: np.ndarray) -> np.ndarray:
    if p_values.size == 0:
        return p_values.astype(np.float64, copy=True)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty_like(ranked, dtype=np.float64)
    m = len(ranked)
    running = 0.0
    for idx in range(m):
        adjusted_value = (m - idx) * ranked[idx]
        running = max(running, adjusted_value)
        adjusted[idx] = running
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    return restored


def adjust_pvalues(p_values: Sequence[float], method: str) -> list[float]:
    values = np.asarray(list(p_values), dtype=np.float64)
    if method == "none":
        adjusted = np.clip(values, 0.0, 1.0)
    elif method == "fdr_bh":
        adjusted = _adjust_pvalues_bh(values)
    elif method == "holm":
        adjusted = _adjust_pvalues_holm(values)
    else:
        raise ValueError(f"Unsupported multiple-testing method: {method!r}")
    return adjusted.astype(float).tolist()


def _calibrated_score(observed: float, threshold: float, similarity_max: float | None) -> float:
    if observed <= threshold:
        return 0.0
    if similarity_max is None:
        return float(observed - threshold)
    denom = similarity_max - threshold
    if denom <= 0:
        return 1.0 if observed >= similarity_max else 0.0
    return float(max(0.0, min(1.0, (observed - threshold) / denom)))


def calibrate_scalar_similarity(
    X: torch.Tensor,
    Y: torch.Tensor,
    sim_fn: Callable[[torch.Tensor, torch.Tensor], float],
    *,
    metric_name: str,
    representation_kind: str,
    num_permutations: int = 1000,
    alpha: float = 0.05,
    similarity_max: float | None = 1.0,
    permutation_unit: str = "auto",
    group_ids: torch.Tensor | None = None,
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ScalarCalibratedSimilarity:
    validate_similarity_inputs(X, Y, metric_name, metadata)
    rng = np.random.default_rng(seed)
    X_cpu = X.detach().to(dtype=torch.float32, device="cpu")
    Y_cpu = Y.detach().to(dtype=torch.float32, device="cpu")
    group_ids_cpu = None if group_ids is None else group_ids.detach().to(device="cpu", dtype=torch.long)
    resolved_permutation_unit = _resolve_permutation_unit(permutation_unit, group_ids_cpu)
    observed = float(sim_fn(X_cpu, Y_cpu))
    null_scores = np.empty(num_permutations, dtype=np.float64)
    n = X_cpu.shape[0]
    for idx in range(num_permutations):
        if resolved_permutation_unit == "group":
            perm = _build_group_permutation_indices(group_ids_cpu, rng)
        else:
            perm = torch.as_tensor(rng.permutation(n), dtype=torch.long)
        Y_perm = Y_cpu[perm]
        null_scores[idx] = float(sim_fn(X_cpu, Y_perm))
    critical_value = _empirical_right_tail_threshold(observed, null_scores, alpha)
    p_value = _right_tail_p_value(observed, null_scores)
    null_percentile = _null_percentile(observed, null_scores)
    calibrated = _calibrated_score(observed, critical_value, similarity_max)
    return ScalarCalibratedSimilarity(
        metric_name=metric_name,
        representation_kind=representation_kind,
        raw_similarity=observed,
        null_scores=null_scores.astype(float).tolist(),
        null_mean=float(null_scores.mean()) if len(null_scores) else 0.0,
        null_std=float(null_scores.std(ddof=0)) if len(null_scores) else 0.0,
        critical_value=critical_value,
        p_value=p_value,
        adjusted_p_value=None,
        null_percentile=null_percentile,
        calibrated_similarity=calibrated,
        sample_count=int(X_cpu.shape[0]),
        feature_dim_a=int(X_cpu.shape[1]),
        feature_dim_b=int(Y_cpu.shape[1]),
        metadata={
            "permutation_unit": resolved_permutation_unit,
            **dict(metadata or {}),
        },
    )


def calibrate_layer_pair_similarity(
    layers_a: Sequence[torch.Tensor],
    layers_b: Sequence[torch.Tensor],
    sim_fn: Callable[[torch.Tensor, torch.Tensor], float],
    *,
    metric_name: str,
    representation_kind: str,
    layer_indices_a: Sequence[int],
    layer_indices_b: Sequence[int],
    aggregate: str = "max",
    num_permutations: int = 1000,
    alpha: float = 0.05,
    similarity_max: float | None = 1.0,
    permutation_unit: str = "auto",
    group_ids: torch.Tensor | None = None,
    multiple_testing_method: str = "fdr_bh",
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> LayerPairSimilarityArtifact:
    if aggregate != "max":
        raise ValueError(f"Unsupported aggregate mode: {aggregate!r}")
    if len(layers_a) != len(layer_indices_a) or len(layers_b) != len(layer_indices_b):
        raise ValueError("Layer index lists must match layer tensor lists")
    validate_layer_similarity_inputs(layers_a, layers_b, metric_name)
    rng = np.random.default_rng(seed)
    rows = len(layers_a)
    cols = len(layers_b)
    raw_matrix = torch.empty((rows, cols), dtype=torch.float32)
    p_value_matrix = torch.empty((rows, cols), dtype=torch.float32)
    adjusted_p_value_matrix = torch.empty((rows, cols), dtype=torch.float32)
    null_percentile_matrix = torch.empty((rows, cols), dtype=torch.float32)
    calibrated_matrix = torch.empty((rows, cols), dtype=torch.float32)
    null_mean_matrix = torch.empty((rows, cols), dtype=torch.float32)
    sample_count = int(layers_a[0].shape[0])
    group_ids_cpu = None if group_ids is None else group_ids.detach().to(device="cpu", dtype=torch.long)
    resolved_permutation_unit = _resolve_permutation_unit(permutation_unit, group_ids_cpu)

    perm_indices = []
    for _ in range(num_permutations):
        if resolved_permutation_unit == "group":
            perm_indices.append(_build_group_permutation_indices(group_ids_cpu, rng))
        else:
            perm_indices.append(torch.as_tensor(rng.permutation(sample_count), dtype=torch.long))
    aggregate_null_scores = np.empty(num_permutations, dtype=np.float64)
    permuted_maxima = np.empty((num_permutations, rows, cols), dtype=np.float64)

    layers_a_cpu = [x.detach().to(dtype=torch.float32, device="cpu") for x in layers_a]
    layers_b_cpu = [x.detach().to(dtype=torch.float32, device="cpu") for x in layers_b]

    for i, Xa in enumerate(layers_a_cpu):
        for j, Yb in enumerate(layers_b_cpu):
            observed = float(sim_fn(Xa, Yb))
            raw_matrix[i, j] = observed
            null_scores = np.empty(num_permutations, dtype=np.float64)
            for k, perm in enumerate(perm_indices):
                null_scores[k] = float(sim_fn(Xa, Yb[perm]))
                permuted_maxima[k, i, j] = null_scores[k]
            critical_value = _empirical_right_tail_threshold(observed, null_scores, alpha)
            p_value = _right_tail_p_value(observed, null_scores)
            null_percentile = _null_percentile(observed, null_scores)
            calibrated = _calibrated_score(observed, critical_value, similarity_max)
            p_value_matrix[i, j] = p_value
            null_percentile_matrix[i, j] = null_percentile
            calibrated_matrix[i, j] = calibrated
            null_mean_matrix[i, j] = float(null_scores.mean()) if len(null_scores) else 0.0

    adjusted_values = adjust_pvalues(p_value_matrix.flatten().tolist(), multiple_testing_method)
    adjusted_p_value_matrix[:] = torch.tensor(adjusted_values, dtype=torch.float32).reshape(rows, cols)

    observed_max = float(raw_matrix.max().item())
    for k in range(num_permutations):
        aggregate_null_scores[k] = float(permuted_maxima[k].max())
    aggregate_critical_value = _empirical_right_tail_threshold(observed_max, aggregate_null_scores, alpha)
    aggregate_p_value = _right_tail_p_value(observed_max, aggregate_null_scores)
    aggregate_null_percentile = _null_percentile(observed_max, aggregate_null_scores)
    aggregate_calibrated = _calibrated_score(observed_max, aggregate_critical_value, similarity_max)

    return LayerPairSimilarityArtifact(
        metric_name=metric_name,
        representation_kind=representation_kind,
        raw_matrix=raw_matrix,
        calibrated_matrix=calibrated_matrix,
        p_value_matrix=p_value_matrix,
        adjusted_p_value_matrix=adjusted_p_value_matrix,
        null_percentile_matrix=null_percentile_matrix,
        null_mean_matrix=null_mean_matrix,
        layer_indices_a=[int(v) for v in layer_indices_a],
        layer_indices_b=[int(v) for v in layer_indices_b],
        aggregate_statistic_name=aggregate,
        aggregate_raw_value=observed_max,
        aggregate_null_scores=aggregate_null_scores.astype(float).tolist(),
        aggregate_critical_value=aggregate_critical_value,
        aggregate_p_value=aggregate_p_value,
        aggregate_adjusted_p_value=aggregate_p_value,
        aggregate_null_percentile=aggregate_null_percentile,
        aggregate_calibrated_value=aggregate_calibrated,
        metadata={
            "multiple_testing_method": multiple_testing_method,
            "permutation_unit": resolved_permutation_unit,
            **dict(metadata or {}),
        },
    )


__all__ = [
    "adjust_pvalues",
    "calibrate_layer_pair_similarity",
    "calibrate_scalar_similarity",
    "validate_layer_similarity_inputs",
    "validate_similarity_inputs",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class LatentShiftAnalysisExample:
    hidden_states_base: Sequence[torch.Tensor]
    hidden_states_ft: Sequence[torch.Tensor]
    logits_base: Sequence[torch.Tensor]
    logits_ft: Sequence[torch.Tensor]
    label_harmful: bool


def _coerce_latent_shift_example(
    example: LatentShiftAnalysisExample | Dict[str, Any]
) -> LatentShiftAnalysisExample:
    if isinstance(example, LatentShiftAnalysisExample):
        return example
    return LatentShiftAnalysisExample(
        hidden_states_base=example["hidden_states_base"],
        hidden_states_ft=example["hidden_states_ft"],
        logits_base=example["logits_base"],
        logits_ft=example["logits_ft"],
        label_harmful=bool(example["label_harmful"]),
    )


def _validate_hidden_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [1, seq, hidden], got {tuple(tensor.shape)}")
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} batch size must be 1, got {tensor.shape[0]}")
    if tensor.shape[1] < 1:
        raise ValueError(f"{name} seq_len must be >= 1, got {tensor.shape[1]}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or inf values")


def _validate_logits_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [1, seq, vocab], got {tuple(tensor.shape)}")
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} batch size must be 1, got {tensor.shape[0]}")
    if tensor.shape[1] < 1:
        raise ValueError(f"{name} seq_len must be >= 1, got {tensor.shape[1]}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or inf values")


def _extract_last_hidden(hidden_state: torch.Tensor) -> np.ndarray:
    _validate_hidden_tensor("hidden_state", hidden_state)
    return hidden_state[0, -1, :].detach().to(dtype=torch.float32, device="cpu").numpy()


def _extract_last_logits(logits: torch.Tensor) -> np.ndarray:
    _validate_logits_tensor("logits", logits)
    return logits[0, -1, :].detach().to(dtype=torch.float32, device="cpu").numpy()


def _validate_examples(examples: Sequence[LatentShiftAnalysisExample]) -> int:
    if not examples:
        raise ValueError("examples must be non-empty")
    num_layers = len(examples[0].hidden_states_base)
    if num_layers < 1:
        raise ValueError("hidden_states_base must contain at least one layer")
    if len(examples[0].hidden_states_ft) != num_layers:
        raise ValueError("hidden_states_ft layer count mismatch in first example")
    if len(examples[0].logits_base) != num_layers or len(examples[0].logits_ft) != num_layers:
        raise ValueError("logit layer count mismatch in first example")

    for example_idx, example in enumerate(examples):
        if len(example.hidden_states_base) != num_layers:
            raise ValueError(f"Base hidden layer count mismatch in example {example_idx}")
        if len(example.hidden_states_ft) != num_layers:
            raise ValueError(f"FT hidden layer count mismatch in example {example_idx}")
        if len(example.logits_base) != num_layers:
            raise ValueError(f"Base logit layer count mismatch in example {example_idx}")
        if len(example.logits_ft) != num_layers:
            raise ValueError(f"FT logit layer count mismatch in example {example_idx}")
        for layer_idx in range(num_layers):
            base_hidden = example.hidden_states_base[layer_idx]
            ft_hidden = example.hidden_states_ft[layer_idx]
            base_logits = example.logits_base[layer_idx]
            ft_logits = example.logits_ft[layer_idx]
            _validate_hidden_tensor(f"hidden_states_base[{example_idx}][{layer_idx}]", base_hidden)
            _validate_hidden_tensor(f"hidden_states_ft[{example_idx}][{layer_idx}]", ft_hidden)
            _validate_logits_tensor(f"logits_base[{example_idx}][{layer_idx}]", base_logits)
            _validate_logits_tensor(f"logits_ft[{example_idx}][{layer_idx}]", ft_logits)
            if base_hidden.shape[-1] != ft_hidden.shape[-1]:
                raise ValueError(
                    f"Hidden dim mismatch at example {example_idx}, layer {layer_idx}: "
                    f"{base_hidden.shape[-1]} vs {ft_hidden.shape[-1]}"
                )
            if base_logits.shape[-1] != ft_logits.shape[-1]:
                raise ValueError(
                    f"Logit vocab mismatch at example {example_idx}, layer {layer_idx}: "
                    f"{base_logits.shape[-1]} vs {ft_logits.shape[-1]}"
                )
    return num_layers


def _stack_hidden_layer(
    examples: Sequence[LatentShiftAnalysisExample],
    *,
    layer_idx: int,
    source: str,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for example in examples:
        if source == "base":
            rows.append(_extract_last_hidden(example.hidden_states_base[layer_idx]))
        elif source == "ft":
            rows.append(_extract_last_hidden(example.hidden_states_ft[layer_idx]))
        else:
            raise ValueError(f"Unsupported hidden source {source}")
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _stack_logits_layer(
    examples: Sequence[LatentShiftAnalysisExample],
    *,
    layer_idx: int,
    source: str,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for example in examples:
        if source == "base":
            rows.append(_extract_last_logits(example.logits_base[layer_idx]))
        elif source == "ft":
            rows.append(_extract_last_logits(example.logits_ft[layer_idx]))
        else:
            raise ValueError(f"Unsupported logit source {source}")
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return x / safe_norms


def _normalize_vector(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm <= 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return (x / norm).astype(np.float32, copy=False)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError(f"Pearson inputs must match, got {x.shape} vs {y.shape}")
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    if x.size < 2:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def _safe_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size != 2:
        raise ValueError(f"AUROC requires both classes, got labels={unique.tolist()}")
    if np.allclose(scores, scores[0]):
        return 0.5
    return float(roc_auc_score(labels, scores))


def _mean_pairwise_cosine(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    normalized = _normalize_rows(x)
    sim = normalized @ normalized.T
    upper = sim[np.triu_indices(sim.shape[0], k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.mean(upper))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted, dtype=np.float32)
    denom = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / denom


def _kl_per_example(p_logits: np.ndarray, q_logits: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    p = _softmax(p_logits.astype(np.float32, copy=False))
    q = _softmax(q_logits.astype(np.float32, copy=False))
    p_safe = np.clip(p, epsilon, None)
    q_safe = np.clip(q, epsilon, None)
    kl = np.sum(p_safe * np.log(p_safe / q_safe), axis=1)
    if not np.isfinite(kl).all():
        raise ValueError("KL divergence contains NaN or inf values")
    return kl.astype(np.float32, copy=False)


def _fit_probe(x: np.ndarray, y: np.ndarray, *, random_seed: int) -> Dict[str, float]:
    if x.shape[0] < 5:
        raise ValueError("Need at least 5 examples for train/test probing")
    unique = np.unique(y)
    if unique.size != 2:
        raise ValueError(f"Probe requires both classes, got labels={unique.tolist()}")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=int(random_seed),
        stratify=y,
    )
    probe = LogisticRegression(max_iter=1000, random_state=int(random_seed))
    probe.fit(x_train, y_train)
    y_pred = probe.predict(x_test)
    y_score = probe.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auroc": float(roc_auc_score(y_test, y_score)) if not np.allclose(y_score, y_score[0]) else 0.5,
    }


def _fit_pca_safe(x: np.ndarray, *, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    centered = x - np.mean(x, axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return (
            np.zeros((n_components,), dtype=np.float32),
            np.zeros((n_components, x.shape[1]), dtype=np.float32),
        )
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(x)
    explained = np.nan_to_num(
        pca.explained_variance_ratio_.astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    components = np.nan_to_num(
        pca.components_.astype(np.float32, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return explained, components


def analyze_latent_shift_structure(
    examples: Sequence[LatentShiftAnalysisExample | Dict[str, Any]],
    *,
    random_seed: int = 0,
    pca_components: int = 3,
    subspace_top_k: int = 10,
) -> Dict[str, Any]:
    coerced = [_coerce_latent_shift_example(example) for example in examples]
    num_layers = _validate_examples(coerced)
    labels = np.array([int(example.label_harmful) for example in coerced], dtype=np.int64)
    np.random.seed(int(random_seed))

    pca_results: List[Dict[str, Any]] = []
    delta_alignment: List[Dict[str, Any]] = []
    projection_scores: List[Dict[str, Any]] = []
    logit_shift: List[Dict[str, Any]] = []
    kl_divergence: List[Dict[str, Any]] = []
    latent_to_logit_corr: List[Dict[str, Any]] = []
    latent_to_behavior: List[Dict[str, Any]] = []
    logit_to_behavior: List[Dict[str, Any]] = []
    probe_results: List[Dict[str, Any]] = []
    subspace_shift: List[Dict[str, Any]] = []
    delta_stats: List[Dict[str, Any]] = []

    for layer_idx in range(num_layers):
        h_base = _stack_hidden_layer(coerced, layer_idx=layer_idx, source="base")
        h_ft = _stack_hidden_layer(coerced, layer_idx=layer_idx, source="ft")
        logits_base = _stack_logits_layer(coerced, layer_idx=layer_idx, source="base")
        logits_ft = _stack_logits_layer(coerced, layer_idx=layer_idx, source="ft")

        delta_hidden = (h_ft - h_base).astype(np.float32, copy=False)
        mean_delta = np.mean(delta_hidden, axis=0).astype(np.float32, copy=False)
        delta_l2 = np.linalg.norm(delta_hidden, axis=1).astype(np.float32, copy=False)

        n_components = int(min(pca_components, delta_hidden.shape[0], delta_hidden.shape[1]))
        explained_variance_ratio, pca_components_matrix = _fit_pca_safe(
            delta_hidden,
            n_components=n_components,
        )
        components = pca_components_matrix[: min(3, pca_components_matrix.shape[0])]

        base_norm = _normalize_rows(h_base)
        mean_delta_norm = _normalize_vector(mean_delta)
        proj_scores = (base_norm @ mean_delta_norm).astype(np.float32, copy=False)

        delta_logits = (logits_ft - logits_base).astype(np.float32, copy=False)
        logit_l2 = np.linalg.norm(delta_logits, axis=1).astype(np.float32, copy=False)
        kl_values = _kl_per_example(logits_base, logits_ft)

        latent_to_logit = _safe_pearson(proj_scores, logit_l2)
        latent_auroc = _safe_auroc(proj_scores, labels)
        logit_auroc = _safe_auroc(logit_l2, labels)
        probe = _fit_probe(h_base, labels, random_seed=random_seed)

        subspace_k = int(min(subspace_top_k, h_base.shape[0], h_base.shape[1], h_ft.shape[0], h_ft.shape[1]))
        base_pca = PCA(n_components=subspace_k, svd_solver="full").fit(h_base)
        ft_pca = PCA(n_components=subspace_k, svd_solver="full").fit(h_ft)
        angles = subspace_angles(base_pca.components_.T, ft_pca.components_.T).astype(np.float32, copy=False)

        delta_stats.append(
            {
                "layer": layer_idx,
                "mean_delta": mean_delta.tolist(),
                "l2_norm_per_example": delta_l2.tolist(),
            }
        )
        pca_results.append(
            {
                "layer": layer_idx,
                "explained_variance_ratio": explained_variance_ratio.tolist(),
                "top_components": components.tolist(),
            }
        )
        delta_alignment.append(
            {
                "layer": layer_idx,
                "mean_pairwise_cosine": _mean_pairwise_cosine(delta_hidden),
            }
        )
        projection_scores.append(
            {
                "layer": layer_idx,
                "scores": proj_scores.tolist(),
            }
        )
        logit_shift.append(
            {
                "layer": layer_idx,
                "l2_norm_per_example": logit_l2.tolist(),
            }
        )
        kl_divergence.append(
            {
                "layer": layer_idx,
                "values": kl_values.tolist(),
            }
        )
        latent_to_logit_corr.append(
            {
                "layer": layer_idx,
                "pearson": latent_to_logit,
            }
        )
        latent_to_behavior.append(
            {
                "layer": layer_idx,
                "auroc": latent_auroc,
            }
        )
        logit_to_behavior.append(
            {
                "layer": layer_idx,
                "auroc": logit_auroc,
            }
        )
        probe_results.append(
            {
                "layer": layer_idx,
                "accuracy": float(probe["accuracy"]),
                "auroc": float(probe["auroc"]),
            }
        )
        subspace_shift.append(
            {
                "layer": layer_idx,
                "principal_angles": angles.tolist(),
            }
        )

    return {
        "delta_stats": delta_stats,
        "pca_results": pca_results,
        "delta_alignment": delta_alignment,
        "projection_scores": projection_scores,
        "logit_shift": logit_shift,
        "kl_divergence": kl_divergence,
        "latent_to_logit_corr": latent_to_logit_corr,
        "latent_to_behavior": latent_to_behavior,
        "logit_to_behavior": logit_to_behavior,
        "probe_results": probe_results,
        "subspace_shift": subspace_shift,
    }

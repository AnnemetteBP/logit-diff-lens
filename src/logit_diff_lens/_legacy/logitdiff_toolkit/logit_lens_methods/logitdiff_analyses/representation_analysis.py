from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


RepresentationSource = Literal["base", "finetuned", "difference"]


@dataclass(frozen=True)
class RepresentationAnalysisExample:
    input_ids: torch.Tensor
    hidden_states_base: Sequence[torch.Tensor]
    hidden_states_ft: Sequence[torch.Tensor]
    label_harmful: bool


def _coerce_representation_example(
    example: RepresentationAnalysisExample | Dict[str, Any]
) -> RepresentationAnalysisExample:
    if isinstance(example, RepresentationAnalysisExample):
        return example
    return RepresentationAnalysisExample(
        input_ids=example["input_ids"],
        hidden_states_base=example["hidden_states_base"],
        hidden_states_ft=example["hidden_states_ft"],
        label_harmful=bool(example["label_harmful"]),
    )


def _validate_hidden_state_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [1, seq, hidden], got {tuple(tensor.shape)}")
    if tensor.shape[0] != 1:
        raise ValueError(f"{name} batch_size must be 1, got {tensor.shape[0]}")
    if tensor.shape[1] < 1:
        raise ValueError(f"{name} seq_len must be >= 1, got {tensor.shape[1]}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or inf values")


def _extract_last_token_representation(hidden_state: torch.Tensor) -> torch.Tensor:
    _validate_hidden_state_tensor("hidden_state", hidden_state)
    return hidden_state[0, -1, :].detach().to(dtype=torch.float32, device="cpu")


def _validate_examples_layer_alignment(
    examples: Sequence[RepresentationAnalysisExample],
) -> int:
    if not examples:
        raise ValueError("examples must be non-empty")

    num_layers = len(examples[0].hidden_states_base)
    if num_layers < 1:
        raise ValueError("hidden_states_base must contain at least one layer")
    if len(examples[0].hidden_states_ft) != num_layers:
        raise ValueError(
            "Layer-count mismatch between base and finetuned hidden states in first example"
        )

    for idx, example in enumerate(examples):
        if len(example.hidden_states_base) != num_layers:
            raise ValueError(
                f"Base layer-count mismatch in example {idx}: "
                f"expected {num_layers}, got {len(example.hidden_states_base)}"
            )
        if len(example.hidden_states_ft) != num_layers:
            raise ValueError(
                f"Finetuned layer-count mismatch in example {idx}: "
                f"expected {num_layers}, got {len(example.hidden_states_ft)}"
            )
        for layer_idx, (base_tensor, ft_tensor) in enumerate(
            zip(example.hidden_states_base, example.hidden_states_ft)
        ):
            _validate_hidden_state_tensor(
                f"hidden_states_base[{idx}][{layer_idx}]",
                base_tensor,
            )
            _validate_hidden_state_tensor(
                f"hidden_states_ft[{idx}][{layer_idx}]",
                ft_tensor,
            )
            if base_tensor.shape[-1] != ft_tensor.shape[-1]:
                raise ValueError(
                    f"Hidden dimension mismatch at example {idx}, layer {layer_idx}: "
                    f"base={base_tensor.shape[-1]}, finetuned={ft_tensor.shape[-1]}"
                )

    return num_layers


def _stack_layer_features(
    examples: Sequence[RepresentationAnalysisExample],
    *,
    layer_idx: int,
    source: RepresentationSource,
) -> np.ndarray:
    features: List[np.ndarray] = []
    for example in examples:
        base_vec = _extract_last_token_representation(example.hidden_states_base[layer_idx])
        ft_vec = _extract_last_token_representation(example.hidden_states_ft[layer_idx])
        if source == "base":
            vec = base_vec
        elif source == "finetuned":
            vec = ft_vec
        elif source == "difference":
            vec = ft_vec - base_vec
        else:
            raise ValueError(f"Unsupported representation source '{source}'")
        features.append(vec.numpy())
    return np.stack(features, axis=0)


def _labels_array(examples: Sequence[RepresentationAnalysisExample]) -> np.ndarray:
    return np.array([int(example.label_harmful) for example in examples], dtype=np.int64)


def _cosine_similarity(base_vec: torch.Tensor, ft_vec: torch.Tensor) -> float:
    base_norm = F.normalize(base_vec.unsqueeze(0), p=2, dim=-1)
    ft_norm = F.normalize(ft_vec.unsqueeze(0), p=2, dim=-1)
    sim = F.cosine_similarity(base_norm, ft_norm, dim=-1)
    if not torch.isfinite(sim).all():
        raise ValueError("Cosine similarity contains NaN or inf values")
    return float(sim.item())


def _mean_cosine_per_layer(examples: Sequence[RepresentationAnalysisExample]) -> List[float]:
    num_layers = _validate_examples_layer_alignment(examples)
    cosine_per_layer: List[float] = []
    for layer_idx in range(num_layers):
        layer_sims: List[float] = []
        for example in examples:
            base_vec = _extract_last_token_representation(example.hidden_states_base[layer_idx])
            ft_vec = _extract_last_token_representation(example.hidden_states_ft[layer_idx])
            layer_sims.append(_cosine_similarity(base_vec, ft_vec))
        cosine_per_layer.append(float(np.mean(layer_sims)))
    return cosine_per_layer


def _center_matrix(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=0, keepdims=True)


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError(f"CKA shape mismatch: x={x.shape}, y={y.shape}")
    x_centered = _center_matrix(x)
    y_centered = _center_matrix(y)
    x_xt = x_centered.T @ x_centered
    y_yt = y_centered.T @ y_centered
    numerator = np.linalg.norm(x_centered.T @ y_centered, ord="fro") ** 2
    denominator = np.linalg.norm(x_xt, ord="fro") * np.linalg.norm(y_yt, ord="fro")
    if denominator <= 0:
        raise ValueError("CKA is undefined when one representation has zero variance")
    value = numerator / denominator
    if not np.isfinite(value):
        raise ValueError("CKA contains NaN or inf values")
    return float(value)


def _cka_per_layer(examples: Sequence[RepresentationAnalysisExample]) -> List[float]:
    num_layers = _validate_examples_layer_alignment(examples)
    cka_values: List[float] = []
    for layer_idx in range(num_layers):
        x = _stack_layer_features(examples, layer_idx=layer_idx, source="base")
        y = _stack_layer_features(examples, layer_idx=layer_idx, source="finetuned")
        cka_values.append(_linear_cka(x, y))
    return cka_values


def _fit_probe_for_layer(
    x_layer: np.ndarray,
    y: np.ndarray,
    *,
    random_seed: int,
) -> Dict[str, float]:
    if x_layer.shape[0] != y.shape[0]:
        raise ValueError(
            f"Feature/label length mismatch: X={x_layer.shape[0]}, y={y.shape[0]}"
        )
    if x_layer.shape[0] < 5:
        raise ValueError("Need at least 5 examples for train/test probing")
    unique_labels = np.unique(y)
    if unique_labels.size != 2:
        raise ValueError(
            f"Probe requires both classes to be present, got labels={unique_labels.tolist()}"
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x_layer,
        y,
        test_size=0.2,
        random_state=int(random_seed),
        stratify=y,
    )
    probe = LogisticRegression(max_iter=1000, random_state=int(random_seed))
    probe.fit(x_train, y_train)

    y_pred = probe.predict(x_test)
    y_score = probe.predict_proba(x_test)[:, 1]
    accuracy = float(accuracy_score(y_test, y_pred))
    auroc = float(roc_auc_score(y_test, y_score))
    return {
        "accuracy": accuracy,
        "auroc": auroc,
    }


def compute_layerwise_linear_probes(
    examples: Sequence[RepresentationAnalysisExample | Dict[str, Any]],
    *,
    representation_source: RepresentationSource = "finetuned",
    random_seed: int = 0,
) -> List[Dict[str, float]]:
    coerced_examples = [_coerce_representation_example(example) for example in examples]
    num_layers = _validate_examples_layer_alignment(coerced_examples)
    y = _labels_array(coerced_examples)

    probe_results: List[Dict[str, float]] = []
    for layer_idx in range(num_layers):
        x_layer = _stack_layer_features(
            coerced_examples,
            layer_idx=layer_idx,
            source=representation_source,
        )
        metrics = _fit_probe_for_layer(x_layer, y, random_seed=random_seed)
        probe_results.append(metrics)
    return probe_results


def analyze_representation_alignment(
    examples: Sequence[RepresentationAnalysisExample | Dict[str, Any]],
    *,
    representation_source: RepresentationSource = "finetuned",
    include_cka: bool = True,
    random_seed: int = 0,
) -> dict:
    coerced_examples = [_coerce_representation_example(example) for example in examples]
    _validate_examples_layer_alignment(coerced_examples)

    probe_results = compute_layerwise_linear_probes(
        coerced_examples,
        representation_source=representation_source,
        random_seed=random_seed,
    )
    cosine_per_layer = _mean_cosine_per_layer(coerced_examples)
    result = {
        "probe_results": probe_results,
        "cosine_per_layer": cosine_per_layer,
    }
    if include_cka:
        result["cka_per_layer"] = _cka_per_layer(coerced_examples)
    return result


def check_random_label_probe_sanity(
    probe_results: Sequence[Dict[str, float]],
    *,
    expected_auroc: float = 0.5,
    tolerance: float = 0.25,
) -> None:
    for layer_idx, metrics in enumerate(probe_results):
        auroc = float(metrics["auroc"])
        if abs(auroc - expected_auroc) > tolerance:
            raise ValueError(
                f"Random-label sanity check failed at layer {layer_idx}: "
                f"expected AUROC near {expected_auroc}, got {auroc}"
            )


def check_base_equals_ft_cosine_sanity(
    cosine_per_layer: Sequence[float],
    *,
    tolerance: float = 1e-4,
) -> None:
    for layer_idx, value in enumerate(cosine_per_layer):
        if abs(float(value) - 1.0) > tolerance:
            raise ValueError(
                f"Base==FT cosine sanity check failed at layer {layer_idx}: "
                f"expected ~1.0, got {value}"
            )


def check_probe_variation_sanity(
    probe_results: Sequence[Dict[str, float]],
    *,
    min_range: float = 1e-6,
) -> None:
    accuracies = [float(metrics["accuracy"]) for metrics in probe_results]
    if max(accuracies) - min(accuracies) <= min_range:
        raise ValueError(
            "Probe accuracy sanity check failed: accuracy is constant across layers"
        )

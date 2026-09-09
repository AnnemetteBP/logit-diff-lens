from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .lens_outputs import BackendMetadata


@dataclass
class ScalarCalibratedSimilarity:
    metric_name: str
    representation_kind: str
    raw_similarity: float
    null_scores: list[float]
    null_mean: float
    null_std: float
    critical_value: float
    p_value: float
    adjusted_p_value: float | None
    null_percentile: float
    calibrated_similarity: float
    sample_count: int
    feature_dim_a: int | None
    feature_dim_b: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScalarCalibratedSimilarity":
        return cls(
            metric_name=str(payload["metric_name"]),
            representation_kind=str(payload["representation_kind"]),
            raw_similarity=float(payload["raw_similarity"]),
            null_scores=[float(v) for v in payload["null_scores"]],
            null_mean=float(payload["null_mean"]),
            null_std=float(payload["null_std"]),
            critical_value=float(payload["critical_value"]),
            p_value=float(payload["p_value"]),
            adjusted_p_value=(
                None if payload.get("adjusted_p_value") is None else float(payload["adjusted_p_value"])
            ),
            null_percentile=float(payload.get("null_percentile", 1.0 - float(payload["p_value"]))),
            calibrated_similarity=float(payload["calibrated_similarity"]),
            sample_count=int(payload["sample_count"]),
            feature_dim_a=None if payload.get("feature_dim_a") is None else int(payload["feature_dim_a"]),
            feature_dim_b=None if payload.get("feature_dim_b") is None else int(payload["feature_dim_b"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "representation_kind": self.representation_kind,
            "raw_similarity": self.raw_similarity,
            "null_scores": self.null_scores,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "critical_value": self.critical_value,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "null_percentile": self.null_percentile,
            "calibrated_similarity": self.calibrated_similarity,
            "sample_count": self.sample_count,
            "feature_dim_a": self.feature_dim_a,
            "feature_dim_b": self.feature_dim_b,
            "metadata": self.metadata,
        }


@dataclass
class LayerPairSimilarityArtifact:
    metric_name: str
    representation_kind: str
    raw_matrix: torch.Tensor
    calibrated_matrix: torch.Tensor
    p_value_matrix: torch.Tensor
    adjusted_p_value_matrix: torch.Tensor | None
    null_percentile_matrix: torch.Tensor | None
    null_mean_matrix: torch.Tensor | None
    layer_indices_a: list[int]
    layer_indices_b: list[int]
    aggregate_statistic_name: str
    aggregate_raw_value: float
    aggregate_null_scores: list[float]
    aggregate_critical_value: float
    aggregate_p_value: float
    aggregate_adjusted_p_value: float | None
    aggregate_null_percentile: float
    aggregate_calibrated_value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LayerPairSimilarityArtifact":
        return cls(
            metric_name=str(payload["metric_name"]),
            representation_kind=str(payload["representation_kind"]),
            raw_matrix=payload["raw_matrix"],
            calibrated_matrix=payload["calibrated_matrix"],
            p_value_matrix=payload["p_value_matrix"],
            adjusted_p_value_matrix=payload.get("adjusted_p_value_matrix"),
            null_percentile_matrix=payload.get("null_percentile_matrix"),
            null_mean_matrix=payload.get("null_mean_matrix"),
            layer_indices_a=[int(v) for v in payload["layer_indices_a"]],
            layer_indices_b=[int(v) for v in payload["layer_indices_b"]],
            aggregate_statistic_name=str(payload["aggregate_statistic_name"]),
            aggregate_raw_value=float(payload["aggregate_raw_value"]),
            aggregate_null_scores=[float(v) for v in payload["aggregate_null_scores"]],
            aggregate_critical_value=float(payload["aggregate_critical_value"]),
            aggregate_p_value=float(payload["aggregate_p_value"]),
            aggregate_adjusted_p_value=(
                None
                if payload.get("aggregate_adjusted_p_value") is None
                else float(payload["aggregate_adjusted_p_value"])
            ),
            aggregate_null_percentile=float(
                payload.get("aggregate_null_percentile", 1.0 - float(payload["aggregate_p_value"]))
            ),
            aggregate_calibrated_value=float(payload["aggregate_calibrated_value"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "representation_kind": self.representation_kind,
            "raw_matrix": self.raw_matrix,
            "calibrated_matrix": self.calibrated_matrix,
            "p_value_matrix": self.p_value_matrix,
            "adjusted_p_value_matrix": self.adjusted_p_value_matrix,
            "null_percentile_matrix": self.null_percentile_matrix,
            "null_mean_matrix": self.null_mean_matrix,
            "layer_indices_a": self.layer_indices_a,
            "layer_indices_b": self.layer_indices_b,
            "aggregate_statistic_name": self.aggregate_statistic_name,
            "aggregate_raw_value": self.aggregate_raw_value,
            "aggregate_null_scores": self.aggregate_null_scores,
            "aggregate_critical_value": self.aggregate_critical_value,
            "aggregate_p_value": self.aggregate_p_value,
            "aggregate_adjusted_p_value": self.aggregate_adjusted_p_value,
            "aggregate_null_percentile": self.aggregate_null_percentile,
            "aggregate_calibrated_value": self.aggregate_calibrated_value,
            "metadata": self.metadata,
        }


@dataclass
class SimilarityRunArtifact:
    side_a_label: str
    side_b_label: str
    artifact_family: str
    prompt_id: str | None
    prompt_text: str | None
    readout_mode: str | None
    alignment_mode: str
    backend_metadata_a: BackendMetadata | None
    backend_metadata_b: BackendMetadata | None
    scalar_results: list[ScalarCalibratedSimilarity]
    matrix_results: list[LayerPairSimilarityArtifact]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SimilarityRunArtifact":
        backend_a = payload.get("backend_metadata_a")
        backend_b = payload.get("backend_metadata_b")
        return cls(
            side_a_label=str(payload["side_a_label"]),
            side_b_label=str(payload["side_b_label"]),
            artifact_family=str(payload["artifact_family"]),
            prompt_id=payload.get("prompt_id"),
            prompt_text=payload.get("prompt_text"),
            readout_mode=payload.get("readout_mode"),
            alignment_mode=str(payload["alignment_mode"]),
            backend_metadata_a=None if backend_a is None else BackendMetadata.from_dict(backend_a),
            backend_metadata_b=None if backend_b is None else BackendMetadata.from_dict(backend_b),
            scalar_results=[ScalarCalibratedSimilarity.from_dict(item) for item in payload.get("scalar_results", [])],
            matrix_results=[LayerPairSimilarityArtifact.from_dict(item) for item in payload.get("matrix_results", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side_a_label": self.side_a_label,
            "side_b_label": self.side_b_label,
            "artifact_family": self.artifact_family,
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "readout_mode": self.readout_mode,
            "alignment_mode": self.alignment_mode,
            "backend_metadata_a": None if self.backend_metadata_a is None else self.backend_metadata_a.to_dict(),
            "backend_metadata_b": None if self.backend_metadata_b is None else self.backend_metadata_b.to_dict(),
            "scalar_results": [item.to_dict() for item in self.scalar_results],
            "matrix_results": [item.to_dict() for item in self.matrix_results],
            "metadata": self.metadata,
        }


__all__ = [
    "LayerPairSimilarityArtifact",
    "ScalarCalibratedSimilarity",
    "SimilarityRunArtifact",
]

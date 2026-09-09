from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .lens_outputs import BackendMetadata


@dataclass
class LayerPairCorrelationArtifact:
    metric_name: str
    readout_mode: str
    pearson_r_matrix: torch.Tensor
    pearson_p_matrix: torch.Tensor
    pearson_q_matrix: torch.Tensor
    pearson_ci_low_matrix: torch.Tensor
    pearson_ci_high_matrix: torch.Tensor
    spearman_r_matrix: torch.Tensor
    spearman_p_matrix: torch.Tensor
    spearman_q_matrix: torch.Tensor
    spearman_ci_low_matrix: torch.Tensor
    spearman_ci_high_matrix: torch.Tensor
    sample_count_matrix: torch.Tensor
    layer_indices_a: list[int]
    layer_indices_b: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LayerPairCorrelationArtifact":
        return cls(
            metric_name=str(payload["metric_name"]),
            readout_mode=str(payload["readout_mode"]),
            pearson_r_matrix=payload["pearson_r_matrix"],
            pearson_p_matrix=payload["pearson_p_matrix"],
            pearson_q_matrix=payload["pearson_q_matrix"],
            pearson_ci_low_matrix=payload["pearson_ci_low_matrix"],
            pearson_ci_high_matrix=payload["pearson_ci_high_matrix"],
            spearman_r_matrix=payload["spearman_r_matrix"],
            spearman_p_matrix=payload["spearman_p_matrix"],
            spearman_q_matrix=payload["spearman_q_matrix"],
            spearman_ci_low_matrix=payload["spearman_ci_low_matrix"],
            spearman_ci_high_matrix=payload["spearman_ci_high_matrix"],
            sample_count_matrix=payload["sample_count_matrix"],
            layer_indices_a=[int(v) for v in payload["layer_indices_a"]],
            layer_indices_b=[int(v) for v in payload["layer_indices_b"]],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "readout_mode": self.readout_mode,
            "pearson_r_matrix": self.pearson_r_matrix,
            "pearson_p_matrix": self.pearson_p_matrix,
            "pearson_q_matrix": self.pearson_q_matrix,
            "pearson_ci_low_matrix": self.pearson_ci_low_matrix,
            "pearson_ci_high_matrix": self.pearson_ci_high_matrix,
            "spearman_r_matrix": self.spearman_r_matrix,
            "spearman_p_matrix": self.spearman_p_matrix,
            "spearman_q_matrix": self.spearman_q_matrix,
            "spearman_ci_low_matrix": self.spearman_ci_low_matrix,
            "spearman_ci_high_matrix": self.spearman_ci_high_matrix,
            "sample_count_matrix": self.sample_count_matrix,
            "layer_indices_a": self.layer_indices_a,
            "layer_indices_b": self.layer_indices_b,
            "metadata": self.metadata,
        }


@dataclass
class CorrelationRunArtifact:
    side_a_label: str
    side_b_label: str
    artifact_family: str
    readout_mode: str
    alignment_mode: str
    backend_metadata_a: BackendMetadata | None
    backend_metadata_b: BackendMetadata | None
    matrix_results: list[LayerPairCorrelationArtifact]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CorrelationRunArtifact":
        backend_a = payload.get("backend_metadata_a")
        backend_b = payload.get("backend_metadata_b")
        return cls(
            side_a_label=str(payload["side_a_label"]),
            side_b_label=str(payload["side_b_label"]),
            artifact_family=str(payload["artifact_family"]),
            readout_mode=str(payload["readout_mode"]),
            alignment_mode=str(payload["alignment_mode"]),
            backend_metadata_a=None if backend_a is None else BackendMetadata.from_dict(backend_a),
            backend_metadata_b=None if backend_b is None else BackendMetadata.from_dict(backend_b),
            matrix_results=[LayerPairCorrelationArtifact.from_dict(item) for item in payload.get("matrix_results", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side_a_label": self.side_a_label,
            "side_b_label": self.side_b_label,
            "artifact_family": self.artifact_family,
            "readout_mode": self.readout_mode,
            "alignment_mode": self.alignment_mode,
            "backend_metadata_a": None if self.backend_metadata_a is None else self.backend_metadata_a.to_dict(),
            "backend_metadata_b": None if self.backend_metadata_b is None else self.backend_metadata_b.to_dict(),
            "matrix_results": [item.to_dict() for item in self.matrix_results],
            "metadata": self.metadata,
        }


__all__ = [
    "CorrelationRunArtifact",
    "LayerPairCorrelationArtifact",
]

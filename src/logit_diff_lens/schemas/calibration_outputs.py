from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .lens_outputs import BackendMetadata


@dataclass
class CalibrationMatrixResult:
    metric_name: str
    axis_kind: str
    value_matrix: torch.Tensor
    sample_count_matrix: torch.Tensor
    confidence_mean_matrix: torch.Tensor | None = None
    accuracy_mean_matrix: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationMatrixResult":
        return cls(
            metric_name=str(payload["metric_name"]),
            axis_kind=str(payload["axis_kind"]),
            value_matrix=payload["value_matrix"],
            sample_count_matrix=payload["sample_count_matrix"],
            confidence_mean_matrix=payload.get("confidence_mean_matrix"),
            accuracy_mean_matrix=payload.get("accuracy_mean_matrix"),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "axis_kind": self.axis_kind,
            "value_matrix": self.value_matrix,
            "sample_count_matrix": self.sample_count_matrix,
            "confidence_mean_matrix": self.confidence_mean_matrix,
            "accuracy_mean_matrix": self.accuracy_mean_matrix,
            "metadata": self.metadata,
        }


@dataclass
class CalibrationSummaryResult:
    metric_name: str
    axis_kind: str
    axis_indices: list[int]
    values: torch.Tensor
    ci_low: torch.Tensor | None
    ci_high: torch.Tensor | None
    sample_counts: torch.Tensor
    confidence_means: torch.Tensor | None = None
    accuracy_means: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationSummaryResult":
        return cls(
            metric_name=str(payload["metric_name"]),
            axis_kind=str(payload["axis_kind"]),
            axis_indices=[int(v) for v in payload["axis_indices"]],
            values=payload["values"],
            ci_low=payload.get("ci_low"),
            ci_high=payload.get("ci_high"),
            sample_counts=payload["sample_counts"],
            confidence_means=payload.get("confidence_means"),
            accuracy_means=payload.get("accuracy_means"),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "axis_kind": self.axis_kind,
            "axis_indices": self.axis_indices,
            "values": self.values,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "sample_counts": self.sample_counts,
            "confidence_means": self.confidence_means,
            "accuracy_means": self.accuracy_means,
            "metadata": self.metadata,
        }


@dataclass
class CalibrationRunArtifact:
    side_a_label: str
    side_b_label: str
    artifact_family: str
    reference_kind: str
    reference_source: str
    readout_mode_eval: str
    readout_mode_reference: str
    alignment_mode: str
    layer_indices: list[int]
    position_indices: list[int]
    backend_metadata_a: BackendMetadata | None
    backend_metadata_b: BackendMetadata | None
    matrix_results: list[CalibrationMatrixResult]
    summary_results: list[CalibrationSummaryResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationRunArtifact":
        backend_a = payload.get("backend_metadata_a")
        backend_b = payload.get("backend_metadata_b")
        return cls(
            side_a_label=str(payload["side_a_label"]),
            side_b_label=str(payload["side_b_label"]),
            artifact_family=str(payload["artifact_family"]),
            reference_kind=str(payload["reference_kind"]),
            reference_source=str(payload["reference_source"]),
            readout_mode_eval=str(payload["readout_mode_eval"]),
            readout_mode_reference=str(payload["readout_mode_reference"]),
            alignment_mode=str(payload["alignment_mode"]),
            layer_indices=[int(v) for v in payload["layer_indices"]],
            position_indices=[int(v) for v in payload["position_indices"]],
            backend_metadata_a=None if backend_a is None else BackendMetadata.from_dict(backend_a),
            backend_metadata_b=None if backend_b is None else BackendMetadata.from_dict(backend_b),
            matrix_results=[CalibrationMatrixResult.from_dict(item) for item in payload.get("matrix_results", [])],
            summary_results=[CalibrationSummaryResult.from_dict(item) for item in payload.get("summary_results", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side_a_label": self.side_a_label,
            "side_b_label": self.side_b_label,
            "artifact_family": self.artifact_family,
            "reference_kind": self.reference_kind,
            "reference_source": self.reference_source,
            "readout_mode_eval": self.readout_mode_eval,
            "readout_mode_reference": self.readout_mode_reference,
            "alignment_mode": self.alignment_mode,
            "layer_indices": self.layer_indices,
            "position_indices": self.position_indices,
            "backend_metadata_a": None if self.backend_metadata_a is None else self.backend_metadata_a.to_dict(),
            "backend_metadata_b": None if self.backend_metadata_b is None else self.backend_metadata_b.to_dict(),
            "matrix_results": [item.to_dict() for item in self.matrix_results],
            "summary_results": [item.to_dict() for item in self.summary_results],
            "metadata": self.metadata,
        }


__all__ = [
    "CalibrationMatrixResult",
    "CalibrationRunArtifact",
    "CalibrationSummaryResult",
]

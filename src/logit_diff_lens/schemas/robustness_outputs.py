from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .lens_outputs import BackendMetadata


@dataclass
class RobustnessProfileArtifact:
    metric_name: str
    lens_family: str
    regime_name: str
    layer_indices: list[int]
    prompt_values: torch.Tensor
    mean_values: torch.Tensor
    ci_low: torch.Tensor | None
    ci_high: torch.Tensor | None
    sample_counts: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RobustnessProfileArtifact":
        return cls(
            metric_name=str(payload["metric_name"]),
            lens_family=str(payload["lens_family"]),
            regime_name=str(payload["regime_name"]),
            layer_indices=[int(v) for v in payload["layer_indices"]],
            prompt_values=payload["prompt_values"],
            mean_values=payload["mean_values"],
            ci_low=payload.get("ci_low"),
            ci_high=payload.get("ci_high"),
            sample_counts=payload["sample_counts"],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "lens_family": self.lens_family,
            "regime_name": self.regime_name,
            "layer_indices": self.layer_indices,
            "prompt_values": self.prompt_values,
            "mean_values": self.mean_values,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "sample_counts": self.sample_counts,
            "metadata": self.metadata,
        }


@dataclass
class RobustnessAgreementArtifact:
    metric_name: str
    regime_name: str
    lens_family_a: str
    lens_family_b: str
    layer_indices: list[int]
    pearson_r: torch.Tensor
    pearson_p: torch.Tensor
    spearman_r: torch.Tensor
    spearman_p: torch.Tensor
    sample_counts: torch.Tensor
    aggregate_pearson_mean: float
    aggregate_spearman_mean: float
    mean_gap: float
    gap_ci_low: float | None
    gap_ci_high: float | None
    peak_gap_layer_index: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RobustnessAgreementArtifact":
        return cls(
            metric_name=str(payload["metric_name"]),
            regime_name=str(payload["regime_name"]),
            lens_family_a=str(payload["lens_family_a"]),
            lens_family_b=str(payload["lens_family_b"]),
            layer_indices=[int(v) for v in payload["layer_indices"]],
            pearson_r=payload["pearson_r"],
            pearson_p=payload["pearson_p"],
            spearman_r=payload["spearman_r"],
            spearman_p=payload["spearman_p"],
            sample_counts=payload["sample_counts"],
            aggregate_pearson_mean=float(payload["aggregate_pearson_mean"]),
            aggregate_spearman_mean=float(payload["aggregate_spearman_mean"]),
            mean_gap=float(payload["mean_gap"]),
            gap_ci_low=None if payload.get("gap_ci_low") is None else float(payload["gap_ci_low"]),
            gap_ci_high=None if payload.get("gap_ci_high") is None else float(payload["gap_ci_high"]),
            peak_gap_layer_index=(
                None if payload.get("peak_gap_layer_index") is None else int(payload["peak_gap_layer_index"])
            ),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "regime_name": self.regime_name,
            "lens_family_a": self.lens_family_a,
            "lens_family_b": self.lens_family_b,
            "layer_indices": self.layer_indices,
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "spearman_r": self.spearman_r,
            "spearman_p": self.spearman_p,
            "sample_counts": self.sample_counts,
            "aggregate_pearson_mean": self.aggregate_pearson_mean,
            "aggregate_spearman_mean": self.aggregate_spearman_mean,
            "mean_gap": self.mean_gap,
            "gap_ci_low": self.gap_ci_low,
            "gap_ci_high": self.gap_ci_high,
            "peak_gap_layer_index": self.peak_gap_layer_index,
            "metadata": self.metadata,
        }


@dataclass
class RobustnessCalibrationArtifact:
    metric_name: str
    event_name: str
    lens_family: str
    regime_name: str
    layer_indices: list[int]
    ece_values: torch.Tensor
    brier_values: torch.Tensor
    sample_counts: torch.Tensor
    confidence_means: torch.Tensor
    outcome_means: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RobustnessCalibrationArtifact":
        return cls(
            metric_name=str(payload["metric_name"]),
            event_name=str(payload["event_name"]),
            lens_family=str(payload["lens_family"]),
            regime_name=str(payload["regime_name"]),
            layer_indices=[int(v) for v in payload["layer_indices"]],
            ece_values=payload["ece_values"],
            brier_values=payload["brier_values"],
            sample_counts=payload["sample_counts"],
            confidence_means=payload["confidence_means"],
            outcome_means=payload["outcome_means"],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "event_name": self.event_name,
            "lens_family": self.lens_family,
            "regime_name": self.regime_name,
            "layer_indices": self.layer_indices,
            "ece_values": self.ece_values,
            "brier_values": self.brier_values,
            "sample_counts": self.sample_counts,
            "confidence_means": self.confidence_means,
            "outcome_means": self.outcome_means,
            "metadata": self.metadata,
        }


@dataclass
class RobustnessRunArtifact:
    side_a_label: str
    side_b_label: str
    artifact_family: str
    metric_name: str
    alignment_mode: str
    backend_metadata_a: BackendMetadata | None
    backend_metadata_b: BackendMetadata | None
    profiles: list[RobustnessProfileArtifact]
    within_lens_results: list[RobustnessAgreementArtifact]
    cross_lens_results: list[RobustnessAgreementArtifact]
    agreement_calibration_results: list[RobustnessCalibrationArtifact]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RobustnessRunArtifact":
        backend_a = payload.get("backend_metadata_a")
        backend_b = payload.get("backend_metadata_b")
        return cls(
            side_a_label=str(payload["side_a_label"]),
            side_b_label=str(payload["side_b_label"]),
            artifact_family=str(payload["artifact_family"]),
            metric_name=str(payload["metric_name"]),
            alignment_mode=str(payload["alignment_mode"]),
            backend_metadata_a=None if backend_a is None else BackendMetadata.from_dict(backend_a),
            backend_metadata_b=None if backend_b is None else BackendMetadata.from_dict(backend_b),
            profiles=[RobustnessProfileArtifact.from_dict(item) for item in payload.get("profiles", [])],
            within_lens_results=[
                RobustnessAgreementArtifact.from_dict(item) for item in payload.get("within_lens_results", [])
            ],
            cross_lens_results=[
                RobustnessAgreementArtifact.from_dict(item) for item in payload.get("cross_lens_results", [])
            ],
            agreement_calibration_results=[
                RobustnessCalibrationArtifact.from_dict(item)
                for item in payload.get("agreement_calibration_results", [])
            ],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side_a_label": self.side_a_label,
            "side_b_label": self.side_b_label,
            "artifact_family": self.artifact_family,
            "metric_name": self.metric_name,
            "alignment_mode": self.alignment_mode,
            "backend_metadata_a": None if self.backend_metadata_a is None else self.backend_metadata_a.to_dict(),
            "backend_metadata_b": None if self.backend_metadata_b is None else self.backend_metadata_b.to_dict(),
            "profiles": [item.to_dict() for item in self.profiles],
            "within_lens_results": [item.to_dict() for item in self.within_lens_results],
            "cross_lens_results": [item.to_dict() for item in self.cross_lens_results],
            "agreement_calibration_results": [item.to_dict() for item in self.agreement_calibration_results],
            "metadata": self.metadata,
        }


__all__ = [
    "RobustnessAgreementArtifact",
    "RobustnessCalibrationArtifact",
    "RobustnessProfileArtifact",
    "RobustnessRunArtifact",
]

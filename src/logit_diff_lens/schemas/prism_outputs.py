from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .lens_outputs import BackendMetadata


@dataclass
class PromptPrismLayerResult:
    """Legacy compatibility shell for earlier prompt prism payloads."""

    layer_index: int
    layer_name: str
    residual_norms: torch.Tensor
    attention_norms: torch.Tensor
    mlp_norms: torch.Tensor
    cumulative_norms: torch.Tensor
    reconstruction_error: torch.Tensor
    residual_top_token_ids: torch.Tensor
    attention_top_token_ids: torch.Tensor
    mlp_top_token_ids: torch.Tensor
    cumulative_top_token_ids: torch.Tensor
    residual_top_values: torch.Tensor
    attention_top_values: torch.Tensor
    mlp_top_values: torch.Tensor
    cumulative_top_values: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptPrismLayerResult":
        return cls(
            layer_index=int(payload["layer_index"]),
            layer_name=str(payload["layer_name"]),
            residual_norms=payload["residual_norms"],
            attention_norms=payload["attention_norms"],
            mlp_norms=payload["mlp_norms"],
            cumulative_norms=payload["cumulative_norms"],
            reconstruction_error=payload["reconstruction_error"],
            residual_top_token_ids=payload["residual_top_token_ids"],
            attention_top_token_ids=payload["attention_top_token_ids"],
            mlp_top_token_ids=payload["mlp_top_token_ids"],
            cumulative_top_token_ids=payload["cumulative_top_token_ids"],
            residual_top_values=payload["residual_top_values"],
            attention_top_values=payload["attention_top_values"],
            mlp_top_values=payload["mlp_top_values"],
            cumulative_top_values=payload["cumulative_top_values"],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "residual_norms": self.residual_norms,
            "attention_norms": self.attention_norms,
            "mlp_norms": self.mlp_norms,
            "cumulative_norms": self.cumulative_norms,
            "reconstruction_error": self.reconstruction_error,
            "residual_top_token_ids": self.residual_top_token_ids,
            "attention_top_token_ids": self.attention_top_token_ids,
            "mlp_top_token_ids": self.mlp_top_token_ids,
            "cumulative_top_token_ids": self.cumulative_top_token_ids,
            "residual_top_values": self.residual_top_values,
            "attention_top_values": self.attention_top_values,
            "mlp_top_values": self.mlp_top_values,
            "cumulative_top_values": self.cumulative_top_values,
            "metadata": self.metadata,
        }


@dataclass
class PromptPrismArtifact:
    prompt_text: str
    prompt_formatted: str
    prompt_id: str | None
    readout_mode: str
    top_k: int
    token_selection: str
    position_indices: list[int]
    token_ids: torch.Tensor
    token_text: list[str]
    component_labels: list[str]
    selected_token_ids: torch.Tensor
    selected_token_text: list[str]
    contribution_tensor: torch.Tensor
    output_token_logits: torch.Tensor
    additive_component_logits: torch.Tensor
    cumulative_full_layer_logits: torch.Tensor
    output_residual_gap: torch.Tensor
    backend_metadata: BackendMetadata | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptPrismArtifact":
        backend = payload.get("backend_metadata")
        return cls(
            prompt_text=str(payload["prompt_text"]),
            prompt_formatted=str(payload["prompt_formatted"]),
            prompt_id=payload.get("prompt_id"),
            readout_mode=str(payload["readout_mode"]),
            top_k=int(payload["top_k"]),
            token_selection=str(payload.get("token_selection", "largest_positive_logit")),
            position_indices=[int(v) for v in payload["position_indices"]],
            token_ids=payload["token_ids"],
            token_text=list(payload["token_text"]),
            component_labels=list(payload["component_labels"]),
            selected_token_ids=payload["selected_token_ids"],
            selected_token_text=list(payload["selected_token_text"]),
            contribution_tensor=payload["contribution_tensor"],
            output_token_logits=payload["output_token_logits"],
            additive_component_logits=payload["additive_component_logits"],
            cumulative_full_layer_logits=payload["cumulative_full_layer_logits"],
            output_residual_gap=payload["output_residual_gap"],
            backend_metadata=None if backend is None else BackendMetadata.from_dict(backend),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "prompt_formatted": self.prompt_formatted,
            "prompt_id": self.prompt_id,
            "readout_mode": self.readout_mode,
            "top_k": self.top_k,
            "token_selection": self.token_selection,
            "position_indices": self.position_indices,
            "token_ids": self.token_ids,
            "token_text": self.token_text,
            "component_labels": self.component_labels,
            "selected_token_ids": self.selected_token_ids,
            "selected_token_text": self.selected_token_text,
            "contribution_tensor": self.contribution_tensor,
            "output_token_logits": self.output_token_logits,
            "additive_component_logits": self.additive_component_logits,
            "cumulative_full_layer_logits": self.cumulative_full_layer_logits,
            "output_residual_gap": self.output_residual_gap,
            "backend_metadata": None if self.backend_metadata is None else self.backend_metadata.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class PromptDiffPrismLayerResult:
    layer_index: int
    layer_name: str
    delta_residual_norms: torch.Tensor
    delta_attention_norms: torch.Tensor
    delta_mlp_norms: torch.Tensor
    delta_cumulative_norms: torch.Tensor
    delta_reconstruction_error: torch.Tensor
    delta_residual_top_token_ids: torch.Tensor
    delta_attention_top_token_ids: torch.Tensor
    delta_mlp_top_token_ids: torch.Tensor
    delta_cumulative_top_token_ids: torch.Tensor
    delta_residual_top_values: torch.Tensor
    delta_attention_top_values: torch.Tensor
    delta_mlp_top_values: torch.Tensor
    delta_cumulative_top_values: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptDiffPrismLayerResult":
        return cls(
            layer_index=int(payload["layer_index"]),
            layer_name=str(payload["layer_name"]),
            delta_residual_norms=payload["delta_residual_norms"],
            delta_attention_norms=payload["delta_attention_norms"],
            delta_mlp_norms=payload["delta_mlp_norms"],
            delta_cumulative_norms=payload["delta_cumulative_norms"],
            delta_reconstruction_error=payload["delta_reconstruction_error"],
            delta_residual_top_token_ids=payload["delta_residual_top_token_ids"],
            delta_attention_top_token_ids=payload["delta_attention_top_token_ids"],
            delta_mlp_top_token_ids=payload["delta_mlp_top_token_ids"],
            delta_cumulative_top_token_ids=payload["delta_cumulative_top_token_ids"],
            delta_residual_top_values=payload["delta_residual_top_values"],
            delta_attention_top_values=payload["delta_attention_top_values"],
            delta_mlp_top_values=payload["delta_mlp_top_values"],
            delta_cumulative_top_values=payload["delta_cumulative_top_values"],
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "delta_residual_norms": self.delta_residual_norms,
            "delta_attention_norms": self.delta_attention_norms,
            "delta_mlp_norms": self.delta_mlp_norms,
            "delta_cumulative_norms": self.delta_cumulative_norms,
            "delta_reconstruction_error": self.delta_reconstruction_error,
            "delta_residual_top_token_ids": self.delta_residual_top_token_ids,
            "delta_attention_top_token_ids": self.delta_attention_top_token_ids,
            "delta_mlp_top_token_ids": self.delta_mlp_top_token_ids,
            "delta_cumulative_top_token_ids": self.delta_cumulative_top_token_ids,
            "delta_residual_top_values": self.delta_residual_top_values,
            "delta_attention_top_values": self.delta_attention_top_values,
            "delta_mlp_top_values": self.delta_mlp_top_values,
            "delta_cumulative_top_values": self.delta_cumulative_top_values,
            "metadata": self.metadata,
        }


@dataclass
class PromptDiffPrismArtifact:
    prompt_text: str
    prompt_formatted: str
    prompt_id: str | None
    readout_mode: str
    top_k: int
    side_ft_label: str
    side_base_label: str
    delta_definition: str
    position_indices: list[int]
    token_ids: torch.Tensor
    token_text: list[str]
    delta_embedding_norms: torch.Tensor
    delta_embedding_top_token_ids: torch.Tensor
    delta_embedding_top_values: torch.Tensor
    layer_results: list[PromptDiffPrismLayerResult]
    backend_metadata_ft: BackendMetadata | None
    backend_metadata_base: BackendMetadata | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptDiffPrismArtifact":
        backend_ft = payload.get("backend_metadata_ft")
        backend_base = payload.get("backend_metadata_base")
        return cls(
            prompt_text=str(payload["prompt_text"]),
            prompt_formatted=str(payload["prompt_formatted"]),
            prompt_id=payload.get("prompt_id"),
            readout_mode=str(payload["readout_mode"]),
            top_k=int(payload["top_k"]),
            side_ft_label=str(payload["side_ft_label"]),
            side_base_label=str(payload["side_base_label"]),
            delta_definition=str(payload["delta_definition"]),
            position_indices=[int(v) for v in payload["position_indices"]],
            token_ids=payload["token_ids"],
            token_text=list(payload["token_text"]),
            delta_embedding_norms=payload["delta_embedding_norms"],
            delta_embedding_top_token_ids=payload["delta_embedding_top_token_ids"],
            delta_embedding_top_values=payload["delta_embedding_top_values"],
            layer_results=[PromptDiffPrismLayerResult.from_dict(item) for item in payload.get("layer_results", [])],
            backend_metadata_ft=None if backend_ft is None else BackendMetadata.from_dict(backend_ft),
            backend_metadata_base=None if backend_base is None else BackendMetadata.from_dict(backend_base),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "prompt_formatted": self.prompt_formatted,
            "prompt_id": self.prompt_id,
            "readout_mode": self.readout_mode,
            "top_k": self.top_k,
            "side_ft_label": self.side_ft_label,
            "side_base_label": self.side_base_label,
            "delta_definition": self.delta_definition,
            "position_indices": self.position_indices,
            "token_ids": self.token_ids,
            "token_text": self.token_text,
            "delta_embedding_norms": self.delta_embedding_norms,
            "delta_embedding_top_token_ids": self.delta_embedding_top_token_ids,
            "delta_embedding_top_values": self.delta_embedding_top_values,
            "layer_results": [item.to_dict() for item in self.layer_results],
            "backend_metadata_ft": None if self.backend_metadata_ft is None else self.backend_metadata_ft.to_dict(),
            "backend_metadata_base": None if self.backend_metadata_base is None else self.backend_metadata_base.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class PromptDiffPrismHeatmapArtifact:
    prompt_text: str
    prompt_formatted: str
    prompt_id: str | None
    readout_mode: str
    position_index: int
    predictor_token_id: int
    predictor_token_text: str
    target_token_id: int | None
    target_token_text: str | None
    side_ft_label: str
    side_base_label: str
    delta_definition: str
    token_selection: str
    top_k: int
    component_labels: list[str]
    selected_token_ids: torch.Tensor
    selected_token_text: list[str]
    contribution_matrix: torch.Tensor
    total_delta_logits: torch.Tensor
    summed_component_logits: torch.Tensor
    reconstruction_gap: torch.Tensor
    backend_metadata_ft: BackendMetadata | None
    backend_metadata_base: BackendMetadata | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptDiffPrismHeatmapArtifact":
        backend_ft = payload.get("backend_metadata_ft")
        backend_base = payload.get("backend_metadata_base")
        return cls(
            prompt_text=str(payload["prompt_text"]),
            prompt_formatted=str(payload["prompt_formatted"]),
            prompt_id=payload.get("prompt_id"),
            readout_mode=str(payload["readout_mode"]),
            position_index=int(payload["position_index"]),
            predictor_token_id=int(payload["predictor_token_id"]),
            predictor_token_text=str(payload["predictor_token_text"]),
            target_token_id=None if payload.get("target_token_id") is None else int(payload["target_token_id"]),
            target_token_text=payload.get("target_token_text"),
            side_ft_label=str(payload["side_ft_label"]),
            side_base_label=str(payload["side_base_label"]),
            delta_definition=str(payload["delta_definition"]),
            token_selection=str(payload["token_selection"]),
            top_k=int(payload["top_k"]),
            component_labels=list(payload["component_labels"]),
            selected_token_ids=payload["selected_token_ids"],
            selected_token_text=list(payload["selected_token_text"]),
            contribution_matrix=payload["contribution_matrix"],
            total_delta_logits=payload["total_delta_logits"],
            summed_component_logits=payload["summed_component_logits"],
            reconstruction_gap=payload["reconstruction_gap"],
            backend_metadata_ft=None if backend_ft is None else BackendMetadata.from_dict(backend_ft),
            backend_metadata_base=None if backend_base is None else BackendMetadata.from_dict(backend_base),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "prompt_formatted": self.prompt_formatted,
            "prompt_id": self.prompt_id,
            "readout_mode": self.readout_mode,
            "position_index": self.position_index,
            "predictor_token_id": self.predictor_token_id,
            "predictor_token_text": self.predictor_token_text,
            "target_token_id": self.target_token_id,
            "target_token_text": self.target_token_text,
            "side_ft_label": self.side_ft_label,
            "side_base_label": self.side_base_label,
            "delta_definition": self.delta_definition,
            "token_selection": self.token_selection,
            "top_k": self.top_k,
            "component_labels": self.component_labels,
            "selected_token_ids": self.selected_token_ids,
            "selected_token_text": self.selected_token_text,
            "contribution_matrix": self.contribution_matrix,
            "total_delta_logits": self.total_delta_logits,
            "summed_component_logits": self.summed_component_logits,
            "reconstruction_gap": self.reconstruction_gap,
            "backend_metadata_ft": None if self.backend_metadata_ft is None else self.backend_metadata_ft.to_dict(),
            "backend_metadata_base": None if self.backend_metadata_base is None else self.backend_metadata_base.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class PromptDiffPrismSummaryArtifact:
    prompt_text: str
    prompt_formatted: str
    prompt_id: str | None
    readout_mode: str
    position_index: int
    side_ft_label: str
    side_base_label: str
    delta_definition: str
    summary_kind: str
    component_labels: list[str]
    raw_component_scores: torch.Tensor
    signed_component_means: torch.Tensor
    selected_token_ids: torch.Tensor
    selected_token_text: list[str]
    calibrated_component_scores: torch.Tensor | None
    p_value_vector: torch.Tensor | None
    adjusted_p_value_vector: torch.Tensor | None
    critical_value_vector: torch.Tensor | None
    null_mean_vector: torch.Tensor | None
    null_std_vector: torch.Tensor | None
    backend_metadata_ft: BackendMetadata | None
    backend_metadata_base: BackendMetadata | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptDiffPrismSummaryArtifact":
        backend_ft = payload.get("backend_metadata_ft")
        backend_base = payload.get("backend_metadata_base")
        return cls(
            prompt_text=str(payload["prompt_text"]),
            prompt_formatted=str(payload["prompt_formatted"]),
            prompt_id=payload.get("prompt_id"),
            readout_mode=str(payload["readout_mode"]),
            position_index=int(payload["position_index"]),
            side_ft_label=str(payload["side_ft_label"]),
            side_base_label=str(payload["side_base_label"]),
            delta_definition=str(payload["delta_definition"]),
            summary_kind=str(payload["summary_kind"]),
            component_labels=list(payload["component_labels"]),
            raw_component_scores=payload["raw_component_scores"],
            signed_component_means=payload["signed_component_means"],
            selected_token_ids=payload["selected_token_ids"],
            selected_token_text=list(payload["selected_token_text"]),
            calibrated_component_scores=payload.get("calibrated_component_scores"),
            p_value_vector=payload.get("p_value_vector"),
            adjusted_p_value_vector=payload.get("adjusted_p_value_vector"),
            critical_value_vector=payload.get("critical_value_vector"),
            null_mean_vector=payload.get("null_mean_vector"),
            null_std_vector=payload.get("null_std_vector"),
            backend_metadata_ft=None if backend_ft is None else BackendMetadata.from_dict(backend_ft),
            backend_metadata_base=None if backend_base is None else BackendMetadata.from_dict(backend_base),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "prompt_formatted": self.prompt_formatted,
            "prompt_id": self.prompt_id,
            "readout_mode": self.readout_mode,
            "position_index": self.position_index,
            "side_ft_label": self.side_ft_label,
            "side_base_label": self.side_base_label,
            "delta_definition": self.delta_definition,
            "summary_kind": self.summary_kind,
            "component_labels": self.component_labels,
            "raw_component_scores": self.raw_component_scores,
            "signed_component_means": self.signed_component_means,
            "selected_token_ids": self.selected_token_ids,
            "selected_token_text": self.selected_token_text,
            "calibrated_component_scores": self.calibrated_component_scores,
            "p_value_vector": self.p_value_vector,
            "adjusted_p_value_vector": self.adjusted_p_value_vector,
            "critical_value_vector": self.critical_value_vector,
            "null_mean_vector": self.null_mean_vector,
            "null_std_vector": self.null_std_vector,
            "backend_metadata_ft": None if self.backend_metadata_ft is None else self.backend_metadata_ft.to_dict(),
            "backend_metadata_base": None if self.backend_metadata_base is None else self.backend_metadata_base.to_dict(),
            "metadata": self.metadata,
        }


__all__ = [
    "PromptDiffPrismArtifact",
    "PromptDiffPrismHeatmapArtifact",
    "PromptDiffPrismSummaryArtifact",
    "PromptDiffPrismLayerResult",
    "PromptPrismArtifact",
    "PromptPrismLayerResult",
]

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch


OperandOrder = Literal["ft_minus_base", "comparison_minus_reference", "none"]
LensReadoutMode = Literal["raw", "model_norm", "bias_only", "tuned"]
PromptHiddenMode = Literal["raw", "model_norm"]
PromptReadoutMode = Literal["raw", "model_norm", "tuned"]


@dataclass(frozen=True)
class BackendMetadata:
    """Runtime metadata required to reproduce collection and decode behavior."""

    model_backend: str
    activation_backend: str
    decode_backend: str
    device_policy: str
    dtype_compute: str
    dtype_storage: str
    quantization: str
    device_map: str
    model_id: str | None = None
    tokenizer_id: str | None = None
    model_revision: str | None = None
    wrapper_name: str | None = None
    architecture: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BackendMetadata":
        return cls(
            model_backend=payload["model_backend"],
            activation_backend=payload["activation_backend"],
            decode_backend=payload["decode_backend"],
            device_policy=payload["device_policy"],
            dtype_compute=payload["dtype_compute"],
            dtype_storage=payload["dtype_storage"],
            quantization=payload["quantization"],
            device_map=payload["device_map"],
            model_id=payload.get("model_id"),
            tokenizer_id=payload.get("tokenizer_id"),
            model_revision=payload.get("model_revision"),
            wrapper_name=payload.get("wrapper_name"),
            architecture=payload.get("architecture"),
            extra=dict(payload.get("extra", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_backend": self.model_backend,
            "activation_backend": self.activation_backend,
            "decode_backend": self.decode_backend,
            "device_policy": self.device_policy,
            "dtype_compute": self.dtype_compute,
            "dtype_storage": self.dtype_storage,
            "quantization": self.quantization,
            "device_map": self.device_map,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "model_revision": self.model_revision,
            "wrapper_name": self.wrapper_name,
            "architecture": self.architecture,
            "extra": self.extra,
        }


@dataclass
class PromptLayerRecord:
    """Canonical per-layer prompt decode payload."""

    layer_index: int
    layer_name: str
    tokens: torch.Tensor
    token_text: list[str]
    attention_mask: torch.Tensor
    hidden: torch.Tensor
    logits_raw: torch.Tensor | None = None
    logits_model_norm: torch.Tensor | None = None
    logits_tuned: torch.Tensor | None = None
    attention_output: torch.Tensor | None = None
    mlp_output: torch.Tensor | None = None
    attention_logits_raw: torch.Tensor | None = None
    attention_logits_model_norm: torch.Tensor | None = None
    mlp_logits_raw: torch.Tensor | None = None
    mlp_logits_model_norm: torch.Tensor | None = None

    @property
    def hidden_raw(self) -> torch.Tensor:
        return self.hidden

    @property
    def hidden_model_norm(self) -> torch.Tensor:
        # Prompt captures store one canonical residual stream; readout-specific
        # normalization is applied at projection time unless a downstream method
        # explicitly reprojects from this shared hidden state.
        return self.hidden

    def get_hidden(self, mode: PromptHiddenMode = "raw") -> torch.Tensor:
        if mode not in ("raw", "model_norm"):
            raise KeyError(f"Unsupported prompt hidden mode: {mode!r}")
        return self.hidden

    def get_logits(self, mode: PromptReadoutMode = "raw") -> torch.Tensor | None:
        field_name = f"logits_{mode}"
        return getattr(self, field_name, None)

    def get_component_output(self, component: Literal["attention", "mlp"]) -> torch.Tensor | None:
        field_name = f"{component}_output"
        return getattr(self, field_name, None)

    def get_component_logits(
        self,
        component: Literal["attention", "mlp"],
        mode: PromptHiddenMode = "raw",
    ) -> torch.Tensor | None:
        field_name = f"{component}_logits_{mode}"
        return getattr(self, field_name, None)

    @classmethod
    def from_legacy_dict(cls, record: dict[str, Any]) -> "PromptLayerRecord":
        return cls(
            layer_index=int(record["layer_index"]),
            layer_name=str(record["layer_name"]),
            tokens=record["tokens"],
            token_text=list(record.get("token_text", [])),
            attention_mask=record["attention_mask"],
            hidden=record["hidden"],
            logits_raw=record.get("logits_raw"),
            logits_model_norm=record.get("logits_model_norm"),
            logits_tuned=record.get("logits_tuned"),
            attention_output=record.get("attention_output"),
            mlp_output=record.get("mlp_output"),
            attention_logits_raw=record.get("attention_logits_raw"),
            attention_logits_model_norm=record.get("attention_logits_model_norm"),
            mlp_logits_raw=record.get("mlp_logits_raw"),
            mlp_logits_model_norm=record.get("mlp_logits_model_norm"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "tokens": self.tokens,
            "token_text": self.token_text,
            "attention_mask": self.attention_mask,
            "hidden": self.hidden,
        }
        optional_fields = (
            "logits_raw",
            "logits_model_norm",
            "logits_tuned",
            "attention_output",
            "mlp_output",
            "attention_logits_raw",
            "attention_logits_model_norm",
            "mlp_logits_raw",
            "mlp_logits_model_norm",
        )
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value
        return payload


@dataclass
class PromptDecodeArtifact:
    """Canonical prompt-level artifact used for reproducible prompt-lens analyses."""

    prompt_text: str
    prompt_formatted: str
    token_ids: torch.Tensor
    token_text: list[str]
    attention_mask: torch.Tensor
    layer_records: list[PromptLayerRecord]
    backend_metadata: BackendMetadata
    lens_modes: list[LensReadoutMode]
    collection_mode: str = "prompt"
    batch_size: int = 1
    batch_semantics: str = "single_sequence_per_record"
    record_semantics: str = "one_record_per_layer"
    operand_order: OperandOrder = "none"
    prompt_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptDecodeArtifact":
        return cls(
            prompt_text=payload["prompt_text"],
            prompt_formatted=payload["prompt_formatted"],
            token_ids=payload["token_ids"],
            token_text=list(payload["token_text"]),
            attention_mask=payload["attention_mask"],
            layer_records=[PromptLayerRecord.from_legacy_dict(record) for record in payload["layer_records"]],
            backend_metadata=BackendMetadata.from_dict(payload["backend_metadata"]),
            lens_modes=list(payload["lens_modes"]),
            collection_mode=payload.get("collection_mode", "prompt"),
            batch_size=int(payload.get("batch_size", 1)),
            batch_semantics=str(payload.get("batch_semantics", "single_sequence_per_record")),
            record_semantics=str(payload.get("record_semantics", "one_record_per_layer")),
            operand_order=payload.get("operand_order", "none"),
            prompt_id=payload.get("prompt_id"),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "prompt_formatted": self.prompt_formatted,
            "token_ids": self.token_ids,
            "token_text": self.token_text,
            "attention_mask": self.attention_mask,
            "layer_records": [record.to_dict() for record in self.layer_records],
            "backend_metadata": self.backend_metadata.to_dict(),
            "lens_modes": self.lens_modes,
            "collection_mode": self.collection_mode,
            "batch_size": self.batch_size,
            "batch_semantics": self.batch_semantics,
            "record_semantics": self.record_semantics,
            "operand_order": self.operand_order,
            "prompt_id": self.prompt_id,
            "metadata": self.metadata,
            "artifact_schema": "PromptDecodeArtifact",
        }


@dataclass
class ComparisonArtifact:
    """Canonical comparison payload for signed ft-minus-base style outputs."""

    comparison_label: str
    reference_label: str
    operand_order: OperandOrder
    backend_metadata: BackendMetadata
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BackendMetadata",
    "ComparisonArtifact",
    "LensReadoutMode",
    "OperandOrder",
    "PromptHiddenMode",
    "PromptReadoutMode",
    "PromptDecodeArtifact",
    "PromptLayerRecord",
]

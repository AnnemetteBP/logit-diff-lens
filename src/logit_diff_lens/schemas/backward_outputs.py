from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .lens_outputs import BackendMetadata


BackwardTargetKind = Literal["token_id", "target_text"]
BackwardLossKind = Literal["nll"]


@dataclass
class BackwardLayerRecord:
    layer_index: int
    layer_name: str
    token_ids: torch.Tensor
    token_text: list[str]
    hidden_vjp: torch.Tensor | None = None
    attention_vjp: torch.Tensor | None = None
    mlp_vjp: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "token_ids": self.token_ids,
            "token_text": self.token_text,
        }
        if self.hidden_vjp is not None:
            payload["hidden_vjp"] = self.hidden_vjp
        if self.attention_vjp is not None:
            payload["attention_vjp"] = self.attention_vjp
        if self.mlp_vjp is not None:
            payload["mlp_vjp"] = self.mlp_vjp
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BackwardLayerRecord":
        return cls(
            layer_index=int(payload["layer_index"]),
            layer_name=str(payload["layer_name"]),
            token_ids=payload["token_ids"],
            token_text=list(payload["token_text"]),
            hidden_vjp=payload.get("hidden_vjp"),
            attention_vjp=payload.get("attention_vjp"),
            mlp_vjp=payload.get("mlp_vjp"),
        )


@dataclass
class BackwardPromptArtifact:
    prompt_text: str
    prompt_formatted: str
    token_ids: torch.Tensor
    token_text: list[str]
    target_token_id: int
    target_token_text: str
    target_position: int
    target_kind: BackwardTargetKind
    loss_kind: BackwardLossKind
    loss_value: float
    backend_metadata: BackendMetadata
    layer_records: list[BackwardLayerRecord]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "prompt_formatted": self.prompt_formatted,
            "token_ids": self.token_ids,
            "token_text": self.token_text,
            "target_token_id": self.target_token_id,
            "target_token_text": self.target_token_text,
            "target_position": self.target_position,
            "target_kind": self.target_kind,
            "loss_kind": self.loss_kind,
            "loss_value": self.loss_value,
            "backend_metadata": self.backend_metadata.to_dict(),
            "layer_records": [record.to_dict() for record in self.layer_records],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BackwardPromptArtifact":
        return cls(
            prompt_text=payload["prompt_text"],
            prompt_formatted=payload["prompt_formatted"],
            token_ids=payload["token_ids"],
            token_text=list(payload["token_text"]),
            target_token_id=int(payload["target_token_id"]),
            target_token_text=str(payload["target_token_text"]),
            target_position=int(payload["target_position"]),
            target_kind=payload["target_kind"],
            loss_kind=payload["loss_kind"],
            loss_value=float(payload["loss_value"]),
            backend_metadata=BackendMetadata.from_dict(payload["backend_metadata"]),
            layer_records=[BackwardLayerRecord.from_dict(item) for item in payload["layer_records"]],
            metadata=dict(payload.get("metadata", {})),
        )


__all__ = [
    "BackwardLayerRecord",
    "BackwardLossKind",
    "BackwardPromptArtifact",
    "BackwardTargetKind",
]

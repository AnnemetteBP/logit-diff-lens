from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .lens_outputs import BackendMetadata


PatchscopeMappingKind = Literal["identity"]
PatchscopeReadoutMode = Literal["raw", "model_norm"]


@dataclass
class PatchscopePromptArtifact:
    source_prompt_text: str
    target_prompt_text: str
    source_layer_index: int
    source_position: int
    target_layer_index: int
    target_position: int
    mapping_kind: PatchscopeMappingKind
    readout_mode: PatchscopeReadoutMode
    patched_token_ids: torch.Tensor
    patched_token_text: list[str]
    patched_logits: torch.Tensor
    patched_topk_token_ids: torch.Tensor
    patched_topk_token_text: list[list[str]]
    backend_metadata: BackendMetadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_prompt_text": self.source_prompt_text,
            "target_prompt_text": self.target_prompt_text,
            "source_layer_index": self.source_layer_index,
            "source_position": self.source_position,
            "target_layer_index": self.target_layer_index,
            "target_position": self.target_position,
            "mapping_kind": self.mapping_kind,
            "readout_mode": self.readout_mode,
            "patched_token_ids": self.patched_token_ids,
            "patched_token_text": self.patched_token_text,
            "patched_logits": self.patched_logits,
            "patched_topk_token_ids": self.patched_topk_token_ids,
            "patched_topk_token_text": self.patched_topk_token_text,
            "backend_metadata": self.backend_metadata.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PatchscopePromptArtifact":
        return cls(
            source_prompt_text=payload["source_prompt_text"],
            target_prompt_text=payload["target_prompt_text"],
            source_layer_index=int(payload["source_layer_index"]),
            source_position=int(payload["source_position"]),
            target_layer_index=int(payload["target_layer_index"]),
            target_position=int(payload["target_position"]),
            mapping_kind=payload["mapping_kind"],
            readout_mode=payload["readout_mode"],
            patched_token_ids=payload["patched_token_ids"],
            patched_token_text=list(payload["patched_token_text"]),
            patched_logits=payload["patched_logits"],
            patched_topk_token_ids=payload["patched_topk_token_ids"],
            patched_topk_token_text=list(payload["patched_topk_token_text"]),
            backend_metadata=BackendMetadata.from_dict(payload["backend_metadata"]),
            metadata=dict(payload.get("metadata", {})),
        )


__all__ = [
    "PatchscopeMappingKind",
    "PatchscopePromptArtifact",
    "PatchscopeReadoutMode",
]

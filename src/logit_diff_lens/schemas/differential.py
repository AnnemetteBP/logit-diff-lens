from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AlignmentMode = Literal["same_prefix_forcing", "own_trajectory", "teacher_forced_shared_continuation"]
ReadoutMode = Literal["raw", "model_norm", "tuned_self", "tuned_reference"]


@dataclass(frozen=True)
class PairwiseModelSpec:
    model_id: str
    label: str
    tuned_lens_resource_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DifferentialPromptExample:
    prompt_id: str
    prompt_text: str
    token_ids: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DifferentialGenerationMetadata:
    prompt_id: str
    template_id: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    max_new_tokens: int | None = None
    alignment_mode: AlignmentMode = "own_trajectory"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerPositionMetricBundle:
    hidden_metrics: dict[str, Any]
    distribution_metrics: dict[str, Any]
    readout_mode: ReadoutMode
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AlignmentMode",
    "DifferentialGenerationMetadata",
    "DifferentialPromptExample",
    "LayerPositionMetricBundle",
    "PairwiseModelSpec",
    "ReadoutMode",
]

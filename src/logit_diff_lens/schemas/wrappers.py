from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


QuantizationKind = Literal["none", "bitsandbytes_4bit", "bitsandbytes_8bit", "gptq", "awq", "unknown"]
ProjectionMode = Literal["native", "stable_fp32", "backend_defined"]
CollectionMode = Literal["prompt", "generation", "patching", "custom"]


@dataclass(frozen=True)
class WrapperCapabilities:
    """Stable capability flags for a lens wrapper/backend pair."""

    collection_mode: CollectionMode
    backend_name: str
    supports_prompt_forward: bool = True
    supports_generation_forward: bool = False
    supports_patching: bool = False
    supports_attention_subblocks: bool = False
    supports_mlp_subblocks: bool = False
    supports_component_registry: bool = False
    supports_final_norm: bool = True
    supports_lm_head_bias: bool = True
    supports_tied_unembed_detection: bool = True
    supports_quantized_lm_head_projection: bool = False
    supports_manual_dequantization_fallback: bool = False
    supports_stable_fp32_projection: bool = True
    supports_hidden_state_replay: bool = False
    supports_generate_api: bool = False
    supports_cache_aware_generation: bool = False


@dataclass(frozen=True)
class WrapperModelMetadata:
    """Minimal model and runtime metadata required for reproducibility."""

    wrapper_name: str
    backend_name: str
    architecture: str
    model_class: str
    tokenizer_class: str
    device: str
    dtype: str
    quantization_kind: QuantizationKind
    is_quantized: bool
    embeddings_tied: bool | None
    has_final_norm: bool
    has_lm_head_bias: bool | None
    layer_count: int | None
    supports_attention_module_resolution: bool
    supports_mlp_module_resolution: bool
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WrapperOutputSemantics:
    """Explicit semantics for activations/logits exposed by a wrapper."""

    collection_mode: CollectionMode
    captures_embedding_output: bool
    captures_block_output: bool
    captures_attention_output: bool
    captures_mlp_output: bool
    captures_final_norm_output: bool
    block_output_semantics: str
    activation_device_policy: str
    generation_semantics: str | None = None


@dataclass
class PromptForwardResult:
    """Typed shape for teacher-forced prompt collection outputs."""

    activations: dict[str, Any]
    outputs: Any
    metadata: WrapperModelMetadata | None = None


@dataclass
class GenerationForwardResult:
    """Typed shape for generation collection outputs."""

    tokens: Any
    attention_mask: Any
    logits: list[Any]
    activations: list[dict[str, Any]]
    metadata: WrapperModelMetadata | None = None

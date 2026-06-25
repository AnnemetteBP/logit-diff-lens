from __future__ import annotations

from typing import Any

import torch

from .schemas.backward_outputs import BackwardPromptArtifact
from .schemas.lens_outputs import PromptDecodeArtifact, PromptLayerRecord
from .schemas.patchscope_outputs import PatchscopePromptArtifact


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_finite_tensor(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        return
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor)!r}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def validate_layer_order(layer_records: list[PromptLayerRecord]) -> None:
    layer_indices = [record.layer_index for record in layer_records]
    _require(layer_indices, "layer_records must not be empty")
    _require(layer_indices == sorted(layer_indices), f"layer indices are not low-to-high: {layer_indices}")


def validate_prompt_layer_record(record: PromptLayerRecord, expected_tokens: int | None = None) -> None:
    _require(record.layer_name != "", "layer_name must not be empty")
    _require(record.tokens.ndim == 2, f"{record.layer_name}: tokens must have shape [batch, seq]")
    _require(record.attention_mask.ndim == 2, f"{record.layer_name}: attention_mask must have shape [batch, seq]")
    _require(record.hidden.ndim == 3, f"{record.layer_name}: hidden must have shape [batch, seq, hidden]")
    _require(record.tokens.shape[0] == 1, f"{record.layer_name}: only batch size 1 prompt artifacts are supported")
    _require(
        record.tokens.shape == record.attention_mask.shape,
        f"{record.layer_name}: tokens and attention_mask shapes must match",
    )
    _require(
        record.hidden.shape[:2] == record.tokens.shape,
        f"{record.layer_name}: hidden token dimensions must match tokens",
    )
    if expected_tokens is not None:
        _require(
            record.tokens.shape[1] == expected_tokens,
            f"{record.layer_name}: token count {record.tokens.shape[1]} != expected {expected_tokens}",
        )
    _require(
        len(record.token_text) == record.tokens.shape[1],
        f"{record.layer_name}: token_text length {len(record.token_text)} != token count {record.tokens.shape[1]}",
    )

    validate_finite_tensor(f"{record.layer_name}.hidden", record.hidden)

    optional_tensors = {
        "logits_raw": record.logits_raw,
        "logits_model_norm": record.logits_model_norm,
        "attention_output": record.attention_output,
        "mlp_output": record.mlp_output,
        "attention_logits_raw": record.attention_logits_raw,
        "attention_logits_model_norm": record.attention_logits_model_norm,
        "mlp_logits_raw": record.mlp_logits_raw,
        "mlp_logits_model_norm": record.mlp_logits_model_norm,
    }
    for field_name, value in optional_tensors.items():
        validate_finite_tensor(f"{record.layer_name}.{field_name}", value)
        if value is None:
            continue
        if field_name.endswith("output"):
            _require(
                value.shape[:2] == record.tokens.shape,
                f"{record.layer_name}.{field_name}: token dimensions must match tokens",
            )
        else:
            _require(
                value.ndim == 3,
                f"{record.layer_name}.{field_name}: logits-like tensors must have shape [batch, seq, vocab]",
            )
            _require(
                value.shape[:2] == record.tokens.shape,
                f"{record.layer_name}.{field_name}: token dimensions must match tokens",
            )


def validate_backend_metadata(metadata: Any) -> None:
    required_strings = (
        "model_backend",
        "activation_backend",
        "decode_backend",
        "device_policy",
        "dtype_compute",
        "dtype_storage",
        "quantization",
        "device_map",
    )
    for field_name in required_strings:
        value = getattr(metadata, field_name, None)
        _require(isinstance(value, str) and value != "", f"backend_metadata.{field_name} must be a non-empty string")


def validate_prompt_decode_artifact(artifact: PromptDecodeArtifact) -> None:
    _require(artifact.collection_mode == "prompt", "PromptDecodeArtifact.collection_mode must be 'prompt'")
    _require(isinstance(artifact.prompt_text, str), "prompt_text must be a string")
    _require(isinstance(artifact.prompt_formatted, str), "prompt_formatted must be a string")
    _require(artifact.token_ids.ndim == 2, "token_ids must have shape [batch, seq]")
    _require(artifact.attention_mask.ndim == 2, "attention_mask must have shape [batch, seq]")
    _require(
        artifact.token_ids.shape == artifact.attention_mask.shape,
        "token_ids and attention_mask must have matching shape",
    )
    _require(
        len(artifact.token_text) == artifact.token_ids.shape[1],
        "token_text length must match token_ids sequence length",
    )
    _require(artifact.layer_records, "PromptDecodeArtifact.layer_records must not be empty")
    validate_finite_tensor("artifact.token_ids", artifact.token_ids.to(dtype=torch.float32))
    validate_finite_tensor("artifact.attention_mask", artifact.attention_mask.to(dtype=torch.float32))
    validate_backend_metadata(artifact.backend_metadata)
    validate_layer_order(artifact.layer_records)

    expected_tokens = artifact.token_ids.shape[1]
    first_tokens = artifact.token_ids
    first_mask = artifact.attention_mask
    for record in artifact.layer_records:
        validate_prompt_layer_record(record, expected_tokens=expected_tokens)
        _require(
            torch.equal(record.tokens, first_tokens),
            f"{record.layer_name}: record tokens do not match artifact token_ids",
        )
        _require(
            torch.equal(record.attention_mask, first_mask),
            f"{record.layer_name}: record attention_mask does not match artifact attention_mask",
        )


def validate_backward_prompt_artifact(artifact: BackwardPromptArtifact) -> None:
    _require(isinstance(artifact.prompt_text, str), "prompt_text must be a string")
    _require(isinstance(artifact.prompt_formatted, str), "prompt_formatted must be a string")
    _require(artifact.token_ids.ndim == 2, "token_ids must have shape [batch, seq]")
    _require(len(artifact.token_text) == artifact.token_ids.shape[1], "token_text length must match token_ids length")
    _require(artifact.layer_records, "BackwardPromptArtifact.layer_records must not be empty")
    _require(artifact.target_position >= 0, "target_position must be non-negative")
    _require(artifact.target_token_id >= 0, "target_token_id must be non-negative")
    validate_backend_metadata(artifact.backend_metadata)
    validate_finite_tensor("backward.token_ids", artifact.token_ids.to(dtype=torch.float32))
    for record in artifact.layer_records:
        _require(record.token_ids.ndim == 2, f"{record.layer_name}: token_ids must have shape [batch, seq]")
        _require(
            len(record.token_text) == record.token_ids.shape[1],
            f"{record.layer_name}: token_text length must match token_ids length",
        )
        _require(
            torch.equal(record.token_ids, artifact.token_ids),
            f"{record.layer_name}: record token_ids do not match artifact token_ids",
        )
        validate_finite_tensor(f"{record.layer_name}.hidden_vjp", record.hidden_vjp)
        validate_finite_tensor(f"{record.layer_name}.attention_vjp", record.attention_vjp)
        validate_finite_tensor(f"{record.layer_name}.mlp_vjp", record.mlp_vjp)


def validate_patchscope_prompt_artifact(artifact: PatchscopePromptArtifact) -> None:
    _require(isinstance(artifact.source_prompt_text, str), "source_prompt_text must be a string")
    _require(isinstance(artifact.target_prompt_text, str), "target_prompt_text must be a string")
    _require(artifact.source_position >= 0, "source_position must be non-negative")
    _require(artifact.target_position >= 0, "target_position must be non-negative")
    _require(artifact.patched_token_ids.ndim == 2, "patched_token_ids must have shape [batch, seq]")
    _require(
        len(artifact.patched_token_text) == artifact.patched_token_ids.shape[1],
        "patched_token_text length must match patched_token_ids length",
    )
    _require(artifact.patched_logits.ndim == 3, "patched_logits must have shape [batch, seq, vocab]")
    _require(
        artifact.patched_logits.shape[:2] == artifact.patched_token_ids.shape,
        "patched_logits token dimensions must match patched_token_ids",
    )
    validate_backend_metadata(artifact.backend_metadata)
    validate_finite_tensor("patchscope.patched_token_ids", artifact.patched_token_ids.to(dtype=torch.float32))
    validate_finite_tensor("patchscope.patched_logits", artifact.patched_logits)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from .schemas import PromptDecodeArtifact, PromptLayerRecord
from .validation import validate_prompt_decode_artifact


PromptPairAlignmentMode = Literal["same_token_ids", "shared_position_mask"]


@dataclass(frozen=True)
class PromptPairValidationResult:
    valid_mask: torch.Tensor
    layer_indices: list[int]
    layer_names: list[str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_logits(record: PromptLayerRecord, *, readout_mode: str, side_label: str) -> None:
    logits = record.get_logits(readout_mode)  # type: ignore[arg-type]
    _require(
        logits is not None,
        f"{side_label} layer {record.layer_name} is missing logits for readout_mode={readout_mode!r}",
    )
    _require(
        logits.ndim == 3 and logits.shape[0] == 1,
        f"{side_label} layer {record.layer_name} has invalid logits shape {tuple(logits.shape)}",
    )
    _require(
        torch.isfinite(logits).all(),
        f"{side_label} layer {record.layer_name} logits for readout_mode={readout_mode!r} contain NaN or Inf",
    )


def _check_component_logits(
    record: PromptLayerRecord,
    *,
    components: Iterable[str],
    readout_mode: str,
    side_label: str,
) -> None:
    for component in components:
        logits = record.get_component_logits(component, readout_mode)  # type: ignore[arg-type]
        _require(
            logits is not None,
            f"{side_label} layer {record.layer_name} is missing {component} component logits for readout_mode={readout_mode!r}",
        )
        _require(
            logits.ndim == 3 and logits.shape[0] == 1,
            f"{side_label} layer {record.layer_name} has invalid {component} component logit shape {tuple(logits.shape)}",
        )
        _require(
            torch.isfinite(logits).all(),
            f"{side_label} layer {record.layer_name} {component} component logits contain NaN or Inf",
        )


def validate_prompt_artifact_pair(
    artifact_a: PromptDecodeArtifact,
    artifact_b: PromptDecodeArtifact,
    *,
    alignment_mode: PromptPairAlignmentMode = "same_token_ids",
    side_a_label: str = "comparison",
    side_b_label: str = "base",
    readout_mode: str | None = None,
    require_component_logits: Iterable[str] = (),
    require_force_include_input: bool = False,
    require_force_include_output: bool = False,
) -> PromptPairValidationResult:
    validate_prompt_decode_artifact(artifact_a)
    validate_prompt_decode_artifact(artifact_b)

    if readout_mode is not None:
        _require(
            readout_mode in artifact_a.lens_modes,
            f"{side_a_label} artifact does not provide readout_mode={readout_mode!r}",
        )
        _require(
            readout_mode in artifact_b.lens_modes,
            f"{side_b_label} artifact does not provide readout_mode={readout_mode!r}",
        )

    _require(
        artifact_a.token_ids.shape == artifact_b.token_ids.shape,
        f"{side_a_label} and {side_b_label} token shapes differ: "
        f"{tuple(artifact_a.token_ids.shape)} != {tuple(artifact_b.token_ids.shape)}",
    )
    _require(
        artifact_a.attention_mask.shape == artifact_b.attention_mask.shape,
        f"{side_a_label} and {side_b_label} attention-mask shapes differ: "
        f"{tuple(artifact_a.attention_mask.shape)} != {tuple(artifact_b.attention_mask.shape)}",
    )
    _require(
        len(artifact_a.layer_records) == len(artifact_b.layer_records),
        f"{side_a_label} and {side_b_label} layer counts differ: "
        f"{len(artifact_a.layer_records)} != {len(artifact_b.layer_records)}",
    )

    if alignment_mode == "same_token_ids":
        _require(
            torch.equal(artifact_a.token_ids, artifact_b.token_ids),
            f"{alignment_mode} alignment requires identical token ids for {side_a_label} and {side_b_label}",
        )
    elif alignment_mode == "shared_position_mask":
        _require(
            artifact_a.prompt_text == artifact_b.prompt_text,
            f"{alignment_mode} alignment requires identical prompt_text for {side_a_label} and {side_b_label}",
        )
    else:
        raise ValueError(f"Unsupported alignment_mode: {alignment_mode!r}")

    if require_force_include_input:
        _require(
            bool(artifact_a.metadata.get("force_include_input", False)),
            f"{side_a_label} artifact must be captured with force_include_input enabled",
        )
        _require(
            bool(artifact_b.metadata.get("force_include_input", False)),
            f"{side_b_label} artifact must be captured with force_include_input enabled",
        )
    if require_force_include_output:
        _require(
            bool(artifact_a.metadata.get("force_include_output", False)),
            f"{side_a_label} artifact must be captured with force_include_output enabled",
        )
        _require(
            bool(artifact_b.metadata.get("force_include_output", False)),
            f"{side_b_label} artifact must be captured with force_include_output enabled",
        )

    if artifact_a.metadata.get("normalize_embedding_for_readout") != artifact_b.metadata.get("normalize_embedding_for_readout"):
        raise ValueError(
            f"{side_a_label} and {side_b_label} disagree on normalize_embedding_for_readout"
        )
    if artifact_a.metadata.get("force_include_input") != artifact_b.metadata.get("force_include_input"):
        raise ValueError(f"{side_a_label} and {side_b_label} disagree on force_include_input")
    if artifact_a.metadata.get("force_include_output") != artifact_b.metadata.get("force_include_output"):
        raise ValueError(f"{side_a_label} and {side_b_label} disagree on force_include_output")

    valid_mask = artifact_a.attention_mask.bool() & artifact_b.attention_mask.bool()
    _require(bool(valid_mask.any().item()), f"{side_a_label} and {side_b_label} share no valid token positions")

    layer_indices: list[int] = []
    layer_names: list[str] = []
    for record_a, record_b in zip(artifact_a.layer_records, artifact_b.layer_records):
        _require(
            record_a.layer_index == record_b.layer_index,
            f"Layer id mismatch: {side_a_label} has {record_a.layer_index} while {side_b_label} has {record_b.layer_index}",
        )
        _require(
            record_a.layer_name == record_b.layer_name,
            f"Layer name mismatch at layer {record_a.layer_index}: "
            f"{side_a_label}={record_a.layer_name!r}, {side_b_label}={record_b.layer_name!r}",
        )
        _require(
            torch.equal(record_a.tokens, record_b.tokens) if alignment_mode == "same_token_ids" else True,
            f"Layer token mismatch at layer {record_a.layer_index}",
        )
        _require(
            torch.equal(record_a.attention_mask, record_b.attention_mask),
            f"Layer attention-mask mismatch at layer {record_a.layer_index}",
        )
        if readout_mode is not None:
            _check_logits(record_a, readout_mode=readout_mode, side_label=side_a_label)
            _check_logits(record_b, readout_mode=readout_mode, side_label=side_b_label)
        if require_component_logits and record_a.layer_name != "output" and record_a.layer_index >= 0:
            _check_component_logits(
                record_a,
                components=require_component_logits,
                readout_mode=readout_mode or "raw",
                side_label=side_a_label,
            )
            _check_component_logits(
                record_b,
                components=require_component_logits,
                readout_mode=readout_mode or "raw",
                side_label=side_b_label,
            )
        layer_indices.append(int(record_a.layer_index))
        layer_names.append(str(record_a.layer_name))

    return PromptPairValidationResult(
        valid_mask=valid_mask,
        layer_indices=layer_indices,
        layer_names=layer_names,
    )


__all__ = ["PromptPairAlignmentMode", "PromptPairValidationResult", "validate_prompt_artifact_pair"]

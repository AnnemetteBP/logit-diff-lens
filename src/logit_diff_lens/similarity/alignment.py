from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from ..diffing.io import load_prompt_decode_artifact, load_prompt_decode_artifact_bundle
from ..pair_validation import validate_prompt_artifact_pair
from ..schemas import PromptDecodeArtifact, PromptLayerRecord


PromptAlignmentMode = Literal["same_token_ids", "shared_position_mask"]
PromptSampleMode = Literal["flatten_all_valid_positions"]
PromptRepresentationKind = Literal["hidden", "logits"]


@dataclass
class PromptSimilarityInputs:
    representation_kind: str
    readout_mode: str | None
    layer_indices_a: list[int]
    layer_indices_b: list[int]
    layers_a: list[torch.Tensor]
    layers_b: list[torch.Tensor]
    sample_group_ids: torch.Tensor | None
    prompt_text: str | None
    prompt_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_prompt_source(
    source: str | Path | PromptDecodeArtifact | dict[str, Any],
) -> list[PromptDecodeArtifact]:
    if isinstance(source, PromptDecodeArtifact):
        return [source]
    if isinstance(source, dict):
        if "artifacts" in source:
            return [
                PromptDecodeArtifact.from_dict(item) if isinstance(item, dict) else item
                for item in source["artifacts"]
            ]
        return [PromptDecodeArtifact.from_dict(source)]
    path = Path(source)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "artifacts" in payload:
        bundle = load_prompt_decode_artifact_bundle(path)
        return bundle["artifacts"]
    return [load_prompt_decode_artifact(path)]


def _match_prompt_artifacts(
    artifacts_a: list[PromptDecodeArtifact],
    artifacts_b: list[PromptDecodeArtifact],
) -> list[tuple[PromptDecodeArtifact, PromptDecodeArtifact]]:
    if len(artifacts_a) == 1 and len(artifacts_b) == 1:
        return [(artifacts_a[0], artifacts_b[0])]
    keyed_a = {
        artifact.prompt_id if artifact.prompt_id is not None else artifact.prompt_text: artifact
        for artifact in artifacts_a
    }
    keyed_b = {
        artifact.prompt_id if artifact.prompt_id is not None else artifact.prompt_text: artifact
        for artifact in artifacts_b
    }
    shared = [key for key in keyed_a if key in keyed_b]
    if not shared:
        raise ValueError("No shared prompt ids/texts found between the two prompt sources")
    return [(keyed_a[key], keyed_b[key]) for key in shared]


def _layer_map(artifact: PromptDecodeArtifact) -> dict[int, PromptLayerRecord]:
    return {record.layer_index: record for record in artifact.layer_records}


def _validate_prompt_artifact_alignment(
    artifact_a: PromptDecodeArtifact,
    artifact_b: PromptDecodeArtifact,
    *,
    alignment_mode: PromptAlignmentMode,
) -> torch.Tensor:
    return validate_prompt_artifact_pair(
        artifact_a,
        artifact_b,
        alignment_mode=alignment_mode,
        side_a_label="side_a",
        side_b_label="side_b",
    ).valid_mask


def _flatten_hidden(record: PromptLayerRecord, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask[0]
    hidden = record.get_hidden("raw")[0]
    return hidden[mask].to(dtype=torch.float32, device="cpu")


def _flatten_logits(record: PromptLayerRecord, valid_mask: torch.Tensor, *, readout_mode: str) -> torch.Tensor:
    mask = valid_mask[0]
    logits = record.get_logits(readout_mode)
    if logits is None:
        raise ValueError(f"Missing logits for readout_mode={readout_mode!r} at layer {record.layer_name}")
    return logits[0][mask].to(dtype=torch.float32, device="cpu")


def build_prompt_similarity_inputs(
    artifact_a: str | Path | PromptDecodeArtifact | dict[str, Any],
    artifact_b: str | Path | PromptDecodeArtifact | dict[str, Any],
    *,
    representation: PromptRepresentationKind,
    alignment_mode: PromptAlignmentMode = "same_token_ids",
    sample_mode: PromptSampleMode = "flatten_all_valid_positions",
    readout_mode: str | None = None,
    layer_mode: str = "pairwise_all",
    layer_indices_a: list[int] | None = None,
    layer_indices_b: list[int] | None = None,
) -> PromptSimilarityInputs:
    if sample_mode != "flatten_all_valid_positions":
        raise ValueError(f"Unsupported prompt sample_mode: {sample_mode!r}")
    if layer_mode not in {"pairwise_all", "fixed_pairs"}:
        raise ValueError(f"Unsupported prompt layer_mode: {layer_mode!r}")
    matched = _match_prompt_artifacts(
        _load_prompt_source(artifact_a),
        _load_prompt_source(artifact_b),
    )
    if not matched:
        raise ValueError("Prompt similarity found no matched prompt artifacts")

    common_layers: set[int] | None = None
    validated_pairs: list[tuple[PromptDecodeArtifact, PromptDecodeArtifact, torch.Tensor, dict[int, PromptLayerRecord], dict[int, PromptLayerRecord]]] = []
    for side_a, side_b in matched:
        valid_mask = _validate_prompt_artifact_alignment(
            side_a,
            side_b,
            alignment_mode=alignment_mode,
        )
        validated = validate_prompt_artifact_pair(
            side_a,
            side_b,
            alignment_mode=alignment_mode,
            side_a_label="side_a",
            side_b_label="side_b",
            readout_mode=readout_mode if representation == "logits" else None,
        )
        layer_set = set(validated.layer_indices)
        common_layers = layer_set if common_layers is None else (common_layers & layer_set)
        validated_pairs.append((side_a, side_b, valid_mask, _layer_map(side_a), _layer_map(side_b)))

    available_a = sorted(common_layers or [])
    available_b = sorted(common_layers or [])
    chosen_a = available_a if layer_indices_a is None else [int(v) for v in layer_indices_a]
    chosen_b = available_b if layer_indices_b is None else [int(v) for v in layer_indices_b]

    if layer_mode == "fixed_pairs" and chosen_a != chosen_b:
        raise ValueError("fixed_pairs mode requires identical layer index lists on both sides")

    per_layer_a: dict[int, list[torch.Tensor]] = {idx: [] for idx in chosen_a}
    per_layer_b: dict[int, list[torch.Tensor]] = {idx: [] for idx in chosen_b}
    sample_group_ids: list[int] = []
    num_valid = 0

    for example_index, (side_a, side_b, valid_mask, mapping_a, mapping_b) in enumerate(validated_pairs):
        missing_a = [idx for idx in chosen_a if idx not in mapping_a]
        missing_b = [idx for idx in chosen_b if idx not in mapping_b]
        if missing_a or missing_b:
            raise ValueError(f"Requested layers missing: side_a={missing_a}, side_b={missing_b}")
        current_valid = int(valid_mask.sum().item())
        if current_valid <= 0:
            continue
        num_valid += current_valid
        sample_group_ids.extend([example_index] * current_valid)
        if representation == "hidden":
            for idx in chosen_a:
                per_layer_a[idx].append(_flatten_hidden(mapping_a[idx], valid_mask))
            for idx in chosen_b:
                per_layer_b[idx].append(_flatten_hidden(mapping_b[idx], valid_mask))
        elif representation == "logits":
            if readout_mode is None:
                raise ValueError("readout_mode is required when representation='logits'")
            for idx in chosen_a:
                per_layer_a[idx].append(_flatten_logits(mapping_a[idx], valid_mask, readout_mode=readout_mode))
            for idx in chosen_b:
                per_layer_b[idx].append(_flatten_logits(mapping_b[idx], valid_mask, readout_mode=readout_mode))
        else:
            raise ValueError(f"Unsupported prompt representation: {representation!r}")

    if num_valid <= 0:
        raise ValueError("Prompt similarity found zero shared valid positions")

    if representation == "hidden":
        layers_a = [torch.cat(per_layer_a[idx], dim=0) for idx in chosen_a]
        layers_b = [torch.cat(per_layer_b[idx], dim=0) for idx in chosen_b]
        resolved_readout_mode = None
    elif representation == "logits":
        layers_a = [torch.cat(per_layer_a[idx], dim=0) for idx in chosen_a]
        layers_b = [torch.cat(per_layer_b[idx], dim=0) for idx in chosen_b]
        resolved_readout_mode = readout_mode
    else:
        raise ValueError(f"Unsupported prompt representation: {representation!r}")

    resolved_prompt_text = matched[0][0].prompt_text if len(matched) == 1 else None
    resolved_prompt_id = matched[0][0].prompt_id if len(matched) == 1 else None

    return PromptSimilarityInputs(
        representation_kind=representation,
        readout_mode=resolved_readout_mode,
        layer_indices_a=chosen_a,
        layer_indices_b=chosen_b,
        layers_a=layers_a,
        layers_b=layers_b,
        sample_group_ids=torch.tensor(sample_group_ids, dtype=torch.long),
        prompt_text=resolved_prompt_text,
        prompt_id=resolved_prompt_id,
        metadata={
            "alignment_mode": alignment_mode,
            "sample_mode": sample_mode,
            "layer_mode": layer_mode,
            "num_valid_positions": num_valid,
            "num_examples": len(matched),
        },
    )


__all__ = [
    "PromptSimilarityInputs",
    "build_prompt_similarity_inputs",
]

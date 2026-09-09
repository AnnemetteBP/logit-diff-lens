from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..schemas.calibration_outputs import CalibrationRunArtifact
from .prompt import PromptReferenceSource
from .prompt import run_prompt_reference_calibration as _run_prompt_reference_calibration
from ..similarity.generation import _load_generation_payload, _select_prompt_generation_rows
from ..schemas import GenerationLayerRecord, PromptDecodeArtifact, PromptLayerRecord, BackendMetadata
import torch


GenerationCalibrationAlignmentMode = Literal[
    "same_prefix_forcing",
    "teacher_forced_shared_continuation",
    "own_trajectory",
]


def _final_step_rows(rows: list[GenerationLayerRecord]) -> list[GenerationLayerRecord]:
    if not rows:
        raise ValueError("Generation calibration received no rows")
    max_step = max(int(row.step) for row in rows)
    chosen = [row for row in rows if int(row.step) == max_step]
    if not chosen:
        raise ValueError("Could not isolate final generation step rows")
    return chosen


def _row_to_prompt_artifact(rows: list[GenerationLayerRecord], backend_metadata: BackendMetadata | None) -> PromptDecodeArtifact:
    layer_rows = _final_step_rows(rows)
    if not layer_rows:
        raise ValueError("Generation calibration needs at least one final-step row")
    token_ids = layer_rows[0].tokens
    attention_mask = layer_rows[0].attention_mask
    token_text = [str(int(token)) for token in token_ids[0].tolist()]
    layer_records: list[PromptLayerRecord] = []
    for row in sorted(layer_rows, key=lambda item: int(item.layer_index)):
        hidden = row.hidden_raw
        if hidden is None:
            hidden = row.hidden_model_norm
        if hidden is None:
            raise ValueError(f"Generation row {row.layer_name} is missing hidden states")
        layer_records.append(
            PromptLayerRecord(
                layer_index=int(row.layer_index),
                layer_name=str(row.layer_name),
                tokens=row.tokens,
                token_text=token_text,
                attention_mask=row.attention_mask,
                hidden=hidden,
                logits_raw=row.logits_raw,
                logits_model_norm=row.logits_model_norm,
                attention_output=row.attention_output,
                mlp_output=row.mlp_output,
                attention_logits_raw=row.attention_logits_raw,
                attention_logits_model_norm=row.attention_logits_model_norm,
                mlp_logits_raw=row.mlp_logits_raw,
                mlp_logits_model_norm=row.mlp_logits_model_norm,
            )
        )
    meta = backend_metadata.to_dict() if backend_metadata is not None else {
        "model_backend": "unknown",
        "activation_backend": "unknown",
        "decode_backend": "unknown",
        "device_policy": "unknown",
        "dtype_compute": "unknown",
        "dtype_storage": "unknown",
        "quantization": "unknown",
        "device_map": "unknown",
    }
    backend = BackendMetadata.from_dict(meta)
    return PromptDecodeArtifact(
        prompt_text=str(layer_rows[0].prompt_text),
        prompt_formatted=str(layer_rows[0].prompt_formatted or layer_rows[0].prompt_text),
        token_ids=token_ids,
        token_text=token_text,
        attention_mask=attention_mask,
        layer_records=layer_records,
        backend_metadata=backend,
        lens_modes=["raw", "model_norm"],
        prompt_id=None if layer_rows[0].prompt_id is None else str(layer_rows[0].prompt_id),
        metadata={"source": "generation_final_step_projection"},
    )


def _load_generation_as_prompt_artifacts(
    source: str | Path | dict[str, Any],
    *,
    prompt_index: int = 0,
    prompt_text: str | None = None,
) -> list[PromptDecodeArtifact]:
    payload = _load_generation_payload(source)
    rows = payload.get("rows", [])
    backend_metadata = None
    if "backend_metadata" in payload and payload["backend_metadata"] is not None:
        backend_metadata = BackendMetadata.from_dict(payload["backend_metadata"])
    if rows and isinstance(rows[0], dict) and "generated_rows" in rows[0]:
        artifacts: list[PromptDecodeArtifact] = []
        candidates = rows
        if prompt_text is not None:
            candidates = [
                item
                for item in rows
                if (item.get("collection_text") or item.get("prompt_text") or item.get("prompt")) == prompt_text
            ]
        elif prompt_index is not None:
            candidates = [rows[prompt_index]]
        for item in candidates:
            artifacts.append(
                _row_to_prompt_artifact(
                    [GenerationLayerRecord.from_dict(row) if isinstance(row, dict) else row for row in item.get("generated_rows", [])],
                    backend_metadata,
                )
            )
        return artifacts
    selected_rows, _, _, _ = _select_prompt_generation_rows(payload, prompt_index=prompt_index, prompt_text=prompt_text)
    return [_row_to_prompt_artifact(selected_rows, backend_metadata)]


def run_generation_reference_calibration(
    artifact_a: str | Path | dict[str, Any],
    artifact_b: str | Path | dict[str, Any],
    *,
    side_a_label: str = "artifact_a",
    side_b_label: str = "artifact_b",
    alignment_mode: GenerationCalibrationAlignmentMode = "same_prefix_forcing",
    readout_mode_eval: str = "model_norm",
    readout_mode_reference: str = "model_norm",
    reference_kind: str = "reference_relative",
    reference_source: PromptReferenceSource = "final_layer",
    top_k_values: list[int] | None = None,
    num_bins: int = 15,
    binning: str = "equal_mass",
    min_samples_per_cell: int = 2,
    num_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
    prompt_index: int = 0,
    prompt_text: str | None = None,
) -> CalibrationRunArtifact:
    if alignment_mode not in {"same_prefix_forcing", "teacher_forced_shared_continuation"}:
        raise ValueError(
            "Generation reference calibration currently supports same_prefix_forcing and teacher_forced_shared_continuation"
        )
    prompt_artifacts_a = _load_generation_as_prompt_artifacts(
        artifact_a,
        prompt_index=prompt_index,
        prompt_text=prompt_text,
    )
    prompt_artifacts_b = _load_generation_as_prompt_artifacts(
        artifact_b,
        prompt_index=prompt_index,
        prompt_text=prompt_text,
    )
    result = _run_prompt_reference_calibration(
        {"artifacts": [artifact.to_dict() for artifact in prompt_artifacts_a]},
        {"artifacts": [artifact.to_dict() for artifact in prompt_artifacts_b]},
        side_a_label=side_a_label,
        side_b_label=side_b_label,
        alignment_mode="same_token_ids",
        readout_mode_eval=readout_mode_eval,
        readout_mode_reference=readout_mode_reference,
        reference_kind=reference_kind,
        reference_source=reference_source,
        top_k_values=top_k_values,
        num_bins=num_bins,
        binning=binning,
        min_samples_per_cell=min_samples_per_cell,
        num_bootstrap=num_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    result.artifact_family = "generation"
    result.alignment_mode = alignment_mode
    result.metadata.update(
        {
            "prompt_index": prompt_index,
            "prompt_text": prompt_text,
        }
    )
    return result


__all__ = ["run_generation_reference_calibration"]

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from ..schemas import GenerationLayerRecord
from ..schemas.similarity_outputs import SimilarityRunArtifact
from .metrics import js_similarity_from_logits, linear_cka, topk_overlap_similarity
from .nulls import adjust_pvalues, calibrate_layer_pair_similarity, calibrate_scalar_similarity


GenerationAlignmentMode = Literal[
    "same_prefix_forcing",
    "teacher_forced_shared_continuation",
    "own_trajectory",
]
GenerationSampleMode = Literal["flatten_all_valid_positions"]
GenerationRepresentationKind = Literal["hidden", "logits"]
GenerationMetric = Literal["linear_cka", "js_similarity", "topk_overlap"]


@dataclass
class GenerationSimilarityInputs:
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


def _load_generation_payload(path_or_payload: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_payload, (str, Path)):
        return torch.load(Path(path_or_payload), map_location="cpu")
    return path_or_payload


def _as_generation_row(row: dict[str, Any] | GenerationLayerRecord) -> GenerationLayerRecord:
    if isinstance(row, GenerationLayerRecord):
        return row
    return GenerationLayerRecord.from_dict(row)


def _select_prompt_generation_rows(
    payload: dict[str, Any],
    *,
    prompt_index: int = 0,
    prompt_text: str | None = None,
) -> tuple[list[GenerationLayerRecord], str | None, str | None, dict[str, Any]]:
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("Generation payload contains no rows")

    if isinstance(rows[0], dict) and "generated_rows" in rows[0]:
        candidates = rows
        selected: dict[str, Any] | None = None
        if prompt_text is not None:
            for item in candidates:
                candidate_text = item.get("collection_text") or item.get("prompt_text") or item.get("prompt")
                if candidate_text == prompt_text:
                    selected = item
                    break
            if selected is None:
                raise ValueError(f"Prompt not found in generation dataset payload: {prompt_text}")
        else:
            selected = candidates[prompt_index]
        chosen_rows = [_as_generation_row(row) for row in selected.get("generated_rows", [])]
        selected_text = selected.get("collection_text") or selected.get("prompt_text") or selected.get("prompt")
        selected_id = str(selected.get("id")) if selected.get("id") is not None else None
        selected_meta = {
            "dataset_path": payload.get("dataset_path"),
            "continuation_kind": selected.get("continuation_kind"),
            "prompt_format": selected.get("collection_prompt_format"),
            "use_chat_template": selected.get("collection_use_chat_template"),
            "system_prompt": selected.get("collection_system_prompt"),
        }
        return chosen_rows, selected_text, selected_id, selected_meta

    selected_text = None
    selected_id = None
    if prompt_text is not None:
        if rows and str(rows[0].get("prompt_text", "")) != prompt_text:
            raise ValueError(f"Direct generation payload prompt does not match requested prompt_text={prompt_text!r}")
    if rows:
        selected_text = rows[0].get("prompt_text")
        prompt_value = rows[0].get("prompt_id")
        selected_id = None if prompt_value is None else str(prompt_value)
    return [_as_generation_row(row) for row in rows], selected_text, selected_id, {
        "force_include_input": payload.get("force_include_input"),
        "force_include_output": payload.get("force_include_output"),
        "max_new_tokens": payload.get("max_new_tokens"),
    }


def _final_step_rows(rows: list[GenerationLayerRecord]) -> tuple[int, list[GenerationLayerRecord]]:
    if not rows:
        raise ValueError("No generation rows available for similarity")
    max_step = max(int(row.step) for row in rows)
    final_rows = [row for row in rows if int(row.step) == max_step]
    if not final_rows:
        raise ValueError("Could not isolate final generation step rows")
    return max_step, final_rows


def _layer_map(rows: list[GenerationLayerRecord]) -> dict[int, GenerationLayerRecord]:
    return {int(row.layer_index): row for row in rows}


def _valid_mask_for_generation_rows(
    row_a: GenerationLayerRecord,
    row_b: GenerationLayerRecord,
    *,
    alignment_mode: GenerationAlignmentMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens_a = row_a.tokens
    tokens_b = row_b.tokens
    mask_a = row_a.attention_mask.bool()
    mask_b = row_b.attention_mask.bool()
    valid_a = mask_a[0]
    valid_b = mask_b[0]

    if alignment_mode in {"same_prefix_forcing", "teacher_forced_shared_continuation"}:
        if tokens_a.shape != tokens_b.shape or not torch.equal(tokens_a, tokens_b):
            raise ValueError(
                f"{alignment_mode} requires identical visible token ids in the selected generation rows"
            )
        valid = valid_a & valid_b
        return valid, valid.clone()

    if alignment_mode == "own_trajectory":
        n = min(int(valid_a.sum().item()), int(valid_b.sum().item()))
        if n <= 0:
            raise ValueError("own_trajectory alignment found zero valid positions")
        aligned_a = torch.zeros_like(valid_a)
        aligned_b = torch.zeros_like(valid_b)
        idx_a = torch.nonzero(valid_a, as_tuple=False).squeeze(-1)[:n]
        idx_b = torch.nonzero(valid_b, as_tuple=False).squeeze(-1)[:n]
        aligned_a[idx_a] = True
        aligned_b[idx_b] = True
        return aligned_a, aligned_b

    raise ValueError(f"Unsupported generation alignment mode: {alignment_mode!r}")


def _flatten_generation_logits(
    row: GenerationLayerRecord,
    valid_mask: torch.Tensor,
    *,
    readout_mode: str,
) -> torch.Tensor:
    logits = row.get_logits(readout_mode)  # type: ignore[arg-type]
    if logits is None:
        raise ValueError(f"Missing logits_{readout_mode} in generation row for layer {row.layer_name}")
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected logits_{readout_mode} with shape [1, seq, vocab], got {tuple(logits.shape)}")
    return logits[0][valid_mask].to(dtype=torch.float32, device="cpu")


def _flatten_generation_hidden(
    row: GenerationLayerRecord,
    valid_mask: torch.Tensor,
    *,
    readout_mode: str,
) -> torch.Tensor:
    hidden = row.get_hidden(readout_mode)  # type: ignore[arg-type]
    if hidden is None:
        raise ValueError(f"Missing hidden_{readout_mode} in generation row for layer {row.layer_name}")
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError(f"Expected hidden_{readout_mode} with shape [1, seq, hidden], got {tuple(hidden.shape)}")
    if hidden.shape[1] != row.tokens.shape[1]:
        raise ValueError(
            "Generation hidden-state similarity currently requires saved hidden states to match the visible token span. "
            "Use generation logits for robust comparison when the hidden-state rows were saved unsliced."
        )
    return hidden[0][valid_mask].to(dtype=torch.float32, device="cpu")


def build_generation_similarity_inputs(
    payload_a: dict[str, Any] | str | Path,
    payload_b: dict[str, Any] | str | Path,
    *,
    representation: GenerationRepresentationKind,
    alignment_mode: GenerationAlignmentMode = "same_prefix_forcing",
    sample_mode: GenerationSampleMode = "flatten_all_valid_positions",
    readout_mode: str | None = None,
    layer_mode: str = "pairwise_all",
    layer_indices_a: list[int] | None = None,
    layer_indices_b: list[int] | None = None,
    prompt_index: int = 0,
    prompt_text: str | None = None,
) -> GenerationSimilarityInputs:
    if sample_mode != "flatten_all_valid_positions":
        raise ValueError(f"Unsupported generation sample_mode: {sample_mode!r}")
    if layer_mode not in {"pairwise_all", "fixed_pairs"}:
        raise ValueError(f"Unsupported generation layer_mode: {layer_mode!r}")

    loaded_a = _load_generation_payload(payload_a)
    loaded_b = _load_generation_payload(payload_b)
    rows_a, prompt_text_a, prompt_id_a, meta_a = _select_prompt_generation_rows(
        loaded_a, prompt_index=prompt_index, prompt_text=prompt_text
    )
    rows_b, prompt_text_b, prompt_id_b, meta_b = _select_prompt_generation_rows(
        loaded_b, prompt_index=prompt_index, prompt_text=prompt_text
    )

    final_step_a, final_rows_a = _final_step_rows(rows_a)
    final_step_b, final_rows_b = _final_step_rows(rows_b)
    map_a = _layer_map(final_rows_a)
    map_b = _layer_map(final_rows_b)

    available_a = sorted(map_a)
    available_b = sorted(map_b)
    chosen_a = available_a if layer_indices_a is None else [int(v) for v in layer_indices_a]
    chosen_b = available_b if layer_indices_b is None else [int(v) for v in layer_indices_b]
    missing_a = [idx for idx in chosen_a if idx not in map_a]
    missing_b = [idx for idx in chosen_b if idx not in map_b]
    if missing_a or missing_b:
        raise ValueError(f"Requested generation layers missing: side_a={missing_a}, side_b={missing_b}")
    if layer_mode == "fixed_pairs" and chosen_a != chosen_b:
        raise ValueError("fixed_pairs mode requires identical layer index lists on both sides")

    if representation == "logits" and readout_mode is None:
        raise ValueError("readout_mode is required when generation representation='logits'")
    if representation == "hidden" and readout_mode is None:
        raise ValueError("readout_mode is required when generation representation='hidden'")

    layers_a: list[torch.Tensor] = []
    layers_b: list[torch.Tensor] = []
    for idx_a, idx_b in zip(chosen_a, chosen_b) if layer_mode == "fixed_pairs" else []:
        del idx_a, idx_b
    if layer_mode == "pairwise_all":
        row_ref_a = map_a[chosen_a[0]]
        row_ref_b = map_b[chosen_b[0]]
        valid_a, valid_b = _valid_mask_for_generation_rows(
            row_ref_a,
            row_ref_b,
            alignment_mode=alignment_mode,
        )
        if representation == "logits":
            layers_a = [_flatten_generation_logits(map_a[idx], valid_a, readout_mode=readout_mode or "model_norm") for idx in chosen_a]
            layers_b = [_flatten_generation_logits(map_b[idx], valid_b, readout_mode=readout_mode or "model_norm") for idx in chosen_b]
        else:
            layers_a = [_flatten_generation_hidden(map_a[idx], valid_a, readout_mode=readout_mode or "model_norm") for idx in chosen_a]
            layers_b = [_flatten_generation_hidden(map_b[idx], valid_b, readout_mode=readout_mode or "model_norm") for idx in chosen_b]
    else:
        for idx_a, idx_b in zip(chosen_a, chosen_b):
            row_a = map_a[idx_a]
            row_b = map_b[idx_b]
            valid_a, valid_b = _valid_mask_for_generation_rows(
                row_a,
                row_b,
                alignment_mode=alignment_mode,
            )
            if representation == "logits":
                layers_a.append(_flatten_generation_logits(row_a, valid_a, readout_mode=readout_mode or "model_norm"))
                layers_b.append(_flatten_generation_logits(row_b, valid_b, readout_mode=readout_mode or "model_norm"))
            else:
                layers_a.append(_flatten_generation_hidden(row_a, valid_a, readout_mode=readout_mode or "model_norm"))
                layers_b.append(_flatten_generation_hidden(row_b, valid_b, readout_mode=readout_mode or "model_norm"))

    num_valid_positions = int(layers_a[0].shape[0]) if layers_a else 0
    if num_valid_positions <= 0:
        raise ValueError("Generation similarity found zero aligned positions")

    resolved_prompt_text = prompt_text_a if prompt_text_a is not None else prompt_text_b
    resolved_prompt_id = prompt_id_a if prompt_id_a is not None else prompt_id_b
    return GenerationSimilarityInputs(
        representation_kind=representation,
        readout_mode=readout_mode,
        layer_indices_a=chosen_a,
        layer_indices_b=chosen_b,
        layers_a=layers_a,
        layers_b=layers_b,
        sample_group_ids=torch.zeros(layers_a[0].shape[0], dtype=torch.long) if layers_a else None,
        prompt_text=resolved_prompt_text,
        prompt_id=resolved_prompt_id,
        metadata={
            "alignment_mode": alignment_mode,
            "sample_mode": sample_mode,
            "layer_mode": layer_mode,
            "num_valid_positions": num_valid_positions,
            "final_step_a": final_step_a,
            "final_step_b": final_step_b,
            "payload_meta_a": meta_a,
            "payload_meta_b": meta_b,
            "warning": "trajectory-relative alignment" if alignment_mode == "own_trajectory" else None,
        },
    )


def _metric_spec(metric_name: GenerationMetric, *, top_k: int):
    if metric_name == "linear_cka":
        return linear_cka, "hidden", 1.0
    if metric_name == "js_similarity":
        return js_similarity_from_logits, "logits", 1.0
    if metric_name == "topk_overlap":
        return (lambda a, b: topk_overlap_similarity(a, b, k=top_k)), "logits", 1.0
    raise ValueError(f"Unsupported generation similarity metric: {metric_name!r}")


def run_generation_similarity(
    payload_a: dict[str, Any] | str | Path,
    payload_b: dict[str, Any] | str | Path,
    *,
    side_a_label: str = "artifact_a",
    side_b_label: str = "artifact_b",
    representation: str,
    metric: GenerationMetric,
    alignment_mode: GenerationAlignmentMode = "same_prefix_forcing",
    sample_mode: GenerationSampleMode = "flatten_all_valid_positions",
    layer_mode: str = "pairwise_all",
    readout_mode: str | None = None,
    layer_indices_a: list[int] | None = None,
    layer_indices_b: list[int] | None = None,
    prompt_index: int = 0,
    prompt_text: str | None = None,
    num_permutations: int = 1000,
    alpha: float = 0.05,
    top_k: int = 20,
    permutation_unit: str = "auto",
    multiple_testing_method: str = "fdr_bh",
    seed: int | None = None,
) -> SimilarityRunArtifact:
    sim_fn, expected_representation, similarity_max = _metric_spec(metric, top_k=top_k)
    if representation != expected_representation:
        raise ValueError(
            f"Metric {metric!r} requires representation {expected_representation!r}, got {representation!r}"
        )

    inputs = build_generation_similarity_inputs(
        payload_a,
        payload_b,
        representation=representation,
        alignment_mode=alignment_mode,
        sample_mode=sample_mode,
        readout_mode=readout_mode,
        layer_mode=layer_mode,
        layer_indices_a=layer_indices_a,
        layer_indices_b=layer_indices_b,
        prompt_index=prompt_index,
        prompt_text=prompt_text,
    )

    metadata = {
        "metric": metric,
        "representation": representation,
        "top_k": top_k,
        "num_permutations": num_permutations,
        "alpha": alpha,
        "permutation_unit": permutation_unit,
        "multiple_testing_method": multiple_testing_method,
        **inputs.metadata,
    }
    scalar_results = []
    matrix_results = []

    if layer_mode == "fixed_pairs":
        for idx_a, idx_b, Xa, Yb in zip(
            inputs.layer_indices_a,
            inputs.layer_indices_b,
            inputs.layers_a,
            inputs.layers_b,
        ):
            scalar_results.append(
                calibrate_scalar_similarity(
                    Xa,
                    Yb,
                    sim_fn,
                    metric_name=metric,
                    representation_kind=representation,
                    num_permutations=num_permutations,
                    alpha=alpha,
                    similarity_max=similarity_max,
                    permutation_unit=permutation_unit,
                    group_ids=inputs.sample_group_ids,
                    seed=seed,
                    metadata={
                        "layer_index_a": idx_a,
                        "layer_index_b": idx_b,
                        "readout_mode": readout_mode,
                    },
                )
            )
        adjusted = adjust_pvalues([item.p_value for item in scalar_results], multiple_testing_method)
        for item, adjusted_p in zip(scalar_results, adjusted):
            item.adjusted_p_value = float(adjusted_p)
    else:
        matrix_results.append(
            calibrate_layer_pair_similarity(
                inputs.layers_a,
                inputs.layers_b,
                sim_fn,
                metric_name=metric,
                representation_kind=representation,
                layer_indices_a=inputs.layer_indices_a,
                layer_indices_b=inputs.layer_indices_b,
                aggregate="max",
                num_permutations=num_permutations,
                alpha=alpha,
                similarity_max=similarity_max,
                permutation_unit=permutation_unit,
                group_ids=inputs.sample_group_ids,
                multiple_testing_method=multiple_testing_method,
                seed=seed,
                metadata={"readout_mode": readout_mode},
            )
        )

    return SimilarityRunArtifact(
        side_a_label=side_a_label,
        side_b_label=side_b_label,
        artifact_family="generation",
        prompt_id=inputs.prompt_id,
        prompt_text=inputs.prompt_text,
        readout_mode=readout_mode,
        alignment_mode=alignment_mode,
        backend_metadata_a=None,
        backend_metadata_b=None,
        scalar_results=scalar_results,
        matrix_results=matrix_results,
        metadata=metadata,
    )


__all__ = [
    "GenerationSimilarityInputs",
    "build_generation_similarity_inputs",
    "run_generation_similarity",
]

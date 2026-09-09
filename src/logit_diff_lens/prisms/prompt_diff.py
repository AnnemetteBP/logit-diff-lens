from __future__ import annotations

from pathlib import Path

import torch

from ..pair_validation import validate_prompt_artifact_pair
from ..schemas import PromptDecodeArtifact
from ..schemas.prism_outputs import PromptDiffPrismHeatmapArtifact
from .prompt import _load_prompt_artifact, _require_logits


def _validate_pair(ft_artifact: PromptDecodeArtifact, base_artifact: PromptDecodeArtifact, *, readout_mode: str) -> None:
    validate_prompt_artifact_pair(
        ft_artifact,
        base_artifact,
        alignment_mode="same_token_ids",
        side_a_label="comparison",
        side_b_label="base",
        readout_mode=readout_mode,
        require_component_logits=("attention", "mlp"),
        require_force_include_input=True,
        require_force_include_output=True,
    )


def _clean_token(token: str | None) -> str:
    text = "" if token is None else str(token)
    text = text.replace("Ġ", " ").replace("▁", " ")
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text.strip() or " "


def _predictive_positions(artifact: PromptDecodeArtifact) -> list[int]:
    valid = torch.nonzero(artifact.attention_mask[0].bool(), as_tuple=False).squeeze(-1).tolist()
    positions = [int(v) for v in valid]
    if len(positions) < 2:
        raise ValueError("Prompt diff prism requires at least two valid tokens to define a next-token prediction")
    return positions[:-1]


def _infer_bias_from_saved_components(
    embedding_logits: torch.Tensor,
    attention_logits_by_layer: list[torch.Tensor],
    mlp_logits_by_layer: list[torch.Tensor],
    total_logits: torch.Tensor,
) -> torch.Tensor:
    component_count = 1 + len(attention_logits_by_layer) + len(mlp_logits_by_layer)
    if component_count <= 1:
        return torch.zeros_like(total_logits)
    summed = embedding_logits.clone()
    for tensor in attention_logits_by_layer:
        summed = summed + tensor
    for tensor in mlp_logits_by_layer:
        summed = summed + tensor
    return (summed - total_logits) / float(component_count - 1)


def build_prompt_diff_prism_artifact(
    ft_source: str | Path | PromptDecodeArtifact,
    base_source: str | Path | PromptDecodeArtifact,
    *,
    prompt_index: int = 0,
    readout_mode: str = "raw",
    top_k: int = 10,
    position_index: int | None = None,
    token_selection: str = "largest_abs_delta",
    side_ft_label: str = "ft",
    side_base_label: str = "base",
    include_total_row: bool = True,
) -> PromptDiffPrismHeatmapArtifact:
    ft_artifact = _load_prompt_artifact(ft_source, prompt_index=prompt_index)
    base_artifact = _load_prompt_artifact(base_source, prompt_index=prompt_index)
    _validate_pair(ft_artifact, base_artifact, readout_mode=readout_mode)

    predictive_positions = _predictive_positions(ft_artifact)
    selected_position = predictive_positions[-1] if position_index is None else int(position_index)
    if selected_position not in predictive_positions:
        raise ValueError(
            f"position_index={selected_position} is invalid for prompt diff prism; valid predictive positions are {predictive_positions}"
        )

    ft_layer_map = {record.layer_index: record for record in ft_artifact.layer_records}
    base_layer_map = {record.layer_index: record for record in base_artifact.layer_records}
    if -1 not in ft_layer_map or -1 not in base_layer_map:
        raise ValueError("Prompt diff prism requires the embedding/input row; recollect with --force-include-input")

    shared_block_indices = sorted(
        idx
        for idx in ft_layer_map
        if idx >= 0 and idx in base_layer_map and ft_layer_map[idx].layer_name != "output" and base_layer_map[idx].layer_name != "output"
    )
    if not shared_block_indices:
        raise ValueError("Prompt diff prism found no shared block layers to analyze")

    final_layer_index = max(idx for idx in ft_layer_map if idx in base_layer_map)
    ft_total = _require_logits(ft_layer_map[final_layer_index].get_logits(readout_mode), name="ft.total")[selected_position].to(torch.float32)
    base_total = _require_logits(base_layer_map[final_layer_index].get_logits(readout_mode), name="base.total")[selected_position].to(torch.float32)
    total_delta = ft_total - base_total

    ft_embedding = _require_logits(ft_layer_map[-1].get_logits(readout_mode), name="ft.embedding")[selected_position].to(torch.float32)
    base_embedding = _require_logits(base_layer_map[-1].get_logits(readout_mode), name="base.embedding")[selected_position].to(torch.float32)

    ft_attn_rows: list[torch.Tensor] = []
    base_attn_rows: list[torch.Tensor] = []
    ft_mlp_rows: list[torch.Tensor] = []
    base_mlp_rows: list[torch.Tensor] = []
    for layer_index in shared_block_indices:
        ft_attn_rows.append(
            _require_logits(ft_layer_map[layer_index].get_component_logits("attention", readout_mode), name=f"ft.layer_{layer_index}.attention")[selected_position].to(torch.float32)
        )
        base_attn_rows.append(
            _require_logits(base_layer_map[layer_index].get_component_logits("attention", readout_mode), name=f"base.layer_{layer_index}.attention")[selected_position].to(torch.float32)
        )
        ft_mlp_rows.append(
            _require_logits(ft_layer_map[layer_index].get_component_logits("mlp", readout_mode), name=f"ft.layer_{layer_index}.mlp")[selected_position].to(torch.float32)
        )
        base_mlp_rows.append(
            _require_logits(base_layer_map[layer_index].get_component_logits("mlp", readout_mode), name=f"base.layer_{layer_index}.mlp")[selected_position].to(torch.float32)
        )

    ft_bias = _infer_bias_from_saved_components(ft_embedding, ft_attn_rows, ft_mlp_rows, ft_total)
    base_bias = _infer_bias_from_saved_components(base_embedding, base_attn_rows, base_mlp_rows, base_total)
    delta_bias = ft_bias - base_bias

    # Fold the LM-head bias delta into the embedding row so the displayed sequence
    # matches the expected prism traversal over embedding/attn/mlp/full-layer states.
    embedding_delta = (ft_embedding - ft_bias) - (base_embedding - base_bias) + delta_bias
    component_labels: list[str] = ["embedding"]
    component_rows: list[torch.Tensor] = [embedding_delta]
    cumulative = embedding_delta.clone()
    for layer_index, ft_attn, base_attn, ft_mlp, base_mlp in zip(shared_block_indices, ft_attn_rows, base_attn_rows, ft_mlp_rows, base_mlp_rows):
        attn_delta = (ft_attn - ft_bias) - (base_attn - base_bias)
        mlp_delta = (ft_mlp - ft_bias) - (base_mlp - base_bias)
        component_labels.append(f"attn_{layer_index}")
        component_rows.append(attn_delta)
        cumulative = cumulative + attn_delta
        component_labels.append(f"mlp_{layer_index}")
        component_rows.append(mlp_delta)
        cumulative = cumulative + mlp_delta
        component_labels.append(f"full_layer_{layer_index}")
        component_rows.append(cumulative.clone())

    stacked = torch.stack(component_rows, dim=0)
    summed = cumulative
    gap = total_delta - summed

    if token_selection == "largest_abs_delta":
        ranking = torch.topk(total_delta.abs(), k=min(int(top_k), int(total_delta.shape[0])), dim=-1).indices
    elif token_selection == "largest_positive_delta":
        ranking = torch.topk(total_delta, k=min(int(top_k), int(total_delta.shape[0])), dim=-1).indices
    elif token_selection == "largest_negative_delta":
        ranking = torch.topk(-total_delta, k=min(int(top_k), int(total_delta.shape[0])), dim=-1).indices
    else:
        raise ValueError(f"Unsupported token_selection={token_selection!r}")

    selected_token_ids = ranking.to(dtype=torch.long, device="cpu")
    selected_component_rows = stacked.index_select(dim=1, index=selected_token_ids.to(stacked.device))
    selected_total_delta = total_delta.index_select(dim=0, index=selected_token_ids.to(total_delta.device))
    selected_summed = summed.index_select(dim=0, index=selected_token_ids.to(summed.device))
    selected_gap = gap.index_select(dim=0, index=selected_token_ids.to(gap.device))

    final_labels = list(component_labels)
    final_matrix = selected_component_rows
    if include_total_row:
        final_labels = final_labels + ["output_l+1"]
        final_matrix = torch.cat([final_matrix, selected_total_delta.unsqueeze(0)], dim=0)

    selected_token_text = [
        _clean_token(ft_artifact.backend_metadata.tokenizer_id if False else ft_artifact.token_text[selected_position])  # type: ignore[arg-type]
        for _ in range(0)
    ]
    selected_token_text = []
    vocab_token_map = ft_artifact.metadata.get("tokenizer_decode_cache", {})
    for token_id in selected_token_ids.tolist():
        text = None
        if isinstance(vocab_token_map, dict):
            text = vocab_token_map.get(str(int(token_id))) or vocab_token_map.get(int(token_id))
        selected_token_text.append(_clean_token(text if text is not None else str(int(token_id))))

    predictor_token_id = int(ft_artifact.token_ids[0, selected_position].item())
    predictor_token_text = _clean_token(ft_artifact.token_text[selected_position])
    target_token_id = None
    target_token_text = None
    if selected_position + 1 < ft_artifact.token_ids.shape[1]:
        target_token_id = int(ft_artifact.token_ids[0, selected_position + 1].item())
        target_token_text = _clean_token(ft_artifact.token_text[selected_position + 1])

    return PromptDiffPrismHeatmapArtifact(
        prompt_text=ft_artifact.prompt_text,
        prompt_formatted=ft_artifact.prompt_formatted,
        prompt_id=ft_artifact.prompt_id,
        readout_mode=readout_mode,
        position_index=selected_position,
        predictor_token_id=predictor_token_id,
        predictor_token_text=predictor_token_text,
        target_token_id=target_token_id,
        target_token_text=target_token_text,
        side_ft_label=side_ft_label,
        side_base_label=side_base_label,
        delta_definition=f"{side_ft_label} - {side_base_label}",
        token_selection=token_selection,
        top_k=int(top_k),
        component_labels=final_labels,
        selected_token_ids=selected_token_ids,
        selected_token_text=selected_token_text,
        contribution_matrix=final_matrix.to(torch.float32).cpu(),
        total_delta_logits=selected_total_delta.to(torch.float32).cpu(),
        summed_component_logits=selected_summed.to(torch.float32).cpu(),
        reconstruction_gap=selected_gap.to(torch.float32).cpu(),
        backend_metadata_ft=ft_artifact.backend_metadata,
        backend_metadata_base=base_artifact.backend_metadata,
        metadata={
            "source_kind": "prompt_capture_artifact_pair",
            "component_labels_without_total": component_labels,
            "max_abs_gap": float(selected_gap.abs().max().item()),
            "mean_abs_gap": float(selected_gap.abs().mean().item()),
            "final_layer_index": final_layer_index,
            "shared_block_indices": shared_block_indices,
            "bias_delta_abs_max": float(delta_bias.abs().max().item()),
            "bias_folded_into_embedding": True,
            "decomposition_exact": readout_mode == "raw",
            "output_row_label": "output_l+1",
        },
    )


__all__ = ["build_prompt_diff_prism_artifact"]

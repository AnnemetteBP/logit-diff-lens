from __future__ import annotations

from pathlib import Path

import torch

from ..diffing.io import load_prompt_decode_artifact, load_prompt_decode_artifact_bundle
from ..schemas import PromptDecodeArtifact
from ..schemas.prism_outputs import PromptPrismArtifact


def _load_prompt_artifact(
    source: str | Path | PromptDecodeArtifact,
    *,
    prompt_index: int = 0,
) -> PromptDecodeArtifact:
    if isinstance(source, PromptDecodeArtifact):
        return source
    path = Path(source)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "artifacts" in payload:
        bundle = load_prompt_decode_artifact_bundle(path)
        artifacts = bundle["artifacts"]
        if not artifacts:
            raise ValueError(f"No prompt artifacts found in bundle: {path}")
        try:
            return artifacts[prompt_index]
        except IndexError as exc:
            raise ValueError(f"prompt_index={prompt_index} out of range for bundle with {len(artifacts)} artifacts") from exc
    return load_prompt_decode_artifact(path)


def _require_logits(tensor: torch.Tensor | None, *, name: str) -> torch.Tensor:
    if tensor is None:
        raise ValueError(
            f"Missing required saved logits for {name}. Recollect with --save-logits"
            " and, for subblocks, --collect-components --project-component-logits."
        )
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"Expected logits for {name} with shape [1, seq, vocab], got {tuple(tensor.shape)}")
    return tensor[0].to(dtype=torch.float32, device="cpu")


def _clean_token(token: str | None) -> str:
    text = "" if token is None else str(token)
    text = text.replace("Ġ", " ").replace("▁", " ")
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text.strip() or " "


def _select_positions(artifact: PromptDecodeArtifact, *, start_idx: int | None, end_idx: int | None) -> list[int]:
    valid = torch.nonzero(artifact.attention_mask[0].bool(), as_tuple=False).squeeze(-1).tolist()
    positions = [int(v) for v in valid]
    if start_idx is not None:
        positions = [pos for pos in positions if pos >= int(start_idx)]
    if end_idx is not None:
        positions = [pos for pos in positions if pos <= int(end_idx)]
    if not positions:
        raise ValueError("Selected prompt prism positions are empty")
    return positions


def _select_token_ids(
    output_logits: torch.Tensor,
    *,
    top_k: int,
    token_selection: str,
) -> torch.Tensor:
    score = output_logits.mean(dim=0)
    k = min(int(top_k), int(score.shape[0]))
    if token_selection == "largest_positive_logit":
        ranking = torch.topk(score, k=k, dim=-1).indices
    elif token_selection == "largest_negative_logit":
        ranking = torch.topk(-score, k=k, dim=-1).indices
    elif token_selection == "largest_abs_logit":
        ranking = torch.topk(score.abs(), k=k, dim=-1).indices
    else:
        raise ValueError(f"Unsupported token_selection={token_selection!r}")
    return ranking.to(dtype=torch.long, device="cpu")


def build_prompt_prism_artifact(
    source: str | Path | PromptDecodeArtifact,
    *,
    prompt_index: int = 0,
    readout_mode: str = "model_norm",
    top_k: int = 10,
    start_idx: int | None = None,
    end_idx: int | None = None,
    token_selection: str = "largest_positive_logit",
) -> PromptPrismArtifact:
    artifact = _load_prompt_artifact(source, prompt_index=prompt_index)
    positions = _select_positions(artifact, start_idx=start_idx, end_idx=end_idx)
    layer_map = {record.layer_index: record for record in artifact.layer_records}
    if -1 not in layer_map:
        raise ValueError("Prompt prism analysis requires the embedding/input row; recollect with --force-include-input")

    block_indices = sorted(
        idx for idx, record in layer_map.items() if idx >= 0 and record.layer_name != "output"
    )
    if not block_indices:
        raise ValueError("Prompt prism analysis found no block layers to analyze")

    output_record = None
    for record in artifact.layer_records:
        if record.layer_name == "output":
            output_record = record
            break
    if output_record is None:
        output_record = layer_map[block_indices[-1]]

    embedding_logits = _require_logits(layer_map[-1].get_logits(readout_mode), name="embedding")[positions]
    output_logits = _require_logits(output_record.get_logits(readout_mode), name="output")[positions]
    selected_token_ids = _select_token_ids(
        output_logits,
        top_k=top_k,
        token_selection=token_selection,
    )

    component_labels: list[str] = ["embedding"]
    component_rows: list[torch.Tensor] = [embedding_logits]
    additive_rows: list[torch.Tensor] = [embedding_logits]
    cumulative_rows: list[torch.Tensor] = []

    additive_running = embedding_logits.clone()
    for layer_index in block_indices:
        record = layer_map[layer_index]
        attention_logits = _require_logits(
            record.get_component_logits("attention", readout_mode),
            name=f"layer_{layer_index}.attention",
        )[positions]
        mlp_logits = _require_logits(
            record.get_component_logits("mlp", readout_mode),
            name=f"layer_{layer_index}.mlp",
        )[positions]
        full_layer_logits = _require_logits(
            record.get_logits(readout_mode),
            name=f"layer_{layer_index}.full_layer",
        )[positions]

        component_labels.extend(
            [f"attn_{layer_index}", f"mlp_{layer_index}", f"full_layer_{layer_index}"]
        )
        component_rows.extend([attention_logits, mlp_logits, full_layer_logits])
        additive_rows.extend([attention_logits, mlp_logits])
        additive_running = additive_running + attention_logits + mlp_logits
        cumulative_rows.append(full_layer_logits)

    component_labels.append("output_l+1")
    component_rows.append(output_logits)

    component_tensor = torch.stack(component_rows, dim=1)
    contribution_tensor = component_tensor.index_select(dim=2, index=selected_token_ids)

    additive_component_logits = additive_running.index_select(dim=1, index=selected_token_ids)
    cumulative_full_layer_logits = cumulative_rows[-1].index_select(dim=1, index=selected_token_ids)
    output_token_logits = output_logits.index_select(dim=1, index=selected_token_ids)
    output_residual_gap = output_token_logits - additive_component_logits

    vocab_token_map = artifact.metadata.get("tokenizer_decode_cache", {})
    selected_token_text: list[str] = []
    for token_id in selected_token_ids.tolist():
        decoded = None
        if isinstance(vocab_token_map, dict):
            decoded = vocab_token_map.get(str(int(token_id))) or vocab_token_map.get(int(token_id))
        selected_token_text.append(_clean_token(decoded if decoded is not None else str(int(token_id))))

    return PromptPrismArtifact(
        prompt_text=artifact.prompt_text,
        prompt_formatted=artifact.prompt_formatted,
        prompt_id=artifact.prompt_id,
        readout_mode=readout_mode,
        top_k=int(top_k),
        token_selection=token_selection,
        position_indices=positions,
        token_ids=artifact.token_ids[:, positions].to(dtype=torch.long, device="cpu"),
        token_text=[artifact.token_text[pos] for pos in positions],
        component_labels=component_labels,
        selected_token_ids=selected_token_ids,
        selected_token_text=selected_token_text,
        contribution_tensor=contribution_tensor.to(dtype=torch.float32, device="cpu"),
        output_token_logits=output_token_logits.to(dtype=torch.float32, device="cpu"),
        additive_component_logits=additive_component_logits.to(dtype=torch.float32, device="cpu"),
        cumulative_full_layer_logits=cumulative_full_layer_logits.to(dtype=torch.float32, device="cpu"),
        output_residual_gap=output_residual_gap.to(dtype=torch.float32, device="cpu"),
        backend_metadata=artifact.backend_metadata,
        metadata={
            "source_kind": "prompt_capture_artifact",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_layers": len(block_indices),
            "num_positions": len(positions),
            "component_labels_without_output": component_labels[:-1],
            "max_abs_output_gap": float(output_residual_gap.abs().max().item()),
            "mean_abs_output_gap": float(output_residual_gap.abs().mean().item()),
            "normalize_embedding_for_readout": bool(
                artifact.metadata.get("normalize_embedding_for_readout", False)
            ),
        },
    )


__all__ = ["build_prompt_prism_artifact", "_load_prompt_artifact", "_require_logits"]

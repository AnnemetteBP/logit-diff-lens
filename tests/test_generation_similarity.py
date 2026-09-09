from __future__ import annotations

import torch

import pytest

from logit_diff_lens.similarity import (
    build_generation_similarity_inputs,
    run_generation_similarity,
)


def _row(
    *,
    prompt_id: int = 0,
    prompt_text: str = "demo",
    step: int,
    layer_index: int,
    tokens: list[int],
    raw_shift: float = 0.0,
) -> dict:
    seq = len(tokens)
    tok = torch.tensor([tokens], dtype=torch.long)
    mask = torch.ones_like(tok)
    logits = torch.arange(seq * 5, dtype=torch.float32).reshape(1, seq, 5) + raw_shift
    hidden = torch.arange(seq * 3, dtype=torch.float32).reshape(1, seq, 3) + raw_shift
    return {
        "prompt_id": prompt_id,
        "prompt_text": prompt_text,
        "batch_index": 0,
        "step": step,
        "layer_index": layer_index,
        "layer_name": "embedding" if layer_index == -1 else f"layer_{layer_index}",
        "tokens": tok,
        "attention_mask": mask,
        "hidden_model_norm": hidden.clone(),
        "logits_model_norm": logits.clone(),
        "hidden_raw": hidden.clone(),
        "logits_raw": logits.clone(),
    }


def _direct_payload(*, tokens: list[int], raw_shift: float = 0.0) -> dict:
    return {
        "rows": [
            _row(step=0, layer_index=-1, tokens=tokens[:-1], raw_shift=raw_shift),
            _row(step=0, layer_index=0, tokens=tokens[:-1], raw_shift=raw_shift + 0.1),
            _row(step=1, layer_index=-1, tokens=tokens, raw_shift=raw_shift),
            _row(step=1, layer_index=0, tokens=tokens, raw_shift=raw_shift + 0.1),
        ],
        "force_include_input": True,
        "force_include_output": False,
        "max_new_tokens": 2,
    }


def _dataset_payload(*, tokens: list[int], prompt_text: str = "demo", raw_shift: float = 0.0) -> dict:
    return {
        "rows": [
            {
                "id": 7,
                "collection_text": prompt_text,
                "continuation_kind": "prompt_plus_response",
                "generated_rows": _direct_payload(tokens=tokens, raw_shift=raw_shift)["rows"],
            }
        ]
    }


def test_generation_similarity_inputs_pairwise_logits_same_prefix() -> None:
    payload_a = _direct_payload(tokens=[10, 11, 12], raw_shift=0.0)
    payload_b = _direct_payload(tokens=[10, 11, 12], raw_shift=0.2)
    result = build_generation_similarity_inputs(
        payload_a,
        payload_b,
        representation="logits",
        alignment_mode="same_prefix_forcing",
        sample_mode="flatten_all_valid_positions",
        layer_mode="pairwise_all",
        readout_mode="model_norm",
    )
    assert result.layer_indices_a == [-1, 0]
    assert result.layers_a[0].shape == (3, 5)
    assert result.layers_b[1].shape == (3, 5)


def test_generation_similarity_inputs_dataset_prompt_selection() -> None:
    payload_a = _dataset_payload(tokens=[10, 11, 12], prompt_text="chosen", raw_shift=0.0)
    payload_b = _dataset_payload(tokens=[10, 11, 12], prompt_text="chosen", raw_shift=0.1)
    result = build_generation_similarity_inputs(
        payload_a,
        payload_b,
        representation="logits",
        alignment_mode="same_prefix_forcing",
        sample_mode="flatten_all_valid_positions",
        layer_mode="pairwise_all",
        readout_mode="model_norm",
        prompt_text="chosen",
    )
    assert result.prompt_text == "chosen"
    assert result.prompt_id == "7"


def test_generation_similarity_same_prefix_requires_identical_tokens() -> None:
    payload_a = _direct_payload(tokens=[10, 11, 12], raw_shift=0.0)
    payload_b = _direct_payload(tokens=[10, 99, 12], raw_shift=0.1)
    with pytest.raises(ValueError, match="identical visible token ids"):
        build_generation_similarity_inputs(
            payload_a,
            payload_b,
            representation="logits",
            alignment_mode="same_prefix_forcing",
            sample_mode="flatten_all_valid_positions",
            layer_mode="pairwise_all",
            readout_mode="model_norm",
        )


def test_generation_similarity_own_trajectory_uses_minimum_valid_length() -> None:
    payload_a = _direct_payload(tokens=[10, 11, 12, 13], raw_shift=0.0)
    payload_b = _direct_payload(tokens=[20, 21, 22], raw_shift=0.1)
    result = build_generation_similarity_inputs(
        payload_a,
        payload_b,
        representation="logits",
        alignment_mode="own_trajectory",
        sample_mode="flatten_all_valid_positions",
        layer_mode="pairwise_all",
        readout_mode="model_norm",
    )
    assert result.layers_a[0].shape[0] == 3
    assert result.layers_b[0].shape[0] == 3


def test_run_generation_similarity_pairwise_all_returns_matrix_result() -> None:
    payload_a = _direct_payload(tokens=[10, 11, 12], raw_shift=0.0)
    payload_b = _direct_payload(tokens=[10, 11, 12], raw_shift=0.2)
    result = run_generation_similarity(
        payload_a,
        payload_b,
        representation="logits",
        metric="js_similarity",
        alignment_mode="same_prefix_forcing",
        sample_mode="flatten_all_valid_positions",
        layer_mode="pairwise_all",
        readout_mode="model_norm",
        num_permutations=8,
        seed=0,
    )
    assert result.artifact_family == "generation"
    assert len(result.matrix_results) == 1
    assert tuple(result.matrix_results[0].raw_matrix.shape) == (2, 2)

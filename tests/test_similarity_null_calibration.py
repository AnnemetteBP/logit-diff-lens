from __future__ import annotations

import torch

import pytest

from logit_diff_lens.schemas import BackendMetadata, PromptDecodeArtifact, PromptLayerRecord
from logit_diff_lens.similarity import (
    build_prompt_similarity_inputs,
    calibrate_layer_pair_similarity,
    calibrate_scalar_similarity,
    js_similarity_from_logits,
    linear_cka,
    run_prompt_similarity,
    topk_overlap_similarity,
)


def _backend_metadata() -> BackendMetadata:
    return BackendMetadata(
        model_backend="transformers",
        activation_backend="wrapper",
        decode_backend="wrapper_utils",
        device_policy="follow_lm_head",
        dtype_compute="torch.float32",
        dtype_storage="torch.float32@cpu",
        quantization="none",
        device_map="single_device",
    )


def _make_prompt_artifact(
    *,
    token_ids: list[int],
    hidden_shift: float = 0.0,
    raw_shift: float = 0.0,
    prompt_text: str = "demo",
) -> PromptDecodeArtifact:
    tokens = torch.tensor([token_ids], dtype=torch.long)
    mask = torch.ones_like(tokens)
    seq = len(token_ids)
    hidden0 = torch.arange(seq * 3, dtype=torch.float32).reshape(1, seq, 3) + hidden_shift
    hidden1 = hidden0 + 1.0
    logits0 = torch.arange(seq * 5, dtype=torch.float32).reshape(1, seq, 5) + raw_shift
    logits1 = logits0 + 0.5
    records = [
        PromptLayerRecord(
            layer_index=-1,
            layer_name="embedding",
            tokens=tokens.clone(),
            token_text=[f"tok{v}" for v in token_ids],
            attention_mask=mask.clone(),
            hidden=hidden0.clone(),
            logits_raw=logits0.clone(),
            logits_model_norm=logits0.clone() + 0.1,
        ),
        PromptLayerRecord(
            layer_index=0,
            layer_name="layer_00",
            tokens=tokens.clone(),
            token_text=[f"tok{v}" for v in token_ids],
            attention_mask=mask.clone(),
            hidden=hidden1.clone(),
            logits_raw=logits1.clone(),
            logits_model_norm=logits1.clone() + 0.1,
        ),
    ]
    return PromptDecodeArtifact(
        prompt_text=prompt_text,
        prompt_formatted=prompt_text,
        token_ids=tokens,
        token_text=[f"tok{v}" for v in token_ids],
        attention_mask=mask,
        layer_records=records,
        backend_metadata=_backend_metadata(),
        lens_modes=["raw", "model_norm"],
    )


def test_linear_cka_is_bounded() -> None:
    X = torch.randn(20, 8)
    Y = torch.randn(20, 6)
    score = linear_cka(X, Y)
    assert 0.0 <= score <= 1.0


def test_js_similarity_is_bounded() -> None:
    A = torch.randn(12, 17)
    B = torch.randn(12, 17)
    score = js_similarity_from_logits(A, B)
    assert 0.0 <= score <= 1.0


def test_topk_overlap_is_bounded() -> None:
    A = torch.randn(9, 20)
    B = torch.randn(9, 20)
    score = topk_overlap_similarity(A, B, k=5)
    assert 0.0 <= score <= 1.0


def test_scalar_calibration_detects_identity_signal() -> None:
    X = torch.randn(24, 5)
    result = calibrate_scalar_similarity(
        X,
        X.clone(),
        linear_cka,
        metric_name="linear_cka",
        representation_kind="hidden",
        num_permutations=64,
        seed=0,
    )
    assert result.raw_similarity > 0.99
    assert result.p_value <= 0.05
    assert 0.0 <= result.null_percentile <= 1.0


def test_scalar_group_permutation_requires_multiple_groups() -> None:
    X = torch.randn(6, 4)
    Y = X.clone()
    with pytest.raises(ValueError, match="at least two groups"):
        calibrate_scalar_similarity(
            X,
            Y,
            linear_cka,
            metric_name="linear_cka",
            representation_kind="hidden",
            num_permutations=8,
            permutation_unit="group",
            group_ids=torch.zeros(6, dtype=torch.long),
        )


def test_scalar_calibration_rejects_nonfinite() -> None:
    X = torch.randn(10, 4)
    Y = X.clone()
    Y[0, 0] = float("nan")
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        calibrate_scalar_similarity(
            X,
            Y,
            linear_cka,
            metric_name="linear_cka",
            representation_kind="hidden",
            num_permutations=8,
        )


def test_scalar_calibration_rejects_sample_mismatch() -> None:
    X = torch.randn(10, 4)
    Y = torch.randn(9, 4)
    with pytest.raises(ValueError, match="sample mismatch"):
        calibrate_scalar_similarity(
            X,
            Y,
            linear_cka,
            metric_name="linear_cka",
            representation_kind="hidden",
            num_permutations=8,
        )


def test_layer_pair_calibration_shapes() -> None:
    X0 = torch.randn(18, 4)
    X1 = X0 + 0.1
    Y0 = X0.clone()
    Y1 = X1.clone()
    result = calibrate_layer_pair_similarity(
        [X0, X1],
        [Y0, Y1],
        linear_cka,
        metric_name="linear_cka",
        representation_kind="hidden",
        layer_indices_a=[-1, 0],
        layer_indices_b=[-1, 0],
        num_permutations=16,
        seed=0,
    )
    assert tuple(result.raw_matrix.shape) == (2, 2)
    assert tuple(result.calibrated_matrix.shape) == (2, 2)
    assert tuple(result.p_value_matrix.shape) == (2, 2)
    assert tuple(result.adjusted_p_value_matrix.shape) == (2, 2)
    assert tuple(result.null_percentile_matrix.shape) == (2, 2)
    assert result.metadata["multiple_testing_method"] == "fdr_bh"


def test_adjusted_p_values_dominate_raw_p_values() -> None:
    X0 = torch.randn(18, 4)
    X1 = X0 + 0.1
    Y0 = X0.clone()
    Y1 = X1.clone()
    result = calibrate_layer_pair_similarity(
        [X0, X1],
        [Y0, Y1],
        linear_cka,
        metric_name="linear_cka",
        representation_kind="hidden",
        layer_indices_a=[-1, 0],
        layer_indices_b=[-1, 0],
        num_permutations=16,
        multiple_testing_method="holm",
        seed=0,
    )
    assert torch.all(result.adjusted_p_value_matrix >= result.p_value_matrix)


def test_prompt_similarity_alignment_requires_identical_tokens_for_same_token_ids() -> None:
    artifact_a = _make_prompt_artifact(token_ids=[1, 2, 3], prompt_text="same")
    artifact_b = _make_prompt_artifact(token_ids=[1, 9, 3], prompt_text="same")
    with pytest.raises(ValueError, match="identical token ids"):
        build_prompt_similarity_inputs(
            artifact_a,
            artifact_b,
            representation="hidden",
            alignment_mode="same_token_ids",
            sample_mode="flatten_all_valid_positions",
            layer_mode="pairwise_all",
        )


def test_prompt_similarity_inputs_flatten_hidden() -> None:
    artifact_a = _make_prompt_artifact(token_ids=[1, 2, 3], prompt_text="same")
    artifact_b = _make_prompt_artifact(token_ids=[1, 2, 3], hidden_shift=0.25, prompt_text="same")
    result = build_prompt_similarity_inputs(
        artifact_a,
        artifact_b,
        representation="hidden",
        alignment_mode="same_token_ids",
        sample_mode="flatten_all_valid_positions",
        layer_mode="pairwise_all",
    )
    assert result.layer_indices_a == [-1, 0]
    assert result.layers_a[0].shape == (3, 3)
    assert result.layers_b[0].shape == (3, 3)
    assert torch.equal(result.sample_group_ids, torch.zeros(3, dtype=torch.long))


def test_prompt_similarity_inputs_flatten_logits_model_norm() -> None:
    artifact_a = _make_prompt_artifact(token_ids=[1, 2, 3], prompt_text="same")
    artifact_b = _make_prompt_artifact(token_ids=[1, 2, 3], raw_shift=0.2, prompt_text="same")
    result = build_prompt_similarity_inputs(
        artifact_a,
        artifact_b,
        representation="logits",
        alignment_mode="same_token_ids",
        sample_mode="flatten_all_valid_positions",
        layer_mode="pairwise_all",
        readout_mode="model_norm",
    )
    assert result.readout_mode == "model_norm"
    assert result.layers_a[0].shape == (3, 5)


def test_prompt_similarity_bundle_builds_group_ids() -> None:
    artifact_a0 = _make_prompt_artifact(token_ids=[1, 2, 3], prompt_text="p0")
    artifact_b0 = _make_prompt_artifact(token_ids=[1, 2, 3], hidden_shift=0.2, prompt_text="p0")
    artifact_a0.prompt_id = "0"
    artifact_b0.prompt_id = "0"
    artifact_a1 = _make_prompt_artifact(token_ids=[4, 5, 6], prompt_text="p1")
    artifact_b1 = _make_prompt_artifact(token_ids=[4, 5, 6], hidden_shift=0.3, prompt_text="p1")
    artifact_a1.prompt_id = "1"
    artifact_b1.prompt_id = "1"
    result = build_prompt_similarity_inputs(
        {"artifacts": [artifact_a0.to_dict(), artifact_a1.to_dict()]},
        {"artifacts": [artifact_b0.to_dict(), artifact_b1.to_dict()]},
        representation="hidden",
        alignment_mode="same_token_ids",
        sample_mode="flatten_all_valid_positions",
        layer_mode="pairwise_all",
    )
    assert result.metadata["num_examples"] == 2
    assert result.layers_a[0].shape == (6, 3)
    assert torch.equal(result.sample_group_ids, torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long))


def test_run_prompt_similarity_bundle_group_permutation_records_metadata() -> None:
    artifact_a0 = _make_prompt_artifact(token_ids=[1, 2, 3], prompt_text="p0")
    artifact_b0 = _make_prompt_artifact(token_ids=[1, 2, 3], hidden_shift=0.2, prompt_text="p0")
    artifact_a0.prompt_id = "0"
    artifact_b0.prompt_id = "0"
    artifact_a1 = _make_prompt_artifact(token_ids=[4, 5, 6], prompt_text="p1")
    artifact_b1 = _make_prompt_artifact(token_ids=[4, 5, 6], hidden_shift=0.3, prompt_text="p1")
    artifact_a1.prompt_id = "1"
    artifact_b1.prompt_id = "1"
    run = run_prompt_similarity(
        {"artifacts": [artifact_a0.to_dict(), artifact_a1.to_dict()]},
        {"artifacts": [artifact_b0.to_dict(), artifact_b1.to_dict()]},
        representation="hidden",
        metric="linear_cka",
        layer_mode="pairwise_all",
        permutation_unit="group",
        num_permutations=8,
        seed=0,
    )
    matrix = run.matrix_results[0]
    assert run.metadata["permutation_unit"] == "group"
    assert matrix.metadata["permutation_unit"] == "group"

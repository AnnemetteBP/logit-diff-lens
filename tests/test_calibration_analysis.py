from __future__ import annotations

import torch

from logit_diff_lens.calibration.generation import run_generation_reference_calibration
from logit_diff_lens.calibration.metrics import compute_expected_calibration_error
from logit_diff_lens.calibration.prompt import run_prompt_reference_calibration
from logit_diff_lens.schemas import BackendMetadata, PromptDecodeArtifact, PromptLayerRecord


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
    logits_shift: float = 0.0,
    prompt_text: str = "demo",
    prompt_id: str | None = "1",
) -> PromptDecodeArtifact:
    tokens = torch.tensor([token_ids], dtype=torch.long)
    mask = torch.ones_like(tokens)
    seq = len(token_ids)
    hidden0 = torch.arange(seq * 3, dtype=torch.float32).reshape(1, seq, 3)
    hidden1 = hidden0 + 1.0
    logits0 = torch.tensor(
        [[[5.0, 1.0, 0.0], [1.0, 5.0, 0.0], [1.0, 0.0, 5.0]]],
        dtype=torch.float32,
    ) + logits_shift
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
            logits_model_norm=logits0.clone(),
        ),
        PromptLayerRecord(
            layer_index=0,
            layer_name="layer_00",
            tokens=tokens.clone(),
            token_text=[f"tok{v}" for v in token_ids],
            attention_mask=mask.clone(),
            hidden=hidden1.clone(),
            logits_raw=logits1.clone(),
            logits_model_norm=logits1.clone(),
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
        prompt_id=prompt_id,
    )


def _generation_payload(*, prompt_id: int = 0, shift: float = 0.0) -> dict:
    tokens = torch.tensor([[10, 11, 12]], dtype=torch.long)
    mask = torch.ones_like(tokens)
    logits = torch.tensor(
        [[[5.0, 1.0, 0.0], [1.0, 5.0, 0.0], [1.0, 0.0, 5.0]]],
        dtype=torch.float32,
    ) + shift
    hidden = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    rows = []
    for layer_index in (-1, 0):
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_text": "demo",
                "prompt_formatted": "demo",
                "batch_index": 0,
                "step": 1,
                "layer_index": layer_index,
                "layer_name": "embedding" if layer_index == -1 else "layer_00",
                "tokens": tokens.clone(),
                "attention_mask": mask.clone(),
                "hidden_raw": hidden.clone() + (0.0 if layer_index == -1 else 1.0),
                "hidden_model_norm": hidden.clone() + (0.0 if layer_index == -1 else 1.0),
                "logits_raw": logits.clone() + (0.0 if layer_index == -1 else 0.5),
                "logits_model_norm": logits.clone() + (0.0 if layer_index == -1 else 0.5),
            }
        )
    return {
        "rows": rows,
        "force_include_input": True,
        "force_include_output": False,
    }


def test_expected_calibration_error_zero_for_perfectly_calibrated_two_point_case() -> None:
    stats = compute_expected_calibration_error(
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        torch.tensor([0.0, 1.0], dtype=torch.float32),
        num_bins=2,
        binning="equal_width",
    )
    assert stats.sample_count == 2
    assert stats.ece == 0.0


def test_prompt_reference_calibration_returns_layer_position_matrices() -> None:
    artifact_a = _make_prompt_artifact(token_ids=[1, 2, 3], logits_shift=0.0)
    artifact_b = _make_prompt_artifact(token_ids=[1, 2, 3], logits_shift=0.2)
    result = run_prompt_reference_calibration(
        artifact_a,
        artifact_b,
        readout_mode_eval="model_norm",
        readout_mode_reference="model_norm",
        top_k_values=[2],
        num_bootstrap=8,
        seed=0,
    )
    assert result.artifact_family == "prompt"
    assert result.matrix_results[0].metric_name == "top1_ece"
    assert tuple(result.matrix_results[0].value_matrix.shape) == (2, 3)
    assert len(result.summary_results) >= 2


def test_generation_reference_calibration_accepts_direct_payloads() -> None:
    result = run_generation_reference_calibration(
        _generation_payload(prompt_id=7, shift=0.0),
        _generation_payload(prompt_id=7, shift=0.2),
        readout_mode_eval="model_norm",
        readout_mode_reference="model_norm",
        top_k_values=[2],
        num_bootstrap=8,
        seed=0,
    )
    assert result.artifact_family == "generation"
    assert tuple(result.matrix_results[0].value_matrix.shape) == (2, 3)

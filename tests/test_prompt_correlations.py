from __future__ import annotations

import torch

from logit_diff_lens.correlations.prompt import run_prompt_correlations
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


def _artifact(*, scale: float, prompt_id: str) -> PromptDecodeArtifact:
    token_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    attention_mask = torch.ones_like(token_ids)
    seq = token_ids.shape[1]
    hidden = torch.arange(seq * 3, dtype=torch.float32).reshape(1, seq, 3)
    base_logits = torch.tensor(
        [
            [[5.0, 1.0, 0.0], [4.0, 2.0, 0.0], [3.0, 3.0, 0.0], [2.0, 4.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    records = []
    for layer_index in (-1, 0):
        logits = base_logits * scale + (0.3 if layer_index == 0 else 0.0)
        records.append(
            PromptLayerRecord(
                layer_index=layer_index,
                layer_name="embedding" if layer_index == -1 else "layer_00",
                tokens=token_ids.clone(),
                token_text=[f"tok{i}" for i in token_ids[0].tolist()],
                attention_mask=attention_mask.clone(),
                hidden=hidden.clone() + (1.0 if layer_index == 0 else 0.0),
                logits_raw=logits.clone(),
                logits_model_norm=logits.clone(),
            )
        )
    return PromptDecodeArtifact(
        prompt_text="demo",
        prompt_formatted="demo",
        token_ids=token_ids,
        token_text=[f"tok{i}" for i in token_ids[0].tolist()],
        attention_mask=attention_mask,
        layer_records=records,
        backend_metadata=_backend_metadata(),
        lens_modes=["raw", "model_norm"],
        prompt_id=prompt_id,
    )


def test_run_prompt_correlations_returns_matrix_result() -> None:
    result = run_prompt_correlations(
        {"artifacts": [_artifact(scale=1.0, prompt_id="1").to_dict(), _artifact(scale=1.2, prompt_id="2").to_dict()]},
        {"artifacts": [_artifact(scale=0.9, prompt_id="1").to_dict(), _artifact(scale=1.1, prompt_id="2").to_dict()]},
        readout_mode="model_norm",
        metrics=["top1_confidence"],
        min_samples=4,
    )
    assert result.artifact_family == "prompt"
    assert len(result.matrix_results) == 1
    matrix = result.matrix_results[0]
    assert matrix.metric_name == "top1_confidence"
    assert tuple(matrix.pearson_r_matrix.shape) == (2, 2)
    assert tuple(matrix.spearman_r_matrix.shape) == (2, 2)
    assert torch.isfinite(matrix.sample_count_matrix).all()

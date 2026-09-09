from __future__ import annotations

import torch

from logit_diff_lens.prisms.prompt_diff import build_prompt_diff_prism_artifact
from logit_diff_lens.schemas import BackendMetadata, PromptDecodeArtifact, PromptLayerRecord


def _backend() -> BackendMetadata:
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


def _artifact(scale: float, *, bias: float) -> PromptDecodeArtifact:
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(token_ids)
    hidden = torch.zeros((1, 3, 2), dtype=torch.float32)
    # vocab size 3 so selected top-k is deterministic on the middle predictive position
    embedding_linear = torch.tensor(
        [[[1.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=torch.float32,
    ) * scale
    attention_linear = torch.tensor(
        [[[0.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0]]],
        dtype=torch.float32,
    ) * scale
    mlp_linear = torch.tensor(
        [[[0.0, 0.0, 0.25], [0.0, 0.0, 0.25], [0.0, 0.0, 0.25]]],
        dtype=torch.float32,
    ) * scale
    bias_vec = torch.full((1, 3, 3), float(bias), dtype=torch.float32)
    embedding_logits = embedding_linear + bias_vec
    attention_logits = attention_linear + bias_vec
    mlp_logits = mlp_linear + bias_vec
    residual_logits = embedding_linear + attention_linear + mlp_linear + bias_vec
    return PromptDecodeArtifact(
        prompt_text="demo",
        prompt_formatted="demo",
        token_ids=token_ids,
        token_text=["a", "b", "c"],
        attention_mask=attention_mask,
        layer_records=[
            PromptLayerRecord(
                layer_index=-1,
                layer_name="embedding",
                tokens=token_ids,
                token_text=["a", "b", "c"],
                attention_mask=attention_mask,
                hidden=hidden,
                logits_model_norm=embedding_logits,
                logits_raw=embedding_logits,
            ),
            PromptLayerRecord(
                layer_index=0,
                layer_name="layer_0",
                tokens=token_ids,
                token_text=["a", "b", "c"],
                attention_mask=attention_mask,
                hidden=hidden,
                logits_model_norm=residual_logits,
                logits_raw=residual_logits,
                attention_logits_model_norm=attention_logits,
                mlp_logits_model_norm=mlp_logits,
                attention_logits_raw=attention_logits,
                mlp_logits_raw=mlp_logits,
            ),
            PromptLayerRecord(
                layer_index=1,
                layer_name="output",
                tokens=token_ids,
                token_text=["a", "b", "c"],
                attention_mask=attention_mask,
                hidden=hidden,
                logits_model_norm=residual_logits,
                logits_raw=residual_logits,
            ),
        ],
        backend_metadata=_backend(),
        lens_modes=["raw", "model_norm"],
        prompt_id="1",
        metadata={
            "force_include_input": True,
            "force_include_output": True,
            "normalize_embedding_for_readout": False,
        },
    )


def test_build_prompt_diff_prism_artifact_returns_token_level_delta_heatmap() -> None:
    base = _artifact(1.0, bias=0.1)
    ft = _artifact(2.0, bias=0.3)
    prism = build_prompt_diff_prism_artifact(ft, base, readout_mode="raw", top_k=3, position_index=1)
    assert prism.delta_definition == "ft - base"
    assert prism.position_index == 1
    assert prism.component_labels == ["embedding", "attn_0", "mlp_0", "full_layer_0", "output_l+1"]
    # Token order should follow largest absolute total delta: embedding token, attention token, mlp token.
    assert prism.selected_token_ids.tolist() == [0, 1, 2]
    expected_rows = torch.tensor(
        [
            [1.7, 0.2, 0.2],   # embedding delta with bias folded in
            [0.0, 0.5, 0.0],   # attn delta
            [0.0, 0.0, 0.25],  # mlp delta
            [1.7, 0.7, 0.45],  # full layer cumulative / total delta
            [1.7, 0.7, 0.45],  # actual output L+1
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(prism.contribution_matrix, expected_rows, atol=1e-6)
    assert torch.allclose(prism.summed_component_logits, prism.total_delta_logits, atol=1e-6)
    assert torch.allclose(prism.reconstruction_gap, torch.zeros_like(prism.reconstruction_gap), atol=1e-6)


def test_build_prompt_diff_prism_artifact_supports_model_norm() -> None:
    base = _artifact(1.0, bias=0.1)
    ft = _artifact(2.0, bias=0.3)
    prism = build_prompt_diff_prism_artifact(ft, base, readout_mode="model_norm", top_k=3, position_index=1)
    assert prism.readout_mode == "model_norm"
    assert prism.component_labels[-1] == "output_l+1"

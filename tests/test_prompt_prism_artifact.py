from __future__ import annotations

import torch

from logit_diff_lens.prisms.prompt import build_prompt_prism_artifact
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


def test_build_prompt_prism_artifact_collects_component_trajectories() -> None:
    token_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(token_ids)
    hidden = torch.zeros((1, 3, 2), dtype=torch.float32)
    embedding_logits = torch.tensor(
        [[[1.0, 0.0], [1.5, 0.0], [2.0, 0.0]]],
        dtype=torch.float32,
    )
    attention_logits = torch.tensor(
        [[[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]]],
        dtype=torch.float32,
    )
    mlp_logits = torch.tensor(
        [[[0.25, 0.0], [0.25, 0.0], [0.25, 0.0]]],
        dtype=torch.float32,
    )
    residual_logits = embedding_logits + attention_logits + mlp_logits
    artifact = PromptDecodeArtifact(
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
    )
    prism = build_prompt_prism_artifact(artifact, readout_mode="model_norm", top_k=2)
    assert prism.position_indices == [0, 1, 2]
    assert prism.component_labels == ["embedding", "attn_0", "mlp_0", "full_layer_0", "output_l+1"]
    assert prism.selected_token_ids.tolist() == [0, 1]
    assert prism.contribution_tensor.shape == (3, 5, 2)
    expected_position_0 = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.0],
            [0.25, 0.0],
            [1.75, 0.0],
            [1.75, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(prism.contribution_tensor[0], expected_position_0, atol=1e-6)
    assert torch.allclose(prism.output_token_logits, prism.additive_component_logits, atol=1e-6)
    assert torch.allclose(prism.output_residual_gap, torch.zeros_like(prism.output_residual_gap), atol=1e-6)

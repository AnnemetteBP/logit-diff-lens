from __future__ import annotations

import pytest
import torch

from logit_diff_lens.cli import plot_prompt_prism
from logit_diff_lens.pair_validation import validate_prompt_artifact_pair
from logit_diff_lens.schemas import BackendMetadata, PromptDecodeArtifact, PromptLayerRecord
from pipelines import plot_adl_heatmap, plot_generation_heatmap, plot_prompt_heatmap


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


def _artifact(
    *,
    token_ids: list[int] | None = None,
    layer_names: list[str] | None = None,
    force_include_input: bool = True,
    force_include_output: bool = True,
    normalize_embedding_for_readout: bool = False,
) -> PromptDecodeArtifact:
    token_ids = token_ids or [1, 2, 3]
    layer_names = layer_names or ["embedding", "layer_00", "output"]
    tokens = torch.tensor([token_ids], dtype=torch.long)
    mask = torch.ones_like(tokens)
    hidden = torch.zeros((1, len(token_ids), 2), dtype=torch.float32)
    logits = torch.zeros((1, len(token_ids), 4), dtype=torch.float32)
    records = [
        PromptLayerRecord(
            layer_index=-1,
            layer_name=layer_names[0],
            tokens=tokens.clone(),
            token_text=[f"tok{i}" for i in token_ids],
            attention_mask=mask.clone(),
            hidden=hidden.clone(),
            logits_raw=logits.clone(),
            logits_model_norm=logits.clone(),
        ),
        PromptLayerRecord(
            layer_index=0,
            layer_name=layer_names[1],
            tokens=tokens.clone(),
            token_text=[f"tok{i}" for i in token_ids],
            attention_mask=mask.clone(),
            hidden=hidden.clone(),
            logits_raw=logits.clone(),
            logits_model_norm=logits.clone(),
            attention_logits_raw=logits.clone(),
            attention_logits_model_norm=logits.clone(),
            mlp_logits_raw=logits.clone(),
            mlp_logits_model_norm=logits.clone(),
        ),
        PromptLayerRecord(
            layer_index=1,
            layer_name=layer_names[2],
            tokens=tokens.clone(),
            token_text=[f"tok{i}" for i in token_ids],
            attention_mask=mask.clone(),
            hidden=hidden.clone(),
            logits_raw=logits.clone(),
            logits_model_norm=logits.clone(),
        ),
    ]
    return PromptDecodeArtifact(
        prompt_text="demo",
        prompt_formatted="demo",
        token_ids=tokens,
        token_text=[f"tok{i}" for i in token_ids],
        attention_mask=mask,
        layer_records=records,
        backend_metadata=_backend(),
        lens_modes=["raw", "model_norm"],
        metadata={
            "force_include_input": force_include_input,
            "force_include_output": force_include_output,
            "normalize_embedding_for_readout": normalize_embedding_for_readout,
        },
    )


def test_prompt_prism_plotter_rejects_mixed_modes() -> None:
    with pytest.raises(ValueError, match="either --input-path or both --prism-a and --prism-b"):
        plot_prompt_prism.main(
            [
                "--input-path",
                "tmp/prism.pt",
                "--prism-a",
                "tmp/a.pt",
                "--prism-b",
                "tmp/b.pt",
                "--output-path",
                "tmp/out.pdf",
            ]
        )


def test_prompt_heatmap_rejects_saved_plus_live_inputs() -> None:
    with pytest.raises(ValueError, match="Saved prompt heatmap mode cannot be mixed"):
        plot_prompt_heatmap.main(
            [
                "--input-path",
                "tmp/in.pt",
                "--output-path",
                "tmp/out.pdf",
                "--model-name",
                "EleutherAI/pythia-70m-deduped",
            ]
        )


def test_generation_heatmap_rejects_saved_plus_live_inputs() -> None:
    with pytest.raises(ValueError, match="Saved generation heatmap mode cannot be mixed"):
        plot_generation_heatmap.main(
            [
                "--input-path",
                "tmp/in.json",
                "--output-path",
                "tmp/out.pdf",
                "--prompt",
                "demo",
            ]
        )


def test_adl_heatmap_rejects_saved_plus_live_inputs() -> None:
    with pytest.raises(ValueError, match="Saved ADL heatmap mode cannot be mixed"):
        plot_adl_heatmap.main(
            [
                "--input-path",
                "tmp/in.pt",
                "--output-path",
                "tmp/out.pdf",
                "--model-name",
                "EleutherAI/pythia-70m-deduped",
                "--prompt",
                "demo",
            ]
        )


def test_pair_validation_rejects_layer_name_mismatch() -> None:
    artifact_a = _artifact(layer_names=["embedding", "layer_00", "output"])
    artifact_b = _artifact(layer_names=["embedding", "layer_zero", "output"])
    with pytest.raises(ValueError, match="Layer name mismatch"):
        validate_prompt_artifact_pair(artifact_a, artifact_b, readout_mode="raw")


def test_pair_validation_rejects_token_length_mismatch() -> None:
    artifact_a = _artifact(token_ids=[1, 2, 3])
    artifact_b = _artifact(token_ids=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="token shapes differ"):
        validate_prompt_artifact_pair(artifact_a, artifact_b, readout_mode="raw")


def test_pair_validation_rejects_missing_component_logits() -> None:
    artifact_a = _artifact()
    artifact_b = _artifact()
    artifact_b.layer_records[1].attention_logits_raw = None
    with pytest.raises(ValueError, match="missing attention component logits"):
        validate_prompt_artifact_pair(
            artifact_a,
            artifact_b,
            readout_mode="raw",
            require_component_logits=("attention", "mlp"),
        )


def test_pair_validation_rejects_nonfinite_logits() -> None:
    artifact_a = _artifact()
    artifact_b = _artifact()
    artifact_b.layer_records[1].logits_raw[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        validate_prompt_artifact_pair(artifact_a, artifact_b, readout_mode="raw")


def test_pair_validation_requires_force_include_metadata() -> None:
    artifact_a = _artifact(force_include_output=False)
    artifact_b = _artifact(force_include_output=False)
    with pytest.raises(ValueError, match="force_include_output"):
        validate_prompt_artifact_pair(
            artifact_a,
            artifact_b,
            readout_mode="raw",
            require_force_include_output=True,
        )

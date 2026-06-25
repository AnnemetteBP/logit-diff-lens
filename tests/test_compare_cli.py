from __future__ import annotations

from pathlib import Path

import torch

from logit_diff_lens.diffing.io import load_comparison_artifact, save_prompt_decode_artifact
from logit_diff_lens.logit_lens.compare import main
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


def _make_artifact(offset: float) -> PromptDecodeArtifact:
    record = PromptLayerRecord(
        layer_index=-1,
        layer_name="embedding",
        tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
        token_text=["tok1", "tok2", "tok3"],
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        hidden=torch.full((1, 3, 2), offset, dtype=torch.float32),
        logits_raw=torch.full((1, 3, 5), offset, dtype=torch.float32),
        logits_model_norm=torch.full((1, 3, 5), offset + 0.5, dtype=torch.float32),
    )
    return PromptDecodeArtifact(
        prompt_text="demo",
        prompt_formatted="demo",
        token_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        token_text=["tok1", "tok2", "tok3"],
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        layer_records=[record],
        backend_metadata=_backend_metadata(),
        lens_modes=["raw", "model_norm"],
    )


def test_compare_cli_writes_comparison_artifact(tmp_path: Path) -> None:
    base_path = tmp_path / "base.pt"
    ft_path = tmp_path / "ft.pt"
    comparison_path = tmp_path / "comparison.pt"

    save_prompt_decode_artifact(_make_artifact(0.0), base_path)
    save_prompt_decode_artifact(_make_artifact(1.0), ft_path)

    main(
        [
            "--ft-artifact",
            str(ft_path),
            "--base-artifact",
            str(base_path),
            "--comparison-output",
            str(comparison_path),
            "--readout-mode",
            "model_norm",
            "--metric",
            "jsd_ft_base",
        ]
    )

    comparison = load_comparison_artifact(comparison_path)
    assert comparison["operand_order"] == "ft_minus_base"
    assert comparison["readout_mode"] == "model_norm"
    assert len(comparison["layer_results"]) == 1

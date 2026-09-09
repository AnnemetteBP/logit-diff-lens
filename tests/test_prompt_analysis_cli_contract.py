from __future__ import annotations

from pathlib import Path

import pytest
import torch

from logit_diff_lens.cli import calibration_prompt, correlation_prompt, prompt_diff_prism, prompt_prism, similarity_prompt
from logit_diff_lens.cli.prompt_analysis_utils import resolve_prompt_artifact
from logit_diff_lens.diffing.io import save_prompt_decode_artifact, save_prompt_decode_artifact_bundle
from logit_diff_lens.logit_lens import compare
from logit_diff_lens.schemas import BackendMetadata, PromptDecodeArtifact, PromptLayerRecord


def _artifact() -> PromptDecodeArtifact:
    backend = BackendMetadata(
        model_backend="transformers",
        activation_backend="wrapper",
        decode_backend="wrapper_utils",
        device_policy="follow_lm_head",
        dtype_compute="torch.float32",
        dtype_storage="torch.float32@cpu",
        quantization="none",
        device_map="single_device",
    )
    record = PromptLayerRecord(
        layer_index=-1,
        layer_name="embedding",
        tokens=torch.tensor([[1, 2]], dtype=torch.long),
        token_text=["a", "b"],
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        hidden=torch.zeros((1, 2, 2), dtype=torch.float32),
        logits_raw=torch.zeros((1, 2, 3), dtype=torch.float32),
        logits_model_norm=torch.zeros((1, 2, 3), dtype=torch.float32),
    )
    return PromptDecodeArtifact(
        prompt_text="demo",
        prompt_formatted="demo",
        token_ids=torch.tensor([[1, 2]], dtype=torch.long),
        token_text=["a", "b"],
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        layer_records=[record],
        backend_metadata=backend,
        lens_modes=["raw", "model_norm"],
    )


def test_prompt_prism_rejects_mixed_saved_and_live_modes() -> None:
    with pytest.raises(ValueError, match="cannot be mixed"):
        prompt_prism.main(
            [
                "--input-path",
                "tmp/saved.pt",
                "--output-path",
                "tmp/out.pt",
                "--model-name",
                "EleutherAI/pythia-70m-deduped",
            ]
        )


def test_prompt_prism_live_mode_collects_before_build(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    monkeypatch.setattr(prompt_prism, "build_prompt_wrapper", lambda **kwargs: object())
    monkeypatch.setattr(prompt_prism, "collect_prompt_artifacts_live", lambda *args, **kwargs: [_artifact()])
    monkeypatch.setattr(prompt_prism, "build_prompt_prism_artifact", lambda source, **kwargs: {"source": source, **kwargs})

    def _save(artifact, path):
        captured["artifact"] = artifact
        captured["path"] = path

    monkeypatch.setattr(prompt_prism, "save_prompt_prism_artifact", _save)
    output_path = tmp_path / "prism.pt"
    prompt_prism.main(
        [
            "--model-name",
            "EleutherAI/pythia-70m-deduped",
            "--prompt",
            "demo",
            "--output-path",
            str(output_path),
        ]
    )
    assert isinstance(captured["artifact"]["source"], PromptDecodeArtifact)
    assert Path(captured["path"]) == output_path


def test_prompt_diff_prism_accepts_saved_alias_pair(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    monkeypatch.setattr(prompt_diff_prism, "build_prompt_diff_prism_artifact", lambda ft, base, **kwargs: {"ft": ft, "base": base, **kwargs})
    monkeypatch.setattr(prompt_diff_prism, "save_prompt_diff_prism_heatmap_artifact", lambda artifact, path: captured.update({"artifact": artifact, "path": path}))
    output_path = tmp_path / "diff.pt"
    prompt_diff_prism.main(
        [
            "--base-artifact",
            "tmp/base.pt",
            "--comparison-artifact",
            "tmp/comp.pt",
            "--output-path",
            str(output_path),
        ]
    )
    assert captured["artifact"]["ft"] == "tmp/comp.pt"
    assert captured["artifact"]["base"] == "tmp/base.pt"
    assert Path(captured["path"]) == output_path


def test_prompt_diff_prism_rejects_mixed_saved_and_live_modes() -> None:
    with pytest.raises(ValueError, match="cannot be mixed"):
        prompt_diff_prism.main(
            [
                "--base-artifact",
                "tmp/base.pt",
                "--comparison-artifact",
                "tmp/comp.pt",
                "--output-path",
                "tmp/out.pt",
                "--model-name",
                "EleutherAI/pythia-70m-deduped",
            ]
        )


def test_saved_pair_aliases_work_for_calibration_and_correlation(monkeypatch, tmp_path: Path) -> None:
    seen = {}
    monkeypatch.setattr(calibration_prompt, "run_prompt_reference_calibration", lambda a, b, **kwargs: seen.setdefault("cal", (a, b, kwargs)) or {})
    monkeypatch.setattr(calibration_prompt, "save_calibration_run_artifact", lambda result, path: None)
    monkeypatch.setattr(correlation_prompt, "run_prompt_correlations", lambda a, b, **kwargs: seen.setdefault("corr", (a, b, kwargs)) or {})
    monkeypatch.setattr(correlation_prompt, "save_correlation_run_artifact", lambda result, path: None)

    calibration_prompt.main(
        [
            "--base-artifact",
            "tmp/base.pt",
            "--comparison-artifact",
            "tmp/comp.pt",
            "--output-path",
            str(tmp_path / "cal.pt"),
        ]
    )
    correlation_prompt.main(
        [
            "--base-artifact",
            "tmp/base.pt",
            "--comparison-artifact",
            "tmp/comp.pt",
            "--output-path",
            str(tmp_path / "corr.pt"),
        ]
    )
    assert seen["cal"][0] == "tmp/comp.pt"
    assert seen["cal"][1] == "tmp/base.pt"
    assert seen["corr"][0] == "tmp/comp.pt"
    assert seen["corr"][1] == "tmp/base.pt"


def test_similarity_prompt_accepts_bundle_via_saved_aliases(monkeypatch, tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundle.pt"
    comparison_path = tmp_path / "comp.pt"
    save_prompt_decode_artifact_bundle([_artifact()], bundle_path)
    save_prompt_decode_artifact(_artifact(), comparison_path)

    captured = {}
    monkeypatch.setattr(similarity_prompt, "run_prompt_similarity", lambda a, b, **kwargs: captured.setdefault("pair", (a, b, kwargs)) or {})
    monkeypatch.setattr(similarity_prompt, "save_similarity_run_artifact", lambda result, path: None)

    similarity_prompt.main(
        [
            "--base-artifact",
            str(bundle_path),
            "--comparison-artifact",
            str(comparison_path),
            "--output-path",
            str(tmp_path / "sim.pt"),
            "--representation",
            "hidden",
            "--metric",
            "linear_cka",
        ]
    )
    assert captured["pair"][0] == str(comparison_path)
    assert captured["pair"][1] == str(bundle_path)


def test_resolve_prompt_artifact_can_select_by_prompt_id(tmp_path: Path) -> None:
    first = _artifact()
    second = _artifact()
    first.prompt_id = "first"
    first.prompt_text = "one"
    second.prompt_id = "second"
    second.prompt_text = "two"
    bundle_path = tmp_path / "bundle.pt"
    save_prompt_decode_artifact_bundle([first, second], bundle_path)
    selected = resolve_prompt_artifact(bundle_path, prompt_id="second")
    assert selected.prompt_id == "second"
    assert selected.prompt_text == "two"


def test_compare_cli_can_select_prompt_from_bundle(monkeypatch, tmp_path: Path) -> None:
    first = _artifact()
    second = _artifact()
    first.prompt_id = "first"
    first.prompt_text = "one"
    second.prompt_id = "second"
    second.prompt_text = "two"
    bundle_path = tmp_path / "bundle.pt"
    save_prompt_decode_artifact_bundle([first, second], bundle_path)
    captured = {}

    monkeypatch.setattr(compare, "compare_prompt_artifacts_ft_minus_base", lambda ft, base, **kwargs: captured.setdefault("pair", (ft, base, kwargs)) or {})
    monkeypatch.setattr(compare, "save_comparison_artifact", lambda result, path: None)

    compare.main(
        [
            "--ft-artifact",
            str(bundle_path),
            "--base-artifact",
            str(bundle_path),
            "--comparison-output",
            str(tmp_path / "comparison.pt"),
            "--prompt-id",
            "second",
        ]
    )
    assert captured["pair"][0].prompt_id == "second"
    assert captured["pair"][1].prompt_id == "second"

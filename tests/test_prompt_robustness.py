from __future__ import annotations

from pathlib import Path

import pytest
import torch

from logit_diff_lens.cli.robustness_prompt import main as robustness_cli_main
from logit_diff_lens.robustness import run_prompt_robustness
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
            [[4.5, 1.5, 0.2, -0.1], [4.0, 2.0, 0.2, -0.1], [3.5, 2.5, 0.2, -0.1], [3.0, 3.0, 0.2, -0.1]],
        ],
        dtype=torch.float32,
    )

    records: list[PromptLayerRecord] = []

    embedding_logits = base_logits * scale
    records.append(
        PromptLayerRecord(
            layer_index=-1,
            layer_name="embedding",
            tokens=token_ids.clone(),
            token_text=[f"tok{i}" for i in token_ids[0].tolist()],
            attention_mask=attention_mask.clone(),
            hidden=hidden.clone(),
            logits_raw=embedding_logits.clone(),
            logits_model_norm=(embedding_logits + 0.05).clone(),
            logits_tuned=(embedding_logits + 0.10).clone(),
        )
    )

    running = embedding_logits.clone()
    for layer_index in (0, 1):
        attn = torch.full_like(base_logits, 0.10 * scale * (layer_index + 1))
        mlp = torch.full_like(base_logits, 0.06 * scale * (layer_index + 1))
        running = running + attn + mlp
        records.append(
            PromptLayerRecord(
                layer_index=layer_index,
                layer_name=f"layer_{layer_index:02d}",
                tokens=token_ids.clone(),
                token_text=[f"tok{i}" for i in token_ids[0].tolist()],
                attention_mask=attention_mask.clone(),
                hidden=hidden.clone() + float(layer_index + 1),
                logits_raw=running.clone(),
                logits_model_norm=(running + 0.05).clone(),
                logits_tuned=(running + 0.10).clone(),
                attention_logits_raw=attn.clone(),
                attention_logits_model_norm=(attn + 0.01).clone(),
                mlp_logits_raw=mlp.clone(),
                mlp_logits_model_norm=(mlp + 0.01).clone(),
            )
        )

    records.append(
        PromptLayerRecord(
            layer_index=2,
            layer_name="output",
            tokens=token_ids.clone(),
            token_text=[f"tok{i}" for i in token_ids[0].tolist()],
            attention_mask=attention_mask.clone(),
            hidden=hidden.clone() + 3.0,
            logits_raw=running.clone(),
            logits_model_norm=(running + 0.05).clone(),
            logits_tuned=(running + 0.10).clone(),
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
        lens_modes=["raw", "model_norm", "tuned"],
        prompt_id=prompt_id,
        metadata={
            "force_include_input": True,
            "force_include_output": True,
            "normalize_embedding_for_readout": True,
        },
    )


def test_run_prompt_robustness_returns_profiles_and_agreements() -> None:
    comparison = {
        "artifacts": [
            _artifact(scale=1.10, prompt_id="p0").to_dict(),
            _artifact(scale=1.25, prompt_id="p1").to_dict(),
            _artifact(scale=1.40, prompt_id="p2").to_dict(),
            _artifact(scale=1.55, prompt_id="p3").to_dict(),
        ]
    }
    base = {
        "artifacts": [
            _artifact(scale=1.00, prompt_id="p0").to_dict(),
            _artifact(scale=1.12, prompt_id="p1").to_dict(),
            _artifact(scale=1.24, prompt_id="p2").to_dict(),
            _artifact(scale=1.36, prompt_id="p3").to_dict(),
        ]
    }

    artifact = run_prompt_robustness(
        comparison,
        base,
        include_lens_families=["raw", "model_norm", "tuned", "prisms"],
        prism_readout_mode="raw",
        metric_name="jsd_divergence",
        num_permutations=16,
        num_bootstrap=32,
        alpha=0.05,
        seed=7,
    )

    assert artifact.artifact_family == "prompt"
    assert artifact.metric_name == "jsd_divergence"
    assert len(artifact.profiles) == 8
    assert len(artifact.within_lens_results) == 4
    assert len(artifact.cross_lens_results) == 12
    assert len(artifact.agreement_calibration_results) == 8

    profile_lookup = {(item.lens_family, item.regime_name): item for item in artifact.profiles}
    raw_nc = profile_lookup[("raw", "non_calibrated")]
    prism_c = profile_lookup[("prisms", "calibrated")]

    assert raw_nc.prompt_values.shape == (4, 4)
    assert prism_c.prompt_values.shape == (4, 4)
    assert raw_nc.layer_indices == [-1, 0, 1, 2]
    assert prism_c.layer_indices == [-1, 0, 1, 2]
    assert torch.isfinite(raw_nc.mean_values).all()
    assert torch.isfinite(prism_c.mean_values).all()

    within_raw = next(item for item in artifact.within_lens_results if item.lens_family_a == "raw")
    assert within_raw.regime_name == "non_calibrated"
    assert within_raw.layer_indices == [-1, 0, 1, 2]
    assert tuple(within_raw.pearson_r.shape) == (4,)
    assert tuple(within_raw.spearman_r.shape) == (4,)
    assert torch.isfinite(within_raw.sample_counts).all()

    cross_pair = next(
        item
        for item in artifact.cross_lens_results
        if item.regime_name == "calibrated" and {item.lens_family_a, item.lens_family_b} == {"model_norm", "tuned"}
    )
    assert tuple(cross_pair.pearson_r.shape) == (4,)
    assert tuple(cross_pair.spearman_r.shape) == (4,)
    assert cross_pair.mean_gap >= 0.0

    calib = next(
        item
        for item in artifact.agreement_calibration_results
        if item.lens_family == "raw" and item.regime_name == "non_calibrated"
    )
    assert calib.event_name == "top1_agreement_rate"
    assert tuple(calib.ece_values.shape) == (4,)
    assert tuple(calib.brier_values.shape) == (4,)
    assert torch.isfinite(calib.sample_counts).all()
    assert torch.isfinite(calib.confidence_means).all()
    assert torch.isfinite(calib.outcome_means).all()


def test_run_prompt_robustness_requires_tuned_logits_for_tuned_family() -> None:
    comparison = {"artifacts": [_artifact(scale=1.10, prompt_id="p0").to_dict() for _ in range(4)]}
    base = {"artifacts": [_artifact(scale=1.00, prompt_id="p0").to_dict() for _ in range(4)]}
    for payload in (comparison, base):
        for item in payload["artifacts"]:
            for record in item["layer_records"]:
                record.pop("logits_tuned", None)
        # Make prompt ids unique for matching.
        for idx, item in enumerate(payload["artifacts"]):
            item["prompt_id"] = f"p{idx}"

    with pytest.raises(ValueError, match="readout_mode='tuned'"):
        run_prompt_robustness(
            comparison,
            base,
            include_lens_families=["tuned"],
            num_permutations=4,
            num_bootstrap=4,
            seed=3,
        )


def test_run_prompt_robustness_requires_component_logits_for_prisms() -> None:
    comparison = {"artifacts": [_artifact(scale=1.10, prompt_id=f"p{idx}").to_dict() for idx in range(4)]}
    base = {"artifacts": [_artifact(scale=1.00, prompt_id=f"p{idx}").to_dict() for idx in range(4)]}
    for payload in (comparison, base):
        for item in payload["artifacts"]:
            for record in item["layer_records"]:
                record.pop("attention_logits_raw", None)
                record.pop("mlp_logits_raw", None)

    with pytest.raises(ValueError, match="missing attention component logits"):
        run_prompt_robustness(
            comparison,
            base,
            include_lens_families=["prisms"],
            prism_readout_mode="raw",
            num_permutations=4,
            num_bootstrap=4,
            seed=5,
        )


def test_prompt_robustness_cli_writes_saved_pair_output(tmp_path: Path) -> None:
    comparison = {"artifacts": [_artifact(scale=1.10, prompt_id=f"p{idx}").to_dict() for idx in range(4)]}
    base = {"artifacts": [_artifact(scale=1.00, prompt_id=f"p{idx}").to_dict() for idx in range(4)]}
    comparison_path = tmp_path / "comparison.pt"
    base_path = tmp_path / "base.pt"
    output_path = tmp_path / "robustness.pt"
    torch.save(comparison, comparison_path)
    torch.save(base, base_path)

    robustness_cli_main(
        [
            "--comparison-artifact",
            str(comparison_path),
            "--base-artifact",
            str(base_path),
            "--output-path",
            str(output_path),
            "--metric",
            "jsd_divergence",
            "--include-lens-families",
            "raw",
            "model_norm",
            "prisms",
            "--prism-readout-mode",
            "raw",
            "--num-permutations",
            "4",
            "--num-bootstrap",
            "4",
            "--seed",
            "11",
        ]
    )
    assert output_path.exists()

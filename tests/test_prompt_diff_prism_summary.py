from __future__ import annotations

import torch

from logit_diff_lens.plotting.prism_plots import plot_prompt_diff_prism_summary
from logit_diff_lens.prisms.io import load_prompt_diff_prism_summary_artifact, save_prompt_diff_prism_summary_artifact
from logit_diff_lens.prisms.summary import build_prompt_diff_prism_summary_artifact
from logit_diff_lens.schemas.prism_outputs import PromptDiffPrismHeatmapArtifact


def _heatmap(scale: float) -> PromptDiffPrismHeatmapArtifact:
    matrix = torch.tensor(
        [
            [1.0, -0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.2, -0.2, 0.1],
            [1.7, -0.2, 0.6],
            [1.7, -0.2, 0.6],
        ],
        dtype=torch.float32,
    ) * scale
    return PromptDiffPrismHeatmapArtifact(
        prompt_text="demo",
        prompt_formatted="demo",
        prompt_id="1",
        readout_mode="raw",
        position_index=1,
        predictor_token_id=2,
        predictor_token_text="b",
        target_token_id=3,
        target_token_text="c",
        side_ft_label="ft",
        side_base_label="base",
        delta_definition="ft - base",
        token_selection="largest_abs_delta",
        top_k=3,
        component_labels=["embedding", "attn_0", "mlp_0", "full_layer_0", "output_l+1"],
        selected_token_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        selected_token_text=["tok0", "tok1", "tok2"],
        contribution_matrix=matrix,
        total_delta_logits=matrix[-1].clone(),
        summed_component_logits=matrix[-2].clone(),
        reconstruction_gap=torch.zeros(3, dtype=torch.float32),
        backend_metadata_ft=None,
        backend_metadata_base=None,
        metadata={},
    )


def test_build_prompt_diff_prism_summary_artifact_without_nulls() -> None:
    artifact = build_prompt_diff_prism_summary_artifact(_heatmap(1.0), summary_kind="mean_abs")
    expected = torch.tensor(
        [
            (1.0 + 0.5 + 0.0) / 3.0,
            0.5,
            (0.2 + 0.2 + 0.1) / 3.0,
            (1.7 + 0.2 + 0.6) / 3.0,
            (1.7 + 0.2 + 0.6) / 3.0,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(artifact.raw_component_scores, expected, atol=1e-6)
    assert artifact.calibrated_component_scores is None


def test_build_prompt_diff_prism_summary_artifact_with_nulls() -> None:
    observed = _heatmap(2.0)
    nulls = [_heatmap(0.25), _heatmap(0.5), _heatmap(0.75)]
    artifact = build_prompt_diff_prism_summary_artifact(
        observed,
        summary_kind="mean_abs",
        null_sources=nulls,
        alpha=0.05,
    )
    assert artifact.calibrated_component_scores is not None
    assert artifact.p_value_vector is not None
    assert artifact.adjusted_p_value_vector is not None
    assert artifact.critical_value_vector is not None
    assert artifact.null_mean_vector is not None
    assert artifact.null_std_vector is not None
    assert torch.all(artifact.raw_component_scores >= artifact.critical_value_vector)
    assert torch.all(artifact.calibrated_component_scores >= 0.0)


def test_prompt_diff_prism_summary_round_trip_and_plot(tmp_path) -> None:
    artifact = build_prompt_diff_prism_summary_artifact(
        _heatmap(2.0),
        summary_kind="mean_abs",
        null_sources=[_heatmap(0.25), _heatmap(0.5), _heatmap(0.75)],
    )
    path = tmp_path / "prompt_diff_prism_summary.pt"
    save_prompt_diff_prism_summary_artifact(artifact, path)
    loaded = load_prompt_diff_prism_summary_artifact(path)
    assert loaded.summary_kind == "mean_abs"
    assert loaded.component_labels == artifact.component_labels
    assert loaded.calibrated_component_scores is not None

    fig = plot_prompt_diff_prism_summary(loaded)
    assert len(fig.data) == 2
    assert fig.data[0].name == "Non-calibrated"
    assert fig.data[1].name == "Calibrated"

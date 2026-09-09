from __future__ import annotations

import torch

from logit_diff_lens.plotting.correlation_heatmaps import plot_prompt_correlation_heatmap
from logit_diff_lens.schemas.correlation_outputs import CorrelationRunArtifact, LayerPairCorrelationArtifact


def _artifact() -> CorrelationRunArtifact:
    matrix = LayerPairCorrelationArtifact(
        metric_name="top1_confidence",
        readout_mode="model_norm",
        pearson_r_matrix=torch.tensor([[0.8, 0.1], [0.2, 0.7]], dtype=torch.float32),
        pearson_p_matrix=torch.tensor([[0.001, 0.7], [0.4, 0.01]], dtype=torch.float32),
        pearson_q_matrix=torch.tensor([[0.004, 0.7], [0.5, 0.02]], dtype=torch.float32),
        pearson_ci_low_matrix=torch.tensor([[0.6, -0.5], [-0.4, 0.3]], dtype=torch.float32),
        pearson_ci_high_matrix=torch.tensor([[0.9, 0.6], [0.7, 0.9]], dtype=torch.float32),
        spearman_r_matrix=torch.tensor([[0.75, 0.05], [0.1, 0.65]], dtype=torch.float32),
        spearman_p_matrix=torch.tensor([[0.002, 0.8], [0.6, 0.02]], dtype=torch.float32),
        spearman_q_matrix=torch.tensor([[0.005, 0.8], [0.7, 0.03]], dtype=torch.float32),
        spearman_ci_low_matrix=torch.tensor([[0.5, -0.6], [-0.5, 0.2]], dtype=torch.float32),
        spearman_ci_high_matrix=torch.tensor([[0.88, 0.5], [0.6, 0.87]], dtype=torch.float32),
        sample_count_matrix=torch.tensor([[12, 12], [12, 12]], dtype=torch.float32),
        layer_indices_a=[-1, 0],
        layer_indices_b=[-1, 0],
        metadata={},
    )
    return CorrelationRunArtifact(
        side_a_label="base",
        side_b_label="ft",
        artifact_family="prompt",
        readout_mode="model_norm",
        alignment_mode="same_token_ids",
        backend_metadata_a=None,
        backend_metadata_b=None,
        matrix_results=[matrix],
        metadata={},
    )


def test_plot_prompt_correlation_heatmap_builds_heatmap() -> None:
    fig = plot_prompt_correlation_heatmap(_artifact())
    assert fig.data[0].type == "heatmap"
    assert fig.layout.xaxis.title.text == "ft layers"
    assert fig.layout.yaxis.title.text == "base layers"

from __future__ import annotations

import torch

from logit_diff_lens.plotting.similarity_heatmaps import (
    plot_similarity_aggregate_summary,
    plot_similarity_matrix_heatmap,
)
from logit_diff_lens.schemas.similarity_outputs import (
    LayerPairSimilarityArtifact,
    SimilarityRunArtifact,
)


def _artifact() -> SimilarityRunArtifact:
    matrix = LayerPairSimilarityArtifact(
        metric_name="js_similarity",
        representation_kind="logits",
        raw_matrix=torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32),
        calibrated_matrix=torch.tensor([[0.6, 0.0], [0.1, 0.5]], dtype=torch.float32),
        p_value_matrix=torch.tensor([[0.01, 0.7], [0.4, 0.05]], dtype=torch.float32),
        adjusted_p_value_matrix=torch.tensor([[0.02, 0.7], [0.5333, 0.1]], dtype=torch.float32),
        null_percentile_matrix=torch.tensor([[1.0, 0.4], [0.7, 0.8]], dtype=torch.float32),
        null_mean_matrix=torch.tensor([[0.2, 0.2], [0.2, 0.2]], dtype=torch.float32),
        layer_indices_a=[-1, 0],
        layer_indices_b=[-1, 0],
        aggregate_statistic_name="max",
        aggregate_raw_value=0.8,
        aggregate_null_scores=[0.1, 0.2, 0.25, 0.3],
        aggregate_critical_value=0.3,
        aggregate_p_value=0.01,
        aggregate_adjusted_p_value=0.01,
        aggregate_null_percentile=1.0,
        aggregate_calibrated_value=0.7142857,
        metadata={},
    )
    return SimilarityRunArtifact(
        side_a_label="base",
        side_b_label="ft",
        artifact_family="prompt",
        prompt_id="1",
        prompt_text="demo",
        readout_mode="model_norm",
        alignment_mode="same_token_ids",
        backend_metadata_a=None,
        backend_metadata_b=None,
        scalar_results=[],
        matrix_results=[matrix],
        metadata={},
    )


def test_plot_similarity_matrix_heatmap_uses_requested_matrix_kind() -> None:
    fig = plot_similarity_matrix_heatmap(_artifact(), plot_kind="calibrated")
    assert fig.data[0].type == "heatmap"
    assert fig.layout.xaxis.title.text == "ft layers"
    assert fig.layout.yaxis.title.text == "base layers"


def test_plot_similarity_aggregate_summary_builds_histogram_and_reference_lines() -> None:
    fig = plot_similarity_aggregate_summary(_artifact())
    assert fig.data[0].type == "histogram"
    assert len(fig.layout.shapes) == 2


def test_plot_similarity_matrix_heatmap_supports_adjusted_p_values() -> None:
    fig = plot_similarity_matrix_heatmap(_artifact(), plot_kind="adjusted_p_value")
    assert fig.data[0].type == "heatmap"

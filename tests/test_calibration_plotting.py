from __future__ import annotations

import torch

from logit_diff_lens.plotting.calibration_heatmaps import (
    plot_calibration_matrix_heatmap,
    plot_calibration_summary,
)
from logit_diff_lens.schemas.calibration_outputs import (
    CalibrationMatrixResult,
    CalibrationRunArtifact,
    CalibrationSummaryResult,
)


def _artifact() -> CalibrationRunArtifact:
    return CalibrationRunArtifact(
        side_a_label="base",
        side_b_label="ft",
        artifact_family="prompt",
        reference_kind="reference_relative",
        reference_source="final_layer",
        readout_mode_eval="model_norm",
        readout_mode_reference="model_norm",
        alignment_mode="same_token_ids",
        layer_indices=[-1, 0],
        position_indices=[0, 1, 2],
        backend_metadata_a=None,
        backend_metadata_b=None,
        matrix_results=[
            CalibrationMatrixResult(
                metric_name="top1_ece",
                axis_kind="layer_position",
                value_matrix=torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]], dtype=torch.float32),
                sample_count_matrix=torch.tensor([[4, 4, 4], [4, 4, 4]], dtype=torch.float32),
                confidence_mean_matrix=torch.tensor([[0.6, 0.5, 0.4], [0.5, 0.6, 0.7]], dtype=torch.float32),
                accuracy_mean_matrix=torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.7, 0.8]], dtype=torch.float32),
                metadata={},
            )
        ],
        summary_results=[
            CalibrationSummaryResult(
                metric_name="top1_ece",
                axis_kind="layer",
                axis_indices=[-1, 0],
                values=torch.tensor([0.2, 0.1], dtype=torch.float32),
                ci_low=torch.tensor([0.1, 0.05], dtype=torch.float32),
                ci_high=torch.tensor([0.3, 0.2], dtype=torch.float32),
                sample_counts=torch.tensor([12, 12], dtype=torch.float32),
                confidence_means=torch.tensor([0.5, 0.6], dtype=torch.float32),
                accuracy_means=torch.tensor([0.5, 0.7], dtype=torch.float32),
                metadata={},
            )
        ],
        metadata={},
    )


def test_plot_calibration_matrix_heatmap_builds_layer_position_heatmap() -> None:
    fig = plot_calibration_matrix_heatmap(_artifact())
    assert fig.data[0].type == "heatmap"
    assert fig.layout.yaxis.title.text == "base layers"


def test_plot_calibration_summary_builds_line_plot() -> None:
    fig = plot_calibration_summary(_artifact(), metric_name="top1_ece", axis_kind="layer")
    assert fig.data[0].type == "scatter"

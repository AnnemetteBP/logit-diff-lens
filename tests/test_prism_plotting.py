from __future__ import annotations

import torch

from logit_diff_lens.plotting.prism_plots import (
    plot_prompt_diff_prism,
    plot_prompt_prism,
    plot_prompt_prism_comparison,
)
from logit_diff_lens.schemas.prism_outputs import (
    PromptDiffPrismHeatmapArtifact,
    PromptPrismArtifact,
)


def _artifact(scale: float) -> PromptPrismArtifact:
    return PromptPrismArtifact(
        prompt_text="demo",
        prompt_formatted="demo",
        prompt_id="1",
        readout_mode="model_norm",
        top_k=2,
        token_selection="largest_positive_logit",
        position_indices=[0, 1],
        token_ids=torch.tensor([[1, 2]], dtype=torch.long),
        token_text=["a", "b"],
        component_labels=["embedding", "attn_0", "mlp_0", "full_layer_0", "output_l+1"],
        selected_token_ids=torch.tensor([1, 2], dtype=torch.long),
        selected_token_text=["tok1", "tok2"],
        contribution_tensor=torch.tensor(
            [
                [[1.0, 0.5], [0.3, 0.2], [0.2, 0.1], [1.5, 0.8], [1.5, 0.8]],
                [[1.2, 0.4], [0.4, 0.1], [0.3, 0.1], [1.9, 0.6], [1.9, 0.6]],
            ],
            dtype=torch.float32,
        )
        * scale,
        output_token_logits=torch.tensor([[1.5, 0.8], [1.9, 0.6]], dtype=torch.float32) * scale,
        additive_component_logits=torch.tensor([[1.5, 0.8], [1.9, 0.6]], dtype=torch.float32) * scale,
        cumulative_full_layer_logits=torch.tensor([[1.5, 0.8], [1.9, 0.6]], dtype=torch.float32) * scale,
        output_residual_gap=torch.zeros((2, 2), dtype=torch.float32),
        backend_metadata=None,
        metadata={},
    )


def test_plot_prompt_prism_builds_per_token_figure() -> None:
    fig = plot_prompt_prism(_artifact(1.0), plot_mode="per_token", position_index=0)
    assert len(fig.data) == 2
    assert "Prompt Prism | demo" in fig.layout.title.text


def test_plot_prompt_prism_builds_mean_over_positions_figure() -> None:
    fig = plot_prompt_prism(_artifact(1.0), plot_mode="mean_over_positions", uncertainty="stderr")
    assert len(fig.data) == 4
    assert "Mean over 2 positions" in fig.layout.title.text


def test_plot_prompt_prism_comparison_builds_overlay_figure() -> None:
    fig = plot_prompt_prism_comparison(_artifact(1.0), _artifact(1.5), side_a_label="a", side_b_label="b")
    assert len(fig.data) == 4
    assert fig.layout.title.text == "Prompt Prism Comparison | a vs b"


def test_plot_prompt_diff_prism_builds_delta_figure() -> None:
    artifact = PromptDiffPrismHeatmapArtifact(
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
        selected_token_text=["0", "1", "2"],
        contribution_matrix=torch.tensor(
            [
                [1.7, 0.2, 0.2],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.25],
                [1.7, 0.7, 0.45],
                [1.7, 0.7, 0.45],
            ],
            dtype=torch.float32,
        ),
        total_delta_logits=torch.tensor([1.7, 0.7, 0.45], dtype=torch.float32),
        summed_component_logits=torch.tensor([1.7, 0.7, 0.45], dtype=torch.float32),
        reconstruction_gap=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        backend_metadata_ft=None,
        backend_metadata_base=None,
        metadata={"max_abs_gap": 0.0},
    )
    fig = plot_prompt_diff_prism(artifact)
    assert len(fig.data) == 3
    assert "Prompt Diff Prism | Delta = ft - base" in fig.layout.title.text

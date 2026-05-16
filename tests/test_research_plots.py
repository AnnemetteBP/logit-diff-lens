from __future__ import annotations

from pathlib import Path

import numpy as np

from diffing.logit_lens_methods.logitdiff_ldl.research_plots import (
    compute_auroc_per_layer,
    save_latent_to_output_figure,
    save_latent_to_output_figure_from_results,
)


def test_compute_auroc_per_layer_handles_constant_class_labels() -> None:
    aucs = compute_auroc_per_layer(
        projection_scores_per_layer=[[0.1, 0.2], [0.3, 0.4]],
        labels=[1, 1],
    )
    assert np.isnan(aucs).all()


def test_save_latent_to_output_figure_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "latent_to_output_figure.png"
    save_latent_to_output_figure(
        kl_per_layer=[0.2, 0.4, 0.1],
        projection_scores_per_layer=[
            [0.1, 0.9, 0.2, 0.8],
            [0.2, 0.8, 0.3, 0.7],
            [0.4, 0.6, 0.45, 0.55],
        ],
        labels=[0, 1, 0, 1],
        pca_variance_per_layer=[
            [0.7, 0.2, 0.1],
            [0.6, 0.25, 0.15],
            [0.5, 0.3, 0.2],
        ],
        attn_contrib_norm=[1.0, 2.0, 1.5],
        mlp_contrib_norm=[0.8, 1.8, 1.2],
        output_path=output_path,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_latent_to_output_figure_from_results_aggregates_prism_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "latent_to_output_figure_from_results.png"
    latent_shift_result = {
        "kl_divergence": [
            {"layer": 0, "values": [0.1, 0.2, 0.3, 0.4]},
            {"layer": 1, "values": [0.2, 0.3, 0.4, 0.5]},
        ],
        "projection_scores": [
            {"layer": 0, "scores": [0.1, 0.9, 0.2, 0.8]},
            {"layer": 1, "scores": [0.2, 0.8, 0.3, 0.7]},
        ],
        "pca_results": [
            {"layer": 0, "explained_variance_ratio": [0.6, 0.3, 0.1]},
            {"layer": 1, "explained_variance_ratio": [0.55, 0.3, 0.15]},
        ],
    }
    prism_result = {
        "rows": [
            {
                "base_prism": {
                    "summary": {
                        "0": {"attn_norm": 1.0, "mlp_norm": 2.0},
                        "1": {"attn_norm": 2.0, "mlp_norm": 3.0},
                    }
                }
            },
            {
                "base_prism": {
                    "summary": {
                        "0": {"attn_norm": 3.0, "mlp_norm": 4.0},
                        "1": {"attn_norm": 4.0, "mlp_norm": 5.0},
                    }
                }
            },
        ]
    }

    save_latent_to_output_figure_from_results(
        latent_shift_result,
        labels=[0, 1, 0, 1],
        prism_result=prism_result,
        prism_source="base_prism",
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

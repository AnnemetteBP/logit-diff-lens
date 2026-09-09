from __future__ import annotations

import importlib


def test_plotting_heatmaps_exports_are_lazy_and_resolve_expected_symbols() -> None:
    module = importlib.import_module("logit_diff_lens.plotting.heatmaps")

    assert "plot_comparison_metric_heatmap" in module.__all__
    assert "plot_calibration_matrix_heatmap" in module.__all__
    assert "plot_prompt_correlation_heatmap" in module.__all__
    assert "plot_prompt_prism" in module.__all__
    assert "plot_prompt_prism_comparison" in module.__all__
    assert "plot_jaccard_heatmap" in module.__all__
    assert "plot_logitdiff_jaccard_heatmap" in module.__all__

    comparison_plotter = getattr(module, "plot_comparison_metric_heatmap")

    assert callable(comparison_plotter)


def test_plotting_generation_entrypoint_uses_package_surface() -> None:
    module = importlib.import_module("logit_diff_lens.plotting.logitdiff_generation_plotter")

    assert callable(module.plot_generation_logitdiff_heatmap)
    assert callable(module.save_generation_logitdiff_heatmap)


def test_prompt_and_generation_heatmap_modules_import_without_eager_legacy_loading() -> None:
    prompt_module = importlib.import_module("logit_diff_lens.plotting.prompt_heatmaps")
    generation_module = importlib.import_module("logit_diff_lens.plotting.generation_heatmaps")

    assert "plot_jaccard_heatmap" in prompt_module.__all__
    assert "plot_logitdiff_jaccard_heatmap" in generation_module.__all__

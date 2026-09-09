"""Reusable plotting entry points for package outputs."""

from __future__ import annotations

from importlib import import_module


_ATTR_IMPORTS = {
    "plot_prompt_diff_prism": (".prism_plots", "plot_prompt_diff_prism"),
    "plot_prompt_prism": (".prism_plots", "plot_prompt_prism"),
    "plot_prompt_prism_comparison": (".prism_plots", "plot_prompt_prism_comparison"),
    "plot_prompt_correlation_heatmap": (".correlation_heatmaps", "plot_prompt_correlation_heatmap"),
    "plot_calibration_matrix_heatmap": (".calibration_heatmaps", "plot_calibration_matrix_heatmap"),
    "plot_calibration_summary": (".calibration_heatmaps", "plot_calibration_summary"),
    "plot_comparison_metric_heatmap": (".comparison_heatmaps", "plot_comparison_metric_heatmap"),
    "plot_similarity_aggregate_summary": (".similarity_heatmaps", "plot_similarity_aggregate_summary"),
    "plot_similarity_matrix_heatmap": (".similarity_heatmaps", "plot_similarity_matrix_heatmap"),
    "save_similarity_figure": (".similarity_heatmaps", "save_similarity_figure"),
    "plot_generation_logitdiff_heatmap": (".logitdiff_generation_plotter", "plot_generation_logitdiff_heatmap"),
    "plot_generation_logitdiff_jaccard_heatmap": (
        ".logitdiff_generation_plotter",
        "plot_generation_logitdiff_jaccard_heatmap",
    ),
    "plot_generation_logitdiff_next_token_heatmap": (
        ".logitdiff_generation_plotter",
        "plot_generation_logitdiff_next_token_heatmap",
    ),
    "save_generation_logitdiff_heatmap": (".logitdiff_generation_plotter", "save_generation_logitdiff_heatmap"),
    "save_generation_logitdiff_jaccard_heatmap": (
        ".logitdiff_generation_plotter",
        "save_generation_logitdiff_jaccard_heatmap",
    ),
    "save_generation_logitdiff_next_token_heatmap": (
        ".logitdiff_generation_plotter",
        "save_generation_logitdiff_next_token_heatmap",
    ),
    "plot_prompt_logitdiff_heatmap": (".logitdiff_prompt_plotter", "plot_prompt_logitdiff_heatmap"),
    "plot_prompt_logitdiff_jaccard_heatmap": (
        ".logitdiff_prompt_plotter",
        "plot_prompt_logitdiff_jaccard_heatmap",
    ),
    "plot_prompt_logitdiff_next_token_heatmap": (
        ".logitdiff_prompt_plotter",
        "plot_prompt_logitdiff_next_token_heatmap",
    ),
    "save_prompt_logitdiff_heatmap": (".logitdiff_prompt_plotter", "save_prompt_logitdiff_heatmap"),
    "save_prompt_logitdiff_jaccard_heatmap": (
        ".logitdiff_prompt_plotter",
        "save_prompt_logitdiff_jaccard_heatmap",
    ),
    "save_prompt_logitdiff_next_token_heatmap": (
        ".logitdiff_prompt_plotter",
        "save_prompt_logitdiff_next_token_heatmap",
    ),
    "list_available_prompts": (".generation_heatmaps", "list_available_prompts"),
    "plot_logitdiff_jaccard_heatmap": (".generation_heatmaps", "plot_logitdiff_jaccard_heatmap"),
    "plot_logitdiff_jaccard_heatmap_interactive": (
        ".generation_heatmaps",
        "plot_logitdiff_jaccard_heatmap_interactive",
    ),
    "save_logitdiff_heatmap": (".generation_heatmaps", "save_logitdiff_heatmap"),
    "save_logitdiff_heatmap_html": (".generation_heatmaps", "save_logitdiff_heatmap_html"),
    "save_logitdiff_heatmap_pdf": (".generation_heatmaps", "save_logitdiff_heatmap_pdf"),
    "plot_jaccard_heatmap": (".prompt_heatmaps", "plot_jaccard_heatmap"),
    "plot_logitdiff_next_token_verification_heatmap": (
        ".prompt_heatmaps",
        "plot_logitdiff_next_token_verification_heatmap",
    ),
    "save_jaccard_heatmap": (".prompt_heatmaps", "save_jaccard_heatmap"),
    "save_jaccard_heatmap_html": (".prompt_heatmaps", "save_jaccard_heatmap_html"),
    "save_jaccard_heatmap_pdf": (".prompt_heatmaps", "save_jaccard_heatmap_pdf"),
    "save_logitdiff_next_token_verification_html": (
        ".prompt_heatmaps",
        "save_logitdiff_next_token_verification_html",
    ),
    "save_logitdiff_next_token_verification_pdf": (
        ".prompt_heatmaps",
        "save_logitdiff_next_token_verification_pdf",
    ),
    "plot_single_model_logit_lens_heatmap": (
        ".single_model_logit_lens_plotter",
        "plot_single_model_logit_lens_heatmap",
    ),
    "save_single_model_logit_lens_heatmap": (
        ".single_model_logit_lens_plotter",
        "save_single_model_logit_lens_heatmap",
    ),
    "plot_adl_case_heatmap": (".adl_heatmap_plotter", "plot_adl_case_heatmap"),
    "save_adl_case_heatmap": (".adl_heatmap_plotter", "save_adl_case_heatmap"),
    "save_calibration_figure": (".calibration_heatmaps", "save_calibration_figure"),
    "save_correlation_figure": (".correlation_heatmaps", "save_correlation_figure"),
    "save_prompt_prism_comparison_figure": (".prism_plots", "save_prompt_prism_comparison_figure"),
    "plot_logitdiff_top_layer_chunked_heatmap": (
        ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_paper_plotter",
        "plot_logitdiff_top_layer_chunked_heatmap",
    ),
    "save_logitdiff_top_layer_chunked_heatmap_pdf": (
        ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_paper_plotter",
        "save_logitdiff_top_layer_chunked_heatmap_pdf",
    ),
    "save_logitdiff_top_layer_chunked_heatmap_png": (
        ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_paper_plotter",
        "save_logitdiff_top_layer_chunked_heatmap_png",
    ),
    "plot_logitdiff_top_layer_selected_rows_heatmap": (
        ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_paper_plotter_selected_rows",
        "plot_logitdiff_top_layer_selected_rows_heatmap",
    ),
    "save_logitdiff_top_layer_selected_rows_heatmap_pdf": (
        ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_paper_plotter_selected_rows",
        "save_logitdiff_top_layer_selected_rows_heatmap_pdf",
    ),
    "save_logitdiff_top_layer_selected_rows_heatmap_png": (
        ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_gen_paper_plotter_selected_rows",
        "save_logitdiff_top_layer_selected_rows_heatmap_png",
    ),
}

_MODULE_IMPORTS = {
    "adl_plotter": ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.adl_plotter",
    "ldl_plotter": ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.ldl_plotter",
    "logit_lens_plotter": ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logit_lens_plotter",
    "logitdiff_gen_plotter": ".logitdiff_gen_plotter",
    "logitdiff_pair_heatmap_plotter": (
        ".._legacy.logitdiff_toolkit.logit_lens_methods.plotting.heatmaps.logitdiff_pair_heatmap_plotter"
    ),
}

__all__ = [
    "adl_plotter",
    "build_single_prompt_payload",
    "ldl_plotter",
    "list_available_prompts",
    "logit_lens_plotter",
    "logitdiff_gen_plotter",
    "logitdiff_pair_heatmap_plotter",
    "plot_adl_case_heatmap",
    "plot_calibration_matrix_heatmap",
    "plot_calibration_summary",
    "plot_prompt_correlation_heatmap",
    "plot_prompt_diff_prism",
    "plot_prompt_prism",
    "plot_prompt_prism_comparison",
    "plot_comparison_metric_heatmap",
    "plot_generation_logitdiff_heatmap",
    "plot_generation_logitdiff_jaccard_heatmap",
    "plot_generation_logitdiff_next_token_heatmap",
    "plot_jaccard_heatmap",
    "plot_logitdiff_jaccard_heatmap",
    "plot_logitdiff_jaccard_heatmap_interactive",
    "plot_logitdiff_next_token_verification_heatmap",
    "plot_logitdiff_top_layer_chunked_heatmap",
    "plot_logitdiff_top_layer_selected_rows_heatmap",
    "plot_prompt_logitdiff_heatmap",
    "plot_prompt_logitdiff_jaccard_heatmap",
    "plot_prompt_logitdiff_next_token_heatmap",
    "plot_prompt_style_verification_heatmap",
    "plot_similarity_aggregate_summary",
    "plot_similarity_matrix_heatmap",
    "plot_single_model_logit_lens_heatmap",
    "plot_tuned_lens_trajectory_figure",
    "save_adl_case_heatmap",
    "save_calibration_figure",
    "save_correlation_figure",
    "save_prompt_prism_comparison_figure",
    "save_generation_logitdiff_heatmap",
    "save_generation_logitdiff_jaccard_heatmap",
    "save_generation_logitdiff_next_token_heatmap",
    "save_jaccard_heatmap",
    "save_jaccard_heatmap_html",
    "save_jaccard_heatmap_pdf",
    "save_logitdiff_heatmap",
    "save_logitdiff_heatmap_html",
    "save_logitdiff_heatmap_pdf",
    "save_logitdiff_next_token_verification_html",
    "save_logitdiff_next_token_verification_pdf",
    "save_prompt_logitdiff_heatmap",
    "save_prompt_logitdiff_jaccard_heatmap",
    "save_prompt_logitdiff_next_token_heatmap",
    "save_similarity_figure",
    "save_single_model_logit_lens_heatmap",
    "save_tuned_lens_trajectory_figure",
    "save_logitdiff_top_layer_chunked_heatmap_pdf",
    "save_logitdiff_top_layer_chunked_heatmap_png",
    "save_logitdiff_top_layer_selected_rows_heatmap_pdf",
    "save_logitdiff_top_layer_selected_rows_heatmap_png",
]


def __getattr__(name: str):
    if name == "build_single_prompt_payload":
        from .tuned_vs_modelnorm import build_single_prompt_payload as _impl

        return _impl
    if name == "plot_prompt_style_verification_heatmap":
        from .tuned_vs_modelnorm import plot_prompt_style_verification_heatmap as _impl

        return _impl
    if name == "plot_tuned_lens_trajectory_figure":
        from .tuned_lens_trajectory import build_trajectory_figure as _impl

        return _impl
    if name == "save_tuned_lens_trajectory_figure":
        from .tuned_lens_trajectory import save_trajectory_figure as _impl

        return _impl
    if name in _ATTR_IMPORTS:
        module_name, attr_name = _ATTR_IMPORTS[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    if name in _MODULE_IMPORTS:
        return import_module(_MODULE_IMPORTS[name], __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

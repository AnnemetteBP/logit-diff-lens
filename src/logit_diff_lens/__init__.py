"""
Canonical package surface for the project.

The implementation currently lives under ``diffing.logit_lens_methods`` while
the donor ``diffing.methods`` tree is gradually being pruned. Re-export the
owned method family here so the public package name matches the project name.
"""

from diffing.logit_lens_methods import (
    AgentBudgetConfig,
    AgentLLMConfig,
    AskModelConfig,
    LogitDiffAgent,
    LogitDiffAgentConfig,
    LogitDiffRunConfig,
    build_logitdiff_overview,
    build_model_organisms_logitdiff_path,
    gen,
    logitdiff,
    logitdiff_adl,
    logitdiff_ldl,
    make_broad_misalignment_config,
    make_misalignment_config,
    make_quantization_config,
    run_logitdiff,
    run_logitdiff_agent,
    save_agent_messages,
    save_agent_stats,
    save_logitdiff_heatmap_html,
    save_logitdiff_heatmap_pdf,
    save_logitdiff_overview,
    save_model_organisms_logitdiff,
    plot_logitdiff_jaccard_heatmap,
    list_available_prompts,
)

__all__ = [
    "logitdiff",
    "logitdiff_adl",
    "logitdiff_ldl",
    "gen",
    "AgentLLMConfig",
    "AgentBudgetConfig",
    "AskModelConfig",
    "LogitDiffRunConfig",
    "LogitDiffAgentConfig",
    "LogitDiffAgent",
    "build_model_organisms_logitdiff_path",
    "build_logitdiff_overview",
    "run_logitdiff",
    "run_logitdiff_agent",
    "save_agent_messages",
    "save_agent_stats",
    "save_logitdiff_overview",
    "save_model_organisms_logitdiff",
    "list_available_prompts",
    "plot_logitdiff_jaccard_heatmap",
    "save_logitdiff_heatmap_html",
    "save_logitdiff_heatmap_pdf",
    "make_quantization_config",
    "make_misalignment_config",
    "make_broad_misalignment_config",
]

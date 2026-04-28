from . import logitdiff_adl
from . import gen
from . import logitdiff_ldl
from . import logitdiff

from .logitdiff import (
    AgentBudgetConfig,
    AgentLLMConfig,
    AskModelConfig,
    LogitDiffRunConfig,
    LogitDiffAgent,
    LogitDiffAgentConfig,
    build_model_organisms_logitdiff_path,
    build_logitdiff_overview,
    make_broad_misalignment_config,
    run_logitdiff,
    run_logitdiff_agent,
    save_agent_messages,
    save_agent_stats,
    save_logitdiff_overview,
    save_model_organisms_logitdiff,
    make_quantization_config,
    make_misalignment_config,
)

try:
    from .logitdiff import (
        list_available_prompts,
        plot_logitdiff_jaccard_heatmap,
        save_logitdiff_heatmap_html,
        save_logitdiff_heatmap_pdf,
    )
except (ModuleNotFoundError, ImportError):
    plotting = None
    list_available_prompts = None
    plot_logitdiff_jaccard_heatmap = None
    save_logitdiff_heatmap_html = None
    save_logitdiff_heatmap_pdf = None


__all__ = [
    "logitdiff_adl",
    "gen",
    "logitdiff_ldl",
    "logitdiff",
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
    "make_quantization_config",
    "make_misalignment_config",
    "make_broad_misalignment_config",
    "list_available_prompts",
    "plot_logitdiff_jaccard_heatmap",
    "save_logitdiff_heatmap_html",
    "save_logitdiff_heatmap_pdf",
]

from .core import (
    LogitDiffRunConfig,
    build_model_organisms_logitdiff_path,
    run_logitdiff,
    run_logitdiff_repeated,
    save_model_organisms_logitdiff,
)
from .cross_model_eval import (
    CrossModelEvalConfig,
    run_cross_model_eval,
)
from .logit_prism import (
    collect_teacher_forced_activations,
    collect_teacher_forced_activations_dataset_pair_incremental,
    LogitPrismConfig,
    run_logit_prism,
    run_logit_prism_batch,
)
from .agent_tools import (
    build_logitdiff_overview,
    flatten_logitdiff_records,
    save_flattened_logitdiff_records,
    save_logitdiff_overview,
)
from .agent import (
    AgentBudgetConfig,
    AgentLLMConfig,
    AskModelConfig,
    LogitDiffAgent,
    LogitDiffAgentConfig,
    run_logitdiff_agent,
)
from .agent_io import save_agent_messages, save_agent_stats
from .presets import (
    EM_PROMPTS,
    BROAD_MISALIGNMENT_PROMPTS,
    FT_MISALIGNMENT_PROMPTS,
    QUANTIZATION_PROMPTS,
    make_em_config,
    make_em_sampling_config,
    make_broad_misalignment_config,
    make_misalignment_config,
    make_quantization_config,
)

try:
    from .plotting import (
        list_available_prompts,
        plot_logitdiff_jaccard_heatmap,
        save_logitdiff_heatmap_html,
        save_logitdiff_heatmap_pdf,
    )
except ModuleNotFoundError:
    pass


__all__ = [
    "LogitDiffRunConfig",
    "build_model_organisms_logitdiff_path",
    "build_logitdiff_overview",
    "flatten_logitdiff_records",
    "AgentLLMConfig",
    "AgentBudgetConfig",
    "AskModelConfig",
    "LogitDiffAgentConfig",
    "LogitDiffAgent",
    "run_logitdiff",
    "run_logitdiff_repeated",
    "run_logitdiff_agent",
    "save_agent_messages",
    "save_agent_stats",
    "save_flattened_logitdiff_records",
    "save_logitdiff_overview",
    "save_model_organisms_logitdiff",
    "list_available_prompts",
    "plot_logitdiff_jaccard_heatmap",
    "save_logitdiff_heatmap_html",
    "save_logitdiff_heatmap_pdf",
    "EM_PROMPTS",
    "QUANTIZATION_PROMPTS",
    "FT_MISALIGNMENT_PROMPTS",
    "BROAD_MISALIGNMENT_PROMPTS",
    "make_em_config",
    "make_em_sampling_config",
    "make_quantization_config",
    "make_misalignment_config",
    "make_broad_misalignment_config",
    "collect_teacher_forced_activations",
    "collect_teacher_forced_activations_dataset_pair_incremental",
]

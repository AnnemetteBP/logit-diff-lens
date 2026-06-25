"""Prompt, generation, and backward collection APIs."""

from __future__ import annotations

__all__ = [
    "BackwardLensCollectorConfig",
    "GenerationActivationCollectorConfig",
    "PatchscopePromptConfig",
    "PromptLensActivationCollectorConfig",
    "collect_backward_prompt_artifact",
    "collect_generation_activation_dataset_incremental",
    "collect_generation_activations",
    "collect_patchscope_prompt_artifact",
    "collect_prompt_activation_dataset_incremental",
    "collect_prompt_lens_activations",
    "collect_prompt_logits_for_plotter",
]


def __getattr__(name: str):
    if name in {"BackwardLensCollectorConfig", "collect_backward_prompt_artifact"}:
        from .backward import BackwardLensCollectorConfig, collect_backward_prompt_artifact

        mapping = {
            "BackwardLensCollectorConfig": BackwardLensCollectorConfig,
            "collect_backward_prompt_artifact": collect_backward_prompt_artifact,
        }
        return mapping[name]

    if name in {
        "PromptLensActivationCollectorConfig",
        "collect_prompt_activation_dataset_incremental",
        "collect_prompt_lens_activations",
        "collect_prompt_logits_for_plotter",
    }:
        from .prompt import (
            PromptLensActivationCollectorConfig,
            collect_prompt_activation_dataset_incremental,
            collect_prompt_lens_activations,
            collect_prompt_logits_for_plotter,
        )

        mapping = {
            "PromptLensActivationCollectorConfig": PromptLensActivationCollectorConfig,
            "collect_prompt_activation_dataset_incremental": collect_prompt_activation_dataset_incremental,
            "collect_prompt_lens_activations": collect_prompt_lens_activations,
            "collect_prompt_logits_for_plotter": collect_prompt_logits_for_plotter,
        }
        return mapping[name]

    if name in {"PatchscopePromptConfig", "collect_patchscope_prompt_artifact"}:
        from .patchscope import PatchscopePromptConfig, collect_patchscope_prompt_artifact

        mapping = {
            "PatchscopePromptConfig": PatchscopePromptConfig,
            "collect_patchscope_prompt_artifact": collect_patchscope_prompt_artifact,
        }
        return mapping[name]

    if name in {
        "GenerationActivationCollectorConfig",
        "collect_generation_activation_dataset_incremental",
        "collect_generation_activations",
    }:
        from .generation import (
            GenerationActivationCollectorConfig,
            collect_generation_activation_dataset_incremental,
            collect_generation_activations,
        )

        mapping = {
            "GenerationActivationCollectorConfig": GenerationActivationCollectorConfig,
            "collect_generation_activation_dataset_incremental": collect_generation_activation_dataset_incremental,
            "collect_generation_activations": collect_generation_activations,
        }
        return mapping[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Main-project entry point for single-prompt patch sweeps."""

from __future__ import annotations

from .._legacy.logitdiff_toolkit.logit_lens_methods.pipelines.run_single_prompt_patch_sweep import (
    main,
    run_patch_sweep,
)

__all__ = ["main", "run_patch_sweep"]

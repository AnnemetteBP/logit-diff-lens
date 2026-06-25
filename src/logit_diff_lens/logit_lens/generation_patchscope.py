"""Main-project entry point for generation-focused patchscope runs."""

from __future__ import annotations

from .._legacy.logitdiff_toolkit.logit_lens_methods.pipelines.run_autoregressive_patch_scope import (
    main,
    run_autoregressive_patch_scope,
)

__all__ = ["main", "run_autoregressive_patch_scope"]

"""Compatibility package for the current mixed repo layout.

This package makes the existing `diffing.*` imports resolvable without moving
or deleting the older toolkit code yet. It extends the package search path to
the two current donor-code locations:

- `diffing-toolkit/` for methods, utils, evaluators, cli
- `src/logitdiff-toolkit/` for logit-lens-specific modules
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LEGACY_DIFFING = _ROOT / "diffing-toolkit"
_LEGACY_LOGITDIFF = _ROOT / "src" / "logitdiff-toolkit"

if _LEGACY_DIFFING.is_dir():
    __path__.append(str(_LEGACY_DIFFING))
if _LEGACY_LOGITDIFF.is_dir():
    __path__.append(str(_LEGACY_LOGITDIFF))

__all__ = ["methods", "evaluators", "logit_lens_methods", "logit_lens_pipelines", "utils", "cli"]

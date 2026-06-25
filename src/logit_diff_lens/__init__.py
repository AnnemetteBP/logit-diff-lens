"""Core package for the conference-oriented Logit Diff Lens refactor.

This package is being introduced conservatively. Existing research code remains
in place while stable functionality is copied over into clearer package
boundaries.
"""

from . import wrappers

__all__ = ["wrappers"]

from __future__ import annotations

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


_PLOTTING_ROOT = Path(__file__).resolve().parents[1] / "_legacy" / "logitdiff_toolkit" / "logit_lens_methods" / "plotting" / "heatmaps"


@lru_cache(maxsize=None)
def load_canonical_plotter_module(module_filename: str) -> ModuleType:
    module_path = _PLOTTING_ROOT / module_filename
    module_name = f"logit_diff_lens.plotting._canonical_{module_path.stem}"
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load canonical plotter module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


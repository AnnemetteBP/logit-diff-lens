from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def main() -> None:
    _bootstrap_src()
    from logit_diff_lens.plotting.tuned_vs_modelnorm import main as package_main

    package_main()


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_root() -> None:
    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> None:
    _bootstrap_root()
    from pipelines.train_tuned_lens import main as root_main

    root_main()


if __name__ == "__main__":
    main()

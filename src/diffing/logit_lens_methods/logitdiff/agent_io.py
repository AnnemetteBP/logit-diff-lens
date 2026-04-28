from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_agent_messages(stats: Dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "messages.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats.get("messages", []), f, indent=2, ensure_ascii=False)
    return path


def save_agent_stats(stats: Dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stats.json"
    payload = {k: v for k, v in stats.items() if k != "messages"}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path

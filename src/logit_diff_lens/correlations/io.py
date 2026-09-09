from __future__ import annotations

from pathlib import Path

import torch

from ..schemas.correlation_outputs import CorrelationRunArtifact


def save_correlation_run_artifact(artifact: CorrelationRunArtifact, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_correlation_run_artifact(path: str | Path) -> CorrelationRunArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    return CorrelationRunArtifact.from_dict(payload)


__all__ = [
    "load_correlation_run_artifact",
    "save_correlation_run_artifact",
]

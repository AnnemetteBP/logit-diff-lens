from __future__ import annotations

from pathlib import Path

import torch

from ..schemas.robustness_outputs import RobustnessRunArtifact


def save_robustness_run_artifact(artifact: RobustnessRunArtifact, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_robustness_run_artifact(path: str | Path) -> RobustnessRunArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    return RobustnessRunArtifact.from_dict(payload)


__all__ = ["load_robustness_run_artifact", "save_robustness_run_artifact"]

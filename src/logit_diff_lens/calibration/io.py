from __future__ import annotations

from pathlib import Path

import torch

from ..schemas.calibration_outputs import CalibrationRunArtifact


def save_calibration_run_artifact(artifact: CalibrationRunArtifact, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_calibration_run_artifact(path: str | Path) -> CalibrationRunArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    return CalibrationRunArtifact.from_dict(payload)


__all__ = [
    "load_calibration_run_artifact",
    "save_calibration_run_artifact",
]

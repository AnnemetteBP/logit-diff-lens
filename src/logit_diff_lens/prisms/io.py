from __future__ import annotations

from pathlib import Path

import torch

from ..schemas.prism_outputs import (
    PromptDiffPrismArtifact,
    PromptDiffPrismHeatmapArtifact,
    PromptDiffPrismSummaryArtifact,
    PromptPrismArtifact,
)


def save_prompt_prism_artifact(artifact: PromptPrismArtifact, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_prompt_prism_artifact(path: str | Path) -> PromptPrismArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    return PromptPrismArtifact.from_dict(payload)


def save_prompt_diff_prism_artifact(artifact: PromptDiffPrismArtifact, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_prompt_diff_prism_artifact(path: str | Path) -> PromptDiffPrismArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    return PromptDiffPrismArtifact.from_dict(payload)


def save_prompt_diff_prism_heatmap_artifact(artifact: PromptDiffPrismHeatmapArtifact, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_prompt_diff_prism_heatmap_artifact(path: str | Path) -> PromptDiffPrismHeatmapArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    return PromptDiffPrismHeatmapArtifact.from_dict(payload)


def save_prompt_diff_prism_summary_artifact(artifact: PromptDiffPrismSummaryArtifact, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_prompt_diff_prism_summary_artifact(path: str | Path) -> PromptDiffPrismSummaryArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    return PromptDiffPrismSummaryArtifact.from_dict(payload)


__all__ = [
    "load_prompt_diff_prism_heatmap_artifact",
    "load_prompt_diff_prism_artifact",
    "load_prompt_diff_prism_summary_artifact",
    "load_prompt_prism_artifact",
    "save_prompt_diff_prism_heatmap_artifact",
    "save_prompt_diff_prism_artifact",
    "save_prompt_diff_prism_summary_artifact",
    "save_prompt_prism_artifact",
]

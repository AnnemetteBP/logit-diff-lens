from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..schemas import BackwardPromptArtifact, PatchscopePromptArtifact, PromptDecodeArtifact
from ..validation import (
    validate_backward_prompt_artifact,
    validate_patchscope_prompt_artifact,
    validate_prompt_decode_artifact,
)


def save_prompt_decode_artifact(
    artifact: PromptDecodeArtifact,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_prompt_decode_artifact(artifact)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_prompt_decode_artifact(path: str | Path) -> PromptDecodeArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    artifact = PromptDecodeArtifact.from_dict(payload)
    validate_prompt_decode_artifact(artifact)
    return artifact


def save_prompt_decode_artifact_bundle(
    artifacts: list[PromptDecodeArtifact],
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        validate_prompt_decode_artifact(artifact)
    payload = {
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "metadata": dict(metadata or {}),
        "num_artifacts": len(artifacts),
    }
    torch.save(payload, output_path)
    return output_path


def load_prompt_decode_artifact_bundle(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    artifacts = [PromptDecodeArtifact.from_dict(item) for item in payload.get("artifacts", [])]
    for artifact in artifacts:
        validate_prompt_decode_artifact(artifact)
    return {
        "artifacts": artifacts,
        "metadata": dict(payload.get("metadata", {})),
        "num_artifacts": int(payload.get("num_artifacts", len(artifacts))),
    }


def save_comparison_artifact(
    comparison: dict[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(comparison, output_path)
    return output_path


def load_comparison_artifact(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def save_backward_prompt_artifact(
    artifact: BackwardPromptArtifact,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_backward_prompt_artifact(artifact)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_backward_prompt_artifact(path: str | Path) -> BackwardPromptArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    artifact = BackwardPromptArtifact.from_dict(payload)
    validate_backward_prompt_artifact(artifact)
    return artifact


def save_patchscope_prompt_artifact(
    artifact: PatchscopePromptArtifact,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_patchscope_prompt_artifact(artifact)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_patchscope_prompt_artifact(path: str | Path) -> PatchscopePromptArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    artifact = PatchscopePromptArtifact.from_dict(payload)
    validate_patchscope_prompt_artifact(artifact)
    return artifact


__all__ = [
    "load_backward_prompt_artifact",
    "load_comparison_artifact",
    "load_patchscope_prompt_artifact",
    "load_prompt_decode_artifact",
    "load_prompt_decode_artifact_bundle",
    "save_backward_prompt_artifact",
    "save_comparison_artifact",
    "save_patchscope_prompt_artifact",
    "save_prompt_decode_artifact",
    "save_prompt_decode_artifact_bundle",
]

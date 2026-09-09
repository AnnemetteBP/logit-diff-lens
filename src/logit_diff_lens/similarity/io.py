from __future__ import annotations

from pathlib import Path

import torch

from ..schemas.similarity_outputs import SimilarityRunArtifact


def save_similarity_run_artifact(artifact: SimilarityRunArtifact, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact.to_dict(), output_path)
    return output_path


def load_similarity_run_artifact(path: str | Path) -> SimilarityRunArtifact:
    payload = torch.load(Path(path), map_location="cpu")
    return SimilarityRunArtifact.from_dict(payload)


__all__ = [
    "load_similarity_run_artifact",
    "save_similarity_run_artifact",
]

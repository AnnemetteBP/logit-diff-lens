from __future__ import annotations

import torch


def recovery_score(
    reference_distance: torch.Tensor,
    patched_distance: torch.Tensor,
) -> torch.Tensor:
    return reference_distance - patched_distance


def residual_delta_direction(hidden_a: torch.Tensor, hidden_b: torch.Tensor, *, dim: int = 0) -> torch.Tensor:
    return (hidden_b.to(dtype=torch.float32) - hidden_a.to(dtype=torch.float32)).mean(dim=dim)


def apply_residual_delta(hidden: torch.Tensor, direction: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    return hidden + (alpha * direction)


__all__ = [
    "apply_residual_delta",
    "recovery_score",
    "residual_delta_direction",
]

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_distance(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    a = a.to(dtype=torch.float32)
    b = b.to(dtype=torch.float32)
    return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=eps)


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a.to(dtype=torch.float32) - b.to(dtype=torch.float32), ord=2, dim=-1)


def normalized_l2_distance(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    numerator = l2_distance(a, b)
    denominator = torch.linalg.vector_norm(a.to(dtype=torch.float32), ord=2, dim=-1).clamp_min(eps)
    return numerator / denominator


def norm_ratio(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    a_norm = torch.linalg.vector_norm(a.to(dtype=torch.float32), ord=2, dim=-1).clamp_min(eps)
    b_norm = torch.linalg.vector_norm(b.to(dtype=torch.float32), ord=2, dim=-1)
    return b_norm / a_norm


__all__ = [
    "cosine_distance",
    "l2_distance",
    "normalized_l2_distance",
    "norm_ratio",
]

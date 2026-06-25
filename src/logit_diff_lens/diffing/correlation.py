from __future__ import annotations

import torch


def pearson_correlation(x: torch.Tensor, y: torch.Tensor, *, dim: int = 0, eps: float = 1e-12) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    x = x - x.mean(dim=dim, keepdim=True)
    y = y - y.mean(dim=dim, keepdim=True)
    numerator = (x * y).sum(dim=dim)
    denominator = torch.sqrt((x.square().sum(dim=dim) * y.square().sum(dim=dim)).clamp_min(eps))
    return numerator / denominator


def spearman_correlation(x: torch.Tensor, y: torch.Tensor, *, dim: int = 0) -> torch.Tensor:
    x_rank = torch.argsort(torch.argsort(x, dim=dim), dim=dim).to(dtype=torch.float32)
    y_rank = torch.argsort(torch.argsort(y, dim=dim), dim=dim).to(dtype=torch.float32)
    return pearson_correlation(x_rank, y_rank, dim=dim)


def bootstrap_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    num_samples: int = 1000,
    method: str = "pearson",
) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)
    n = x.shape[0]
    indices = torch.randint(low=0, high=n, size=(num_samples, n), device=x.device)
    samples = []
    for row in indices:
        x_sample = x.index_select(0, row)
        y_sample = y.index_select(0, row)
        if method == "pearson":
            samples.append(pearson_correlation(x_sample, y_sample, dim=0))
        elif method == "spearman":
            samples.append(spearman_correlation(x_sample, y_sample, dim=0))
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
    return torch.stack(samples, dim=0)


__all__ = [
    "bootstrap_correlation",
    "pearson_correlation",
    "spearman_correlation",
]

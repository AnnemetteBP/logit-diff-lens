from __future__ import annotations

from typing import Literal

import torch


AggregationMode = Literal["mean", "median", "max"]


def aggregate_over_positions(values: torch.Tensor, *, mode: AggregationMode = "mean", dim: int = -1) -> torch.Tensor:
    values = values.to(dtype=torch.float32)
    if mode == "mean":
        return values.mean(dim=dim)
    if mode == "median":
        return values.median(dim=dim).values
    if mode == "max":
        return values.max(dim=dim).values
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def bootstrap_mean(
    values: torch.Tensor,
    *,
    num_samples: int = 1000,
    dim: int = 0,
) -> torch.Tensor:
    values = values.to(dtype=torch.float32)
    n = values.shape[dim]
    index_shape = [num_samples, n]
    indices = torch.randint(low=0, high=n, size=index_shape, device=values.device)
    sampled = torch.index_select(values, dim=dim, index=indices.reshape(-1))
    sampled = sampled.reshape(num_samples, n, *values.shape[1:])
    return sampled.mean(dim=1)


def layer_vs_final_correlation(layer_values: torch.Tensor, final_values: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    x = layer_values.to(dtype=torch.float32)
    y = final_values.to(dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    numerator = (x * y).sum(dim=0)
    denominator = torch.sqrt((x.square().sum(dim=0) * y.square().sum(dim=0)).clamp_min(eps))
    return numerator / denominator


__all__ = [
    "aggregate_over_positions",
    "bootstrap_mean",
    "layer_vs_final_correlation",
]

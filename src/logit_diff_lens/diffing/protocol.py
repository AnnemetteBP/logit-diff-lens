from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable

import torch


def protocol_condition_key(
    *,
    template: str | None,
    system_prompt: str | None,
    temperature: float | None,
    seed: int | None,
) -> tuple[str | None, str | None, float | None, int | None]:
    return (template, system_prompt, temperature, seed)


def aggregate_condition_means(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
) -> dict[Hashable, float]:
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["condition_key"]].append(float(row[metric_key]))
    return {key: float(sum(values) / len(values)) for key, values in grouped.items()}


def seed_variance(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
) -> dict[Hashable, float]:
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["condition_key"]].append(float(row[metric_key]))
    variances = {}
    for key, values in grouped.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        variances[key] = float(tensor.var(unbiased=False).item()) if len(values) > 1 else 0.0
    return variances


__all__ = [
    "aggregate_condition_means",
    "protocol_condition_key",
    "seed_variance",
]

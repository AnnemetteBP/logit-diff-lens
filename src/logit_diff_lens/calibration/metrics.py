from __future__ import annotations

import math
from dataclasses import dataclass

import torch


CalibrationBinning = str


@dataclass(frozen=True)
class CalibrationCellStats:
    ece: float
    accuracy_mean: float
    confidence_mean: float
    sample_count: int


def _as_float_vector(values: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(values):
        raise TypeError(f"Expected torch.Tensor, got {type(values)!r}")
    values = values.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if values.numel() == 0:
        return values
    if not torch.isfinite(values).all():
        raise ValueError("Calibration metric inputs must be finite")
    return values


def compute_expected_calibration_error(
    confidences: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    num_bins: int = 15,
    binning: CalibrationBinning = "equal_mass",
    min_samples: int = 1,
) -> CalibrationCellStats:
    conf = _as_float_vector(confidences)
    acc = _as_float_vector(outcomes)
    if conf.numel() != acc.numel():
        raise ValueError("confidences and outcomes must have the same number of samples")
    if conf.numel() < min_samples or conf.numel() == 0:
        return CalibrationCellStats(
            ece=float("nan"),
            accuracy_mean=float("nan"),
            confidence_mean=float("nan"),
            sample_count=int(conf.numel()),
        )
    if torch.any((conf < 0.0) | (conf > 1.0)):
        raise ValueError("confidences must lie in [0, 1]")
    if torch.any((acc < 0.0) | (acc > 1.0)):
        raise ValueError("outcomes must lie in [0, 1]")
    bins = max(int(num_bins), 1)

    if binning == "equal_mass":
        order = torch.argsort(conf)
        conf_sorted = conf[order]
        acc_sorted = acc[order]
        chunk_size = int(math.ceil(conf.numel() / bins))
        weighted_gap = 0.0
        for start in range(0, conf.numel(), chunk_size):
            stop = min(conf.numel(), start + chunk_size)
            conf_bin = conf_sorted[start:stop]
            acc_bin = acc_sorted[start:stop]
            if conf_bin.numel() == 0:
                continue
            weighted_gap += (conf_bin.numel() / conf.numel()) * abs(
                conf_bin.mean().item() - acc_bin.mean().item()
            )
    elif binning == "equal_width":
        edges = torch.linspace(0.0, 1.0, steps=bins + 1, dtype=torch.float32)
        weighted_gap = 0.0
        for idx in range(bins):
            left = edges[idx]
            right = edges[idx + 1]
            if idx == bins - 1:
                mask = (conf >= left) & (conf <= right)
            else:
                mask = (conf >= left) & (conf < right)
            if not mask.any():
                continue
            conf_bin = conf[mask]
            acc_bin = acc[mask]
            weighted_gap += (conf_bin.numel() / conf.numel()) * abs(
                conf_bin.mean().item() - acc_bin.mean().item()
            )
    else:
        raise ValueError(f"Unsupported calibration binning: {binning!r}")

    return CalibrationCellStats(
        ece=float(weighted_gap),
        accuracy_mean=float(acc.mean().item()),
        confidence_mean=float(conf.mean().item()),
        sample_count=int(conf.numel()),
    )


def multiclass_brier_score(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = probs.detach().to(dtype=torch.float32, device="cpu")
    labels = labels.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape [n, vocab], got {tuple(probs.shape)}")
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("labels must match probs sample count")
    target = torch.zeros_like(probs)
    target.scatter_(1, labels.unsqueeze(-1), 1.0)
    return torch.sum((probs - target) ** 2, dim=-1)


def negative_log_likelihood(probs: torch.Tensor, labels: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    probs = probs.detach().to(dtype=torch.float32, device="cpu")
    labels = labels.detach().to(dtype=torch.long, device="cpu").reshape(-1)
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape [n, vocab], got {tuple(probs.shape)}")
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("labels must match probs sample count")
    chosen = probs.gather(1, labels.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
    return -torch.log(chosen)


__all__ = [
    "CalibrationCellStats",
    "compute_expected_calibration_error",
    "multiclass_brier_score",
    "negative_log_likelihood",
]

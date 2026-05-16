from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from ..plotting.multi_plots.bootstrap_plots import (
    save_mds_boxplot,
    save_mds_errorbar_plot,
    save_roc_curve_plot,
)


def _as_numpy_1d(values: Sequence[float] | Sequence[int], *, name: str) -> np.ndarray:
    array = np.asarray(list(values))
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence")
    if not np.isfinite(array.astype(float)).all():
        raise ValueError(f"{name} contains NaN or inf values")
    return array


def _bootstrap_confidence_interval(samples: np.ndarray) -> list[float]:
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("bootstrap samples must be a non-empty 1D array")
    low = float(np.percentile(samples, 2.5))
    high = float(np.percentile(samples, 97.5))
    return [low, high]


def bootstrap_auroc(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, list[float]]:
    y = _as_numpy_1d(y_true, name="y_true").astype(int)
    s = _as_numpy_1d(scores, name="scores").astype(float)
    if y.shape[0] != s.shape[0]:
        raise ValueError(f"y_true and scores length mismatch: {y.shape[0]} vs {s.shape[0]}")
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be > 0, got {n_bootstrap}")

    rng = np.random.default_rng(seed)
    n = y.shape[0]
    boot_values = []
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, n, size=n)
        y_sample = y[indices]
        s_sample = s[indices]
        if np.unique(y_sample).size < 2:
            continue
        boot_values.append(float(roc_auc_score(y_sample, s_sample)))

    if not boot_values:
        raise ValueError("All bootstrap AUROC samples were single-class; cannot compute confidence interval")

    boot_array = np.asarray(boot_values, dtype=float)
    return float(np.mean(boot_array)), _bootstrap_confidence_interval(boot_array)


def bootstrap_mean_diff(
    group1: Sequence[float],
    group2: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, list[float]]:
    g1 = _as_numpy_1d(group1, name="group1").astype(float)
    g2 = _as_numpy_1d(group2, name="group2").astype(float)
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be > 0, got {n_bootstrap}")

    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(int(n_bootstrap)):
        sample1 = g1[rng.integers(0, g1.shape[0], size=g1.shape[0])]
        sample2 = g2[rng.integers(0, g2.shape[0], size=g2.shape[0])]
        boot_diffs.append(float(np.mean(sample1) - np.mean(sample2)))

    boot_array = np.asarray(boot_diffs, dtype=float)
    return float(np.mean(boot_array)), _bootstrap_confidence_interval(boot_array)




def summarize_bootstrap_statistics(
    y_true: Sequence[int],
    mds_scores: Sequence[float],
    self_report_scores: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    y = _as_numpy_1d(y_true, name="y_true").astype(int)
    mds = _as_numpy_1d(mds_scores, name="mds_scores").astype(float)
    self_scores = _as_numpy_1d(self_report_scores, name="self_report_scores").astype(float)
    if not (len(y) == len(mds) == len(self_scores)):
        raise ValueError("y_true, mds_scores, and self_report_scores must have the same length")

    harmful = mds[y == 1]
    neutral = mds[y == 0]
    if harmful.size == 0 or neutral.size == 0:
        raise ValueError("Both harmful and neutral groups must be non-empty")

    auroc_mds, ci_mds = bootstrap_auroc(y, mds, n_bootstrap=n_bootstrap, seed=seed)
    auroc_self, ci_self = bootstrap_auroc(
        y,
        self_scores,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )
    mean_diff, ci_diff = bootstrap_mean_diff(
        harmful,
        neutral,
        n_bootstrap=n_bootstrap,
        seed=seed + 2,
    )

    return {
        "auroc_mds": auroc_mds,
        "ci_mds": ci_mds,
        "auroc_self": auroc_self,
        "ci_self": ci_self,
        "mean_diff": mean_diff,
        "ci_diff": ci_diff,
    }

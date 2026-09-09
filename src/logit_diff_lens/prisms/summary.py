from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..schemas.prism_outputs import PromptDiffPrismHeatmapArtifact, PromptDiffPrismSummaryArtifact
from ..similarity.nulls import adjust_pvalues
from .prompt_diff import build_prompt_diff_prism_artifact
from .io import load_prompt_diff_prism_heatmap_artifact


def _load_diff_prism_heatmap(
    source: PromptDiffPrismHeatmapArtifact | str | Path,
) -> PromptDiffPrismHeatmapArtifact:
    if isinstance(source, PromptDiffPrismHeatmapArtifact):
        return source
    return load_prompt_diff_prism_heatmap_artifact(source)


def _validate_summary_kind(summary_kind: str) -> str:
    if summary_kind not in {"mean_abs", "max_abs"}:
        raise ValueError(f"Unsupported summary_kind={summary_kind!r}; expected 'mean_abs' or 'max_abs'")
    return summary_kind


def _component_scores(matrix: torch.Tensor, *, summary_kind: str) -> torch.Tensor:
    matrix = matrix.to(dtype=torch.float32, device="cpu")
    if matrix.ndim != 2:
        raise ValueError(f"Expected prism contribution matrix with shape [components, tokens], got {tuple(matrix.shape)}")
    summary_kind = _validate_summary_kind(summary_kind)
    if summary_kind == "mean_abs":
        scores = matrix.abs().mean(dim=1)
    else:
        scores = matrix.abs().max(dim=1).values
    if not torch.isfinite(scores).all():
        raise ValueError("Prism summary scores contain NaN or Inf values")
    return scores


def _signed_component_means(matrix: torch.Tensor) -> torch.Tensor:
    means = matrix.to(dtype=torch.float32, device="cpu").mean(dim=1)
    if not torch.isfinite(means).all():
        raise ValueError("Prism signed component means contain NaN or Inf values")
    return means


def _empirical_right_tail_threshold(observed: float, null_scores: np.ndarray, alpha: float) -> float:
    combined = np.concatenate([np.asarray([observed], dtype=np.float64), null_scores.astype(np.float64, copy=False)])
    return float(np.quantile(combined, 1.0 - alpha, method="higher"))


def _right_tail_p_value(observed: float, null_scores: np.ndarray) -> float:
    return float((1.0 + np.sum(null_scores >= observed)) / (len(null_scores) + 1.0))


def _calibrated_score(observed: float, threshold: float) -> float:
    if observed <= threshold:
        return 0.0
    return float(observed - threshold)


def build_prompt_diff_prism_summary_artifact(
    source: PromptDiffPrismHeatmapArtifact | str | Path,
    *,
    summary_kind: str = "mean_abs",
    null_sources: Sequence[PromptDiffPrismHeatmapArtifact | str | Path] | None = None,
    alpha: float = 0.05,
    multiple_testing_method: str = "fdr_bh",
) -> PromptDiffPrismSummaryArtifact:
    artifact = _load_diff_prism_heatmap(source)
    summary_kind = _validate_summary_kind(summary_kind)
    raw_component_scores = _component_scores(artifact.contribution_matrix, summary_kind=summary_kind)
    signed_component_means = _signed_component_means(artifact.contribution_matrix)

    calibrated_component_scores = None
    p_value_vector = None
    adjusted_p_value_vector = None
    critical_value_vector = None
    null_mean_vector = None
    null_std_vector = None

    if null_sources is not None:
        loaded_nulls = [_load_diff_prism_heatmap(item) for item in null_sources]
        if not loaded_nulls:
            raise ValueError("null_sources must be non-empty when provided")
        for idx, null_artifact in enumerate(loaded_nulls):
            if null_artifact.component_labels != artifact.component_labels:
                raise ValueError(
                    f"Null prism artifact {idx} has mismatched component labels; "
                    "all prism summaries must share the same component axis"
                )
        null_score_matrix = torch.stack(
            [_component_scores(item.contribution_matrix, summary_kind=summary_kind) for item in loaded_nulls],
            dim=0,
        )
        if not torch.isfinite(null_score_matrix).all():
            raise ValueError("Null prism summary scores contain NaN or Inf values")
        component_count = raw_component_scores.shape[0]
        calibrated_values: list[float] = []
        p_values: list[float] = []
        critical_values: list[float] = []
        null_means: list[float] = []
        null_stds: list[float] = []
        for component_idx in range(component_count):
            observed = float(raw_component_scores[component_idx].item())
            null_scores = null_score_matrix[:, component_idx].numpy().astype(np.float64, copy=False)
            critical = _empirical_right_tail_threshold(observed, null_scores, alpha)
            p_value = _right_tail_p_value(observed, null_scores)
            calibrated = _calibrated_score(observed, critical)
            calibrated_values.append(calibrated)
            p_values.append(p_value)
            critical_values.append(critical)
            null_means.append(float(null_scores.mean()))
            null_stds.append(float(null_scores.std(ddof=0)))
        adjusted = adjust_pvalues(p_values, multiple_testing_method)
        calibrated_component_scores = torch.tensor(calibrated_values, dtype=torch.float32)
        p_value_vector = torch.tensor(p_values, dtype=torch.float32)
        adjusted_p_value_vector = torch.tensor(adjusted, dtype=torch.float32)
        critical_value_vector = torch.tensor(critical_values, dtype=torch.float32)
        null_mean_vector = torch.tensor(null_means, dtype=torch.float32)
        null_std_vector = torch.tensor(null_stds, dtype=torch.float32)

    return PromptDiffPrismSummaryArtifact(
        prompt_text=artifact.prompt_text,
        prompt_formatted=artifact.prompt_formatted,
        prompt_id=artifact.prompt_id,
        readout_mode=artifact.readout_mode,
        position_index=artifact.position_index,
        side_ft_label=artifact.side_ft_label,
        side_base_label=artifact.side_base_label,
        delta_definition=artifact.delta_definition,
        summary_kind=summary_kind,
        component_labels=list(artifact.component_labels),
        raw_component_scores=raw_component_scores.to(torch.float32),
        signed_component_means=signed_component_means.to(torch.float32),
        selected_token_ids=artifact.selected_token_ids.to(dtype=torch.long, device="cpu"),
        selected_token_text=list(artifact.selected_token_text),
        calibrated_component_scores=calibrated_component_scores,
        p_value_vector=p_value_vector,
        adjusted_p_value_vector=adjusted_p_value_vector,
        critical_value_vector=critical_value_vector,
        null_mean_vector=null_mean_vector,
        null_std_vector=null_std_vector,
        backend_metadata_ft=artifact.backend_metadata_ft,
        backend_metadata_base=artifact.backend_metadata_base,
        metadata={
            "source_kind": "prompt_diff_prism_heatmap_artifact",
            "has_null_calibration": null_sources is not None,
            "null_sample_count": 0 if null_sources is None else len(null_sources),
            "multiple_testing_method": multiple_testing_method,
            **dict(artifact.metadata),
        },
    )


def build_prompt_diff_prism_summary_from_pair(
    ft_source,
    base_source,
    *,
    prompt_index: int = 0,
    readout_mode: str = "raw",
    top_k: int = 10,
    position_index: int | None = None,
    token_selection: str = "largest_abs_delta",
    side_ft_label: str = "ft",
    side_base_label: str = "base",
    summary_kind: str = "mean_abs",
) -> PromptDiffPrismSummaryArtifact:
    heatmap = build_prompt_diff_prism_artifact(
        ft_source,
        base_source,
        prompt_index=prompt_index,
        readout_mode=readout_mode,
        top_k=top_k,
        position_index=position_index,
        token_selection=token_selection,
        side_ft_label=side_ft_label,
        side_base_label=side_base_label,
    )
    return build_prompt_diff_prism_summary_artifact(heatmap, summary_kind=summary_kind)


__all__ = [
    "build_prompt_diff_prism_summary_artifact",
    "build_prompt_diff_prism_summary_from_pair",
]

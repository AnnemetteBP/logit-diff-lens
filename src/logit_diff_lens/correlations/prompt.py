from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from scipy import stats

from ..calibration.prompt import _load_prompt_source, _match_prompt_artifacts
from ..pair_validation import validate_prompt_artifact_pair
from ..schemas import PromptDecodeArtifact, PromptLayerRecord
from ..schemas.correlation_outputs import CorrelationRunArtifact, LayerPairCorrelationArtifact


PromptCorrelationAlignmentMode = Literal["same_token_ids", "shared_position_mask"]
PromptCorrelationMetric = Literal["target_probability", "top1_confidence", "entropy"]


@dataclass
class _AlignedPromptExample:
    artifact_a: PromptDecodeArtifact
    artifact_b: PromptDecodeArtifact
    valid_positions: torch.Tensor


def _layer_map(artifact: PromptDecodeArtifact) -> dict[int, PromptLayerRecord]:
    return {record.layer_index: record for record in artifact.layer_records}


def _resolve_valid_positions(
    artifact_a: PromptDecodeArtifact,
    artifact_b: PromptDecodeArtifact,
    *,
    alignment_mode: PromptCorrelationAlignmentMode,
    metric_name: PromptCorrelationMetric,
    readout_mode: str,
) -> torch.Tensor:
    if alignment_mode == "shared_position_mask" and metric_name == "target_probability":
        raise ValueError("target_probability requires same_token_ids alignment")
    valid = validate_prompt_artifact_pair(
        artifact_a,
        artifact_b,
        alignment_mode=alignment_mode,
        side_a_label="side_a",
        side_b_label="side_b",
        readout_mode=readout_mode,
    ).valid_mask[0]
    if metric_name == "target_probability":
        valid = valid.clone()
        valid[-1] = False
    return valid


def _get_logits(record: PromptLayerRecord, *, readout_mode: str) -> torch.Tensor:
    logits = record.get_logits(readout_mode)  # type: ignore[arg-type]
    if logits is None:
        raise ValueError(f"Missing logits_{readout_mode} for prompt layer {record.layer_name}")
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected prompt logits shape [1, seq, vocab], got {tuple(logits.shape)}")
    return logits[0].to(dtype=torch.float32, device="cpu")


def _metric_vector(
    artifact: PromptDecodeArtifact,
    record: PromptLayerRecord,
    positions: torch.Tensor,
    *,
    metric_name: PromptCorrelationMetric,
    readout_mode: str,
) -> torch.Tensor:
    logits = _get_logits(record, readout_mode=readout_mode)
    probs = torch.softmax(logits, dim=-1)
    pos = positions.to(dtype=torch.long, device="cpu")
    if metric_name == "top1_confidence":
        return torch.max(probs[pos], dim=-1).values
    if metric_name == "entropy":
        selected = probs[pos].clamp_min(1e-12)
        return -(selected * torch.log(selected)).sum(dim=-1)
    if metric_name == "target_probability":
        target_ids = artifact.token_ids[0, pos + 1].to(dtype=torch.long, device="cpu")
        return probs[pos].gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    raise ValueError(f"Unsupported metric_name: {metric_name!r}")


def _fisher_ci(corr: float, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    if not math.isfinite(corr) or n < 4 or abs(corr) >= 1.0:
        return float("nan"), float("nan")
    z = math.atanh(max(min(corr, 0.999999), -0.999999))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    return math.tanh(z - z_crit * se), math.tanh(z + z_crit * se)


def _fdr_bh(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    q = [float("nan")] * m
    running = 1.0
    for rank, (idx, p_value) in reversed(list(enumerate(indexed, start=1))):
        adj = min(running, (p_value * m) / rank)
        running = adj
        q[idx] = adj
    return q


def run_prompt_correlations(
    artifact_a: str | Path | PromptDecodeArtifact | dict[str, Any],
    artifact_b: str | Path | PromptDecodeArtifact | dict[str, Any],
    *,
    side_a_label: str = "artifact_a",
    side_b_label: str = "artifact_b",
    readout_mode: str = "model_norm",
    alignment_mode: PromptCorrelationAlignmentMode = "same_token_ids",
    metrics: list[PromptCorrelationMetric] | None = None,
    min_samples: int = 4,
    alpha: float = 0.05,
) -> CorrelationRunArtifact:
    metrics = metrics or ["target_probability", "top1_confidence", "entropy"]
    matched = _match_prompt_artifacts(_load_prompt_source(artifact_a), _load_prompt_source(artifact_b))
    if not matched:
        raise ValueError("No aligned prompt artifacts found for correlations")

    examples: list[_AlignedPromptExample] = []
    common_layers_a: set[int] | None = None
    common_layers_b: set[int] | None = None
    for art_a, art_b in matched:
        validated = validate_prompt_artifact_pair(
            art_a,
            art_b,
            alignment_mode=alignment_mode,
            side_a_label=side_a_label,
            side_b_label=side_b_label,
            readout_mode=readout_mode,
        )
        shared_a = set(validated.layer_indices)
        shared_b = set(validated.layer_indices)
        common_layers_a = shared_a if common_layers_a is None else (common_layers_a & shared_a)
        common_layers_b = shared_b if common_layers_b is None else (common_layers_b & shared_b)
        examples.append(
            _AlignedPromptExample(
                artifact_a=art_a,
                artifact_b=art_b,
                valid_positions=torch.empty(0, dtype=torch.long),
            )
        )
    layer_indices_a = sorted(common_layers_a or [])
    layer_indices_b = sorted(common_layers_b or [])
    if not layer_indices_a or not layer_indices_b:
        raise ValueError("Prompt correlations found no shared layers")

    matrix_results: list[LayerPairCorrelationArtifact] = []
    for metric_name in metrics:
        pearson_r = torch.full((len(layer_indices_a), len(layer_indices_b)), float("nan"), dtype=torch.float32)
        pearson_p = torch.full_like(pearson_r, float("nan"))
        pearson_ci_low = torch.full_like(pearson_r, float("nan"))
        pearson_ci_high = torch.full_like(pearson_r, float("nan"))
        spearman_r = torch.full_like(pearson_r, float("nan"))
        spearman_p = torch.full_like(pearson_r, float("nan"))
        spearman_ci_low = torch.full_like(pearson_r, float("nan"))
        spearman_ci_high = torch.full_like(pearson_r, float("nan"))
        sample_count = torch.zeros_like(pearson_r)

        pearson_p_list: list[float] = []
        spearman_p_list: list[float] = []
        pearson_coords: list[tuple[int, int]] = []
        spearman_coords: list[tuple[int, int]] = []

        for row_idx, layer_a in enumerate(layer_indices_a):
            for col_idx, layer_b in enumerate(layer_indices_b):
                values_a: list[float] = []
                values_b: list[float] = []
                for pair in matched:
                    art_a, art_b = pair
                    valid = _resolve_valid_positions(
                        art_a,
                        art_b,
                        alignment_mode=alignment_mode,
                        metric_name=metric_name,
                        readout_mode=readout_mode,
                    )
                    positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                    if positions.numel() == 0:
                        continue
                    vec_a = _metric_vector(
                        art_a,
                        _layer_map(art_a)[layer_a],
                        positions,
                        metric_name=metric_name,
                        readout_mode=readout_mode,
                    )
                    vec_b = _metric_vector(
                        art_b,
                        _layer_map(art_b)[layer_b],
                        positions,
                        metric_name=metric_name,
                        readout_mode=readout_mode,
                    )
                    values_a.extend(float(v) for v in vec_a.tolist())
                    values_b.extend(float(v) for v in vec_b.tolist())
                n = min(len(values_a), len(values_b))
                sample_count[row_idx, col_idx] = float(n)
                if n < min_samples:
                    continue
                lhs = values_a[:n]
                rhs = values_b[:n]
                pearson_stat = stats.pearsonr(lhs, rhs)
                spearman_stat = stats.spearmanr(lhs, rhs)
                pearson_r[row_idx, col_idx] = float(pearson_stat.statistic)
                pearson_p[row_idx, col_idx] = float(pearson_stat.pvalue)
                p_lo, p_hi = _fisher_ci(float(pearson_stat.statistic), n, alpha=alpha)
                pearson_ci_low[row_idx, col_idx] = p_lo
                pearson_ci_high[row_idx, col_idx] = p_hi
                spearman_r[row_idx, col_idx] = float(spearman_stat.statistic)
                spearman_p[row_idx, col_idx] = float(spearman_stat.pvalue)
                s_lo, s_hi = _fisher_ci(float(spearman_stat.statistic), n, alpha=alpha)
                spearman_ci_low[row_idx, col_idx] = s_lo
                spearman_ci_high[row_idx, col_idx] = s_hi
                pearson_p_list.append(float(pearson_stat.pvalue))
                pearson_coords.append((row_idx, col_idx))
                spearman_p_list.append(float(spearman_stat.pvalue))
                spearman_coords.append((row_idx, col_idx))

        pearson_q = torch.full_like(pearson_r, float("nan"))
        spearman_q = torch.full_like(spearman_r, float("nan"))
        for (row_idx, col_idx), q_value in zip(pearson_coords, _fdr_bh(pearson_p_list)):
            pearson_q[row_idx, col_idx] = q_value
        for (row_idx, col_idx), q_value in zip(spearman_coords, _fdr_bh(spearman_p_list)):
            spearman_q[row_idx, col_idx] = q_value

        matrix_results.append(
            LayerPairCorrelationArtifact(
                metric_name=metric_name,
                readout_mode=readout_mode,
                pearson_r_matrix=pearson_r,
                pearson_p_matrix=pearson_p,
                pearson_q_matrix=pearson_q,
                pearson_ci_low_matrix=pearson_ci_low,
                pearson_ci_high_matrix=pearson_ci_high,
                spearman_r_matrix=spearman_r,
                spearman_p_matrix=spearman_p,
                spearman_q_matrix=spearman_q,
                spearman_ci_low_matrix=spearman_ci_low,
                spearman_ci_high_matrix=spearman_ci_high,
                sample_count_matrix=sample_count,
                layer_indices_a=layer_indices_a,
                layer_indices_b=layer_indices_b,
                metadata={"min_samples": min_samples, "alpha": alpha},
            )
        )

    backend_a = _load_prompt_source(artifact_a)[0].backend_metadata
    backend_b = _load_prompt_source(artifact_b)[0].backend_metadata
    return CorrelationRunArtifact(
        side_a_label=side_a_label,
        side_b_label=side_b_label,
        artifact_family="prompt",
        readout_mode=readout_mode,
        alignment_mode=alignment_mode,
        backend_metadata_a=backend_a,
        backend_metadata_b=backend_b,
        matrix_results=matrix_results,
        metadata={
            "num_prompts": len(matched),
            "metrics": list(metrics),
            "min_samples": min_samples,
            "alpha": alpha,
        },
    )


__all__ = ["run_prompt_correlations"]

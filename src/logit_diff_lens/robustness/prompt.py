from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from scipy import stats

from ..calibration.metrics import compute_expected_calibration_error
from ..pair_validation import validate_prompt_artifact_pair
from ..prisms.prompt_diff import _infer_bias_from_saved_components
from ..schemas import PromptDecodeArtifact, PromptLayerRecord
from ..schemas.robustness_outputs import (
    RobustnessAgreementArtifact,
    RobustnessCalibrationArtifact,
    RobustnessProfileArtifact,
    RobustnessRunArtifact,
)
from ..similarity.alignment import _load_prompt_source, _match_prompt_artifacts
from ..similarity.metrics import js_similarity_from_logits, topk_overlap_similarity
from ..similarity.nulls import calibrate_scalar_similarity


PromptRobustnessMetric = Literal["jsd_divergence", "topk_overlap_divergence"]
PromptRobustnessAgreementTarget = Literal["top1_agreement", "topk_overlap"]
PromptRobustnessAlignmentMode = Literal["same_token_ids", "shared_position_mask"]


@dataclass(frozen=True)
class _PromptPair:
    artifact_a: PromptDecodeArtifact
    artifact_b: PromptDecodeArtifact
    positions: torch.Tensor


def _layer_map(artifact: PromptDecodeArtifact) -> dict[int, PromptLayerRecord]:
    return {record.layer_index: record for record in artifact.layer_records}


def _resolve_pairs(
    source_a: str | Path | PromptDecodeArtifact | dict[str, Any],
    source_b: str | Path | PromptDecodeArtifact | dict[str, Any],
    *,
    alignment_mode: PromptRobustnessAlignmentMode,
    readout_modes: list[str],
    include_prisms: bool,
    prism_readout_mode: str,
) -> tuple[list[_PromptPair], list[int], list[str], Any, Any]:
    artifacts_a = _load_prompt_source(source_a)
    artifacts_b = _load_prompt_source(source_b)
    matched = _match_prompt_artifacts(artifacts_a, artifacts_b)
    if not matched:
        raise ValueError("Prompt robustness found no matched prompt artifacts")

    common_layers: set[int] | None = None
    pairs: list[_PromptPair] = []
    prompt_labels: list[str] = []
    require_modes = list(readout_modes)
    if include_prisms and prism_readout_mode not in require_modes:
        require_modes.append(prism_readout_mode)

    for artifact_a, artifact_b in matched:
        for mode in require_modes:
            validate_prompt_artifact_pair(
                artifact_a,
                artifact_b,
                alignment_mode=alignment_mode,
                side_a_label="comparison",
                side_b_label="base",
                readout_mode=mode,
                require_component_logits=("attention", "mlp") if include_prisms and mode == prism_readout_mode else (),
                require_force_include_input=include_prisms and mode == prism_readout_mode,
                require_force_include_output=include_prisms and mode == prism_readout_mode,
            )
        validated = validate_prompt_artifact_pair(
            artifact_a,
            artifact_b,
            alignment_mode=alignment_mode,
            side_a_label="comparison",
            side_b_label="base",
        )
        positions = torch.nonzero(validated.valid_mask[0], as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            continue
        common_layers = set(validated.layer_indices) if common_layers is None else (common_layers & set(validated.layer_indices))
        pairs.append(_PromptPair(artifact_a=artifact_a, artifact_b=artifact_b, positions=positions))
        prompt_labels.append(artifact_a.prompt_id or artifact_a.prompt_text[:80])
    if not pairs:
        raise ValueError("Prompt robustness found no aligned prompts with valid positions")
    return pairs, sorted(common_layers or []), prompt_labels, artifacts_a[0].backend_metadata, artifacts_b[0].backend_metadata


def _bootstrap_profile_ci(values: torch.Tensor, *, num_bootstrap: int, alpha: float, seed: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError(f"Expected [num_prompts, num_layers], got {tuple(values.shape)}")
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    draws = []
    n = values.shape[0]
    for _ in range(num_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen)
        draws.append(values.index_select(0, idx).mean(dim=0))
    stacked = torch.stack(draws, dim=0)
    quantiles = torch.tensor([alpha / 2.0, 1.0 - alpha / 2.0], dtype=torch.float32)
    ci = torch.quantile(stacked, quantiles, dim=0)
    return ci[0], ci[1]


def _bootstrap_gap_ci(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    *,
    num_bootstrap: int,
    alpha: float,
    seed: int | None,
) -> tuple[float, float]:
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    draws = []
    n = values_a.shape[0]
    for _ in range(num_bootstrap):
        idx = torch.randint(0, n, (n,), generator=gen)
        mean_a = values_a.index_select(0, idx).mean(dim=0)
        mean_b = values_b.index_select(0, idx).mean(dim=0)
        draws.append(torch.abs(mean_a - mean_b).mean())
    samples = torch.stack(draws)
    quantiles = torch.tensor([alpha / 2.0, 1.0 - alpha / 2.0], dtype=torch.float32)
    ci = torch.quantile(samples, quantiles)
    return float(ci[0].item()), float(ci[1].item())


def _profile_metric(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    metric_name: PromptRobustnessMetric,
    top_k: int,
    num_permutations: int,
    alpha: float,
    seed: int | None,
) -> tuple[float, float]:
    if metric_name == "jsd_divergence":
        sim_fn = js_similarity_from_logits
    elif metric_name == "topk_overlap_divergence":
        sim_fn = lambda x, y: topk_overlap_similarity(x, y, k=top_k)
    else:
        raise ValueError(f"Unsupported prompt robustness metric: {metric_name!r}")
    raw_similarity = float(sim_fn(logits_a, logits_b))
    calibrated = calibrate_scalar_similarity(
        logits_a,
        logits_b,
        sim_fn,
        metric_name=metric_name,
        representation_kind="logits",
        num_permutations=num_permutations,
        alpha=alpha,
        similarity_max=1.0,
        permutation_unit="row",
        seed=seed,
    )
    return 1.0 - raw_similarity, 1.0 - float(calibrated.calibrated_similarity)


def _rowwise_similarity(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    metric_name: PromptRobustnessMetric,
    top_k: int,
) -> torch.Tensor:
    logits_a = logits_a.detach().to(dtype=torch.float32, device="cpu")
    logits_b = logits_b.detach().to(dtype=torch.float32, device="cpu")
    if logits_a.ndim != 2 or logits_b.ndim != 2 or logits_a.shape != logits_b.shape:
        raise ValueError(f"Rowwise similarity requires matching [n, vocab] tensors, got {tuple(logits_a.shape)} and {tuple(logits_b.shape)}")
    if metric_name == "jsd_divergence":
        probs_a = torch.softmax(logits_a, dim=-1)
        probs_b = torch.softmax(logits_b, dim=-1)
        m = 0.5 * (probs_a + probs_b)
        eps = 1e-12
        probs_a = probs_a.clamp_min(eps)
        probs_b = probs_b.clamp_min(eps)
        m = m.clamp_min(eps)
        js = 0.5 * (
            torch.sum(probs_a * (torch.log(probs_a) - torch.log(m)), dim=-1)
            + torch.sum(probs_b * (torch.log(probs_b) - torch.log(m)), dim=-1)
        )
        js = js.clamp_min(0.0).clamp_max(math.log(2.0))
        return 1.0 - (js / math.log(2.0))
    if metric_name == "topk_overlap_divergence":
        vocab = logits_a.shape[-1]
        k_eff = min(top_k, vocab)
        idx_a = torch.topk(logits_a, k=k_eff, dim=-1).indices.tolist()
        idx_b = torch.topk(logits_b, k=k_eff, dim=-1).indices.tolist()
        vals = []
        for row_a, row_b in zip(idx_a, idx_b):
            set_a = set(int(v) for v in row_a)
            set_b = set(int(v) for v in row_b)
            union = set_a | set_b
            vals.append(0.0 if not union else len(set_a & set_b) / len(union))
        return torch.tensor(vals, dtype=torch.float32)
    raise ValueError(f"Unsupported prompt robustness metric: {metric_name!r}")


def _rowwise_top1_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    logits_a = logits_a.detach().to(dtype=torch.float32, device="cpu")
    logits_b = logits_b.detach().to(dtype=torch.float32, device="cpu")
    if logits_a.ndim != 2 or logits_b.ndim != 2 or logits_a.shape != logits_b.shape:
        raise ValueError(f"Top-1 agreement requires matching [n, vocab] tensors, got {tuple(logits_a.shape)} and {tuple(logits_b.shape)}")
    top1_a = torch.argmax(logits_a, dim=-1)
    top1_b = torch.argmax(logits_b, dim=-1)
    return (top1_a == top1_b).to(dtype=torch.float32)


def _rowwise_agreement_target(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    agreement_target: PromptRobustnessAgreementTarget,
    top_k: int,
) -> torch.Tensor:
    if agreement_target == "top1_agreement":
        return _rowwise_top1_agreement(logits_a, logits_b)
    if agreement_target == "topk_overlap":
        return _rowwise_similarity(
            logits_a,
            logits_b,
            metric_name="topk_overlap_divergence",
            top_k=top_k,
        )
    raise ValueError(f"Unsupported prompt robustness agreement_target: {agreement_target!r}")


def _sanitize_topk_values(top_k_values: list[int] | None, *, required_top_k: int) -> list[int]:
    values = [int(v) for v in (top_k_values or [1, 5, 10]) if int(v) > 0]
    if required_top_k > 0:
        values.append(int(required_top_k))
    return sorted(set(values))


def _build_prism_cumulative_logits(
    artifact: PromptDecodeArtifact,
    *,
    positions: torch.Tensor,
    readout_mode: str,
    shared_layers: list[int],
) -> dict[int, torch.Tensor]:
    layer_map = _layer_map(artifact)
    if -1 not in layer_map:
        raise ValueError("Prism robustness requires force-included embedding row")
    output_layer_index = max(shared_layers)
    embedding_logits = layer_map[-1].get_logits(readout_mode)
    total_logits = layer_map[output_layer_index].get_logits(readout_mode)
    if embedding_logits is None or total_logits is None:
        raise ValueError(f"Prism robustness missing logits for readout_mode={readout_mode!r}")
    attn_rows: list[torch.Tensor] = []
    mlp_rows: list[torch.Tensor] = []
    block_indices = [idx for idx in shared_layers if idx >= 0 and layer_map[idx].layer_name != "output"]
    for layer_index in block_indices:
        attn = layer_map[layer_index].get_component_logits("attention", readout_mode)
        mlp = layer_map[layer_index].get_component_logits("mlp", readout_mode)
        if attn is None or mlp is None:
            raise ValueError(f"Prism robustness missing component logits at layer {layer_index} for mode={readout_mode!r}")
        attn_rows.append(attn[0].to(dtype=torch.float32, device="cpu"))
        mlp_rows.append(mlp[0].to(dtype=torch.float32, device="cpu"))
    embedding_cpu = embedding_logits[0].to(dtype=torch.float32, device="cpu")
    total_cpu = total_logits[0].to(dtype=torch.float32, device="cpu")
    bias = _infer_bias_from_saved_components(embedding_cpu, attn_rows, mlp_rows, total_cpu)
    cumulative_by_layer: dict[int, torch.Tensor] = {}
    cumulative = embedding_cpu - bias
    cumulative_by_layer[-1] = cumulative.index_select(0, positions).clone()
    for layer_index, attn, mlp in zip(block_indices, attn_rows, mlp_rows):
        cumulative = cumulative + (attn - bias) + (mlp - bias)
        cumulative_by_layer[layer_index] = cumulative.index_select(0, positions).clone()
    cumulative_by_layer[output_layer_index] = total_cpu.index_select(0, positions).clone()
    return cumulative_by_layer


def _fisher_ci(corr: float, n: int, *, alpha: float) -> tuple[float, float]:
    if not math.isfinite(corr) or n < 4 or abs(corr) >= 1.0:
        return float("nan"), float("nan")
    z = math.atanh(max(min(corr, 0.999999), -0.999999))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    return math.tanh(z - z_crit * se), math.tanh(z + z_crit * se)


def _build_profile_artifact(
    *,
    metric_name: str,
    lens_family: str,
    regime_name: str,
    layer_indices: list[int],
    prompt_values: torch.Tensor,
    num_bootstrap: int,
    alpha: float,
    seed: int | None,
    metadata: dict[str, Any],
) -> RobustnessProfileArtifact:
    mean_values = prompt_values.mean(dim=0)
    ci_low, ci_high = _bootstrap_profile_ci(prompt_values, num_bootstrap=num_bootstrap, alpha=alpha, seed=seed)
    sample_counts = torch.full((prompt_values.shape[1],), int(prompt_values.shape[0]), dtype=torch.float32)
    return RobustnessProfileArtifact(
        metric_name=metric_name,
        lens_family=lens_family,
        regime_name=regime_name,
        layer_indices=layer_indices,
        prompt_values=prompt_values.to(torch.float32),
        mean_values=mean_values.to(torch.float32),
        ci_low=ci_low.to(torch.float32),
        ci_high=ci_high.to(torch.float32),
        sample_counts=sample_counts,
        metadata=metadata,
    )


def _build_calibration_artifact(
    *,
    metric_name: str,
    event_name: str,
    lens_family: str,
    regime_name: str,
    layer_indices: list[int],
    prompt_scores: torch.Tensor,
    prompt_outcomes: torch.Tensor,
    metadata: dict[str, Any],
) -> RobustnessCalibrationArtifact:
    if prompt_scores.shape != prompt_outcomes.shape or prompt_scores.ndim != 2:
        raise ValueError(
            f"Expected prompt_scores and prompt_outcomes with matching [num_prompts, num_layers] shapes, got "
            f"{tuple(prompt_scores.shape)} and {tuple(prompt_outcomes.shape)}"
        )
    ece_values = torch.full((prompt_scores.shape[1],), float("nan"), dtype=torch.float32)
    brier_values = torch.full_like(ece_values, float("nan"))
    sample_counts = torch.zeros_like(ece_values)
    confidence_means = torch.full_like(ece_values, float("nan"))
    outcome_means = torch.full_like(ece_values, float("nan"))
    for idx in range(prompt_scores.shape[1]):
        scores = prompt_scores[:, idx]
        outcomes = prompt_outcomes[:, idx]
        mask = torch.isfinite(scores) & torch.isfinite(outcomes)
        scores = scores[mask]
        outcomes = outcomes[mask]
        sample_counts[idx] = float(scores.numel())
        if scores.numel() == 0:
            continue
        stats_cell = compute_expected_calibration_error(scores, outcomes, num_bins=15, binning="equal_mass", min_samples=1)
        ece_values[idx] = float(stats_cell.ece)
        brier_values[idx] = float(torch.mean((scores - outcomes) ** 2).item())
        confidence_means[idx] = float(scores.mean().item())
        outcome_means[idx] = float(outcomes.mean().item())
    return RobustnessCalibrationArtifact(
        metric_name=metric_name,
        event_name=event_name,
        lens_family=lens_family,
        regime_name=regime_name,
        layer_indices=layer_indices,
        ece_values=ece_values,
        brier_values=brier_values,
        sample_counts=sample_counts,
        confidence_means=confidence_means,
        outcome_means=outcome_means,
        metadata=metadata,
    )


def _agreement_artifact(
    profile_a: RobustnessProfileArtifact,
    profile_b: RobustnessProfileArtifact,
    *,
    alpha: float,
    num_bootstrap: int,
    seed: int | None,
    metadata: dict[str, Any],
) -> RobustnessAgreementArtifact:
    layer_to_col_a = {layer: idx for idx, layer in enumerate(profile_a.layer_indices)}
    layer_to_col_b = {layer: idx for idx, layer in enumerate(profile_b.layer_indices)}
    shared_layers = [layer for layer in profile_a.layer_indices if layer in layer_to_col_b]
    if not shared_layers:
        raise ValueError(f"No shared layer indices between {profile_a.lens_family} and {profile_b.lens_family}")
    pearson_r = torch.full((len(shared_layers),), float("nan"), dtype=torch.float32)
    pearson_p = torch.full_like(pearson_r, float("nan"))
    spearman_r = torch.full_like(pearson_r, float("nan"))
    spearman_p = torch.full_like(pearson_r, float("nan"))
    sample_counts = torch.zeros_like(pearson_r)
    mean_curve_a = []
    mean_curve_b = []
    for idx, layer in enumerate(shared_layers):
        col_a = layer_to_col_a[layer]
        col_b = layer_to_col_b[layer]
        vals_a = profile_a.prompt_values[:, col_a]
        vals_b = profile_b.prompt_values[:, col_b]
        mask = torch.isfinite(vals_a) & torch.isfinite(vals_b)
        vals_a = vals_a[mask]
        vals_b = vals_b[mask]
        sample_counts[idx] = float(vals_a.numel())
        if vals_a.numel() >= 4:
            vals_a_cpu = vals_a.to(dtype=torch.float32, device="cpu")
            vals_b_cpu = vals_b.to(dtype=torch.float32, device="cpu")
            std_a = float(vals_a_cpu.std(unbiased=False).item())
            std_b = float(vals_b_cpu.std(unbiased=False).item())
            if std_a > 0.0 and std_b > 0.0:
                p_stat = stats.pearsonr(vals_a_cpu.tolist(), vals_b_cpu.tolist())
                s_stat = stats.spearmanr(vals_a_cpu.tolist(), vals_b_cpu.tolist())
                pearson_r[idx] = float(p_stat.statistic)
                pearson_p[idx] = float(p_stat.pvalue)
                spearman_r[idx] = float(s_stat.statistic)
                spearman_p[idx] = float(s_stat.pvalue)
        mean_curve_a.append(float(profile_a.mean_values[col_a].item()))
        mean_curve_b.append(float(profile_b.mean_values[col_b].item()))
    mean_curve_a_t = torch.tensor(mean_curve_a, dtype=torch.float32)
    mean_curve_b_t = torch.tensor(mean_curve_b, dtype=torch.float32)
    gap = torch.abs(mean_curve_a_t - mean_curve_b_t)
    gap_ci_low, gap_ci_high = _bootstrap_gap_ci(
        profile_a.prompt_values[:, [layer_to_col_a[layer] for layer in shared_layers]],
        profile_b.prompt_values[:, [layer_to_col_b[layer] for layer in shared_layers]],
        num_bootstrap=num_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    return RobustnessAgreementArtifact(
        metric_name=profile_a.metric_name,
        regime_name=profile_a.regime_name,
        lens_family_a=profile_a.lens_family,
        lens_family_b=profile_b.lens_family,
        layer_indices=shared_layers,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        sample_counts=sample_counts,
        aggregate_pearson_mean=float(torch.nanmean(pearson_r).item()),
        aggregate_spearman_mean=float(torch.nanmean(spearman_r).item()),
        mean_gap=float(gap.mean().item()),
        gap_ci_low=gap_ci_low,
        gap_ci_high=gap_ci_high,
        peak_gap_layer_index=shared_layers[int(torch.argmax(gap).item())] if gap.numel() else None,
        metadata=metadata,
    )


def run_prompt_robustness(
    artifact_a: str | Path | PromptDecodeArtifact | dict[str, Any],
    artifact_b: str | Path | PromptDecodeArtifact | dict[str, Any],
    *,
    side_a_label: str = "comparison",
    side_b_label: str = "base",
    metric_name: PromptRobustnessMetric = "jsd_divergence",
    alignment_mode: PromptRobustnessAlignmentMode = "same_token_ids",
    include_lens_families: list[str] | None = None,
    prism_readout_mode: str = "raw",
    top_k: int = 5,
    top_k_values: list[int] | None = None,
    agreement_target: PromptRobustnessAgreementTarget = "topk_overlap",
    num_permutations: int = 200,
    num_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int | None = None,
) -> RobustnessRunArtifact:
    include_lens_families = include_lens_families or ["raw", "model_norm", "tuned", "prisms"]
    top_k_values = _sanitize_topk_values(top_k_values, required_top_k=top_k)
    family_mode_map = {"raw": "raw", "model_norm": "model_norm", "tuned": "tuned"}
    requested_modes = [family_mode_map[family] for family in include_lens_families if family in family_mode_map]
    include_prisms = "prisms" in include_lens_families
    pairs, shared_layers, prompt_labels, backend_a, backend_b = _resolve_pairs(
        artifact_a,
        artifact_b,
        alignment_mode=alignment_mode,
        readout_modes=requested_modes,
        include_prisms=include_prisms,
        prism_readout_mode=prism_readout_mode,
    )
    if not shared_layers:
        raise ValueError("Prompt robustness found no shared layer axis")

    metric_configs: list[tuple[str, PromptRobustnessMetric, int | None]] = []
    if metric_name == "jsd_divergence":
        metric_configs.append(("jsd_divergence", "jsd_divergence", None))
    elif metric_name == "topk_overlap_divergence":
        for top_k_value in top_k_values:
            metric_configs.append((f"top{top_k_value}_overlap_divergence", "topk_overlap_divergence", top_k_value))
    else:
        raise ValueError(f"Unsupported prompt robustness metric: {metric_name!r}")

    profile_store: dict[tuple[str, str, str], list[list[float]]] = {}
    score_store: dict[tuple[str, str, str], list[list[float]]] = {}
    outcome_store: dict[tuple[str, int], dict[str, list[list[float]]]] = {}
    layer_index_store: dict[tuple[str, str], list[int]] = {}

    for family in include_lens_families:
        if family == "prisms":
            family_layers = [-1] + [idx for idx in shared_layers if idx >= 0]
        else:
            family_layers = list(shared_layers)
        for metric_label, _, _ in metric_configs:
            layer_index_store[(family, metric_label)] = family_layers
            for regime in ("non_calibrated", "calibrated"):
                profile_store[(family, regime, metric_label)] = []
                score_store[(family, regime, metric_label)] = []
        for top_k_value in top_k_values:
            outcome_store[(family, top_k_value)] = {regime: [] for regime in ("non_calibrated", "calibrated")}

    for prompt_idx, pair in enumerate(pairs):
        layer_map_a = _layer_map(pair.artifact_a)
        layer_map_b = _layer_map(pair.artifact_b)
        prism_logits_a = None
        prism_logits_b = None
        if include_prisms:
            prism_logits_a = _build_prism_cumulative_logits(
                pair.artifact_a,
                positions=pair.positions,
                readout_mode=prism_readout_mode,
                shared_layers=shared_layers,
            )
            prism_logits_b = _build_prism_cumulative_logits(
                pair.artifact_b,
                positions=pair.positions,
                readout_mode=prism_readout_mode,
                shared_layers=shared_layers,
            )

        for family in include_lens_families:
            selected_logits_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            if family == "prisms":
                assert prism_logits_a is not None and prism_logits_b is not None
                for layer_idx in layer_index_store[(family, metric_configs[0][0])]:
                    selected_logits_by_layer[layer_idx] = (prism_logits_a[layer_idx], prism_logits_b[layer_idx])
            else:
                mode = family_mode_map[family]
                for layer_idx in layer_index_store[(family, metric_configs[0][0])]:
                    logits_a = layer_map_a[layer_idx].get_logits(mode)
                    logits_b = layer_map_b[layer_idx].get_logits(mode)
                    if logits_a is None or logits_b is None:
                        raise ValueError(f"Missing logits for family={family!r} at layer={layer_idx}")
                    selected_logits_by_layer[layer_idx] = (
                        logits_a[0].index_select(0, pair.positions).to(dtype=torch.float32, device="cpu"),
                        logits_b[0].index_select(0, pair.positions).to(dtype=torch.float32, device="cpu"),
                    )

            per_topk_outcomes: dict[int, list[float]] = {top_k_value: [] for top_k_value in top_k_values}
            for layer_idx in layer_index_store[(family, metric_configs[0][0])]:
                selected_a, selected_b = selected_logits_by_layer[layer_idx]
                for top_k_value in top_k_values:
                    rowwise_outcomes = _rowwise_agreement_target(
                        selected_a,
                        selected_b,
                        agreement_target=agreement_target,
                        top_k=top_k_value,
                    )
                    per_topk_outcomes[top_k_value].append(float(rowwise_outcomes.mean().item()))

            for metric_label, metric_base, metric_top_k in metric_configs:
                raw_profile: list[float] = []
                cal_profile: list[float] = []
                raw_scores: list[float] = []
                cal_scores: list[float] = []
                effective_top_k = top_k if metric_top_k is None else metric_top_k
                for layer_idx in layer_index_store[(family, metric_label)]:
                    selected_a, selected_b = selected_logits_by_layer[layer_idx]
                    rowwise_scores = _rowwise_similarity(
                        selected_a,
                        selected_b,
                        metric_name=metric_base,
                        top_k=effective_top_k,
                    )
                    raw_value, cal_value = _profile_metric(
                        selected_a,
                        selected_b,
                        metric_name=metric_base,
                        top_k=effective_top_k,
                        num_permutations=num_permutations,
                        alpha=alpha,
                        seed=None if seed is None else seed + prompt_idx + max(layer_idx, 0) + effective_top_k,
                    )
                    raw_profile.append(raw_value)
                    cal_profile.append(cal_value)
                    raw_scores.append(float(rowwise_scores.mean().item()))
                    cal_scores.append(cal_value)
                profile_store[(family, "non_calibrated", metric_label)].append(raw_profile)
                profile_store[(family, "calibrated", metric_label)].append(cal_profile)
                score_store[(family, "non_calibrated", metric_label)].append(raw_scores)
                score_store[(family, "calibrated", metric_label)].append(cal_scores)
                if metric_base == "topk_overlap_divergence":
                    outcome_store[(family, effective_top_k)]["non_calibrated"].append(per_topk_outcomes[effective_top_k])
                    outcome_store[(family, effective_top_k)]["calibrated"].append(per_topk_outcomes[effective_top_k])
            if metric_name == "jsd_divergence":
                for top_k_value in top_k_values:
                    outcome_store[(family, top_k_value)]["non_calibrated"].append(per_topk_outcomes[top_k_value])
                    outcome_store[(family, top_k_value)]["calibrated"].append(per_topk_outcomes[top_k_value])

    profiles: list[RobustnessProfileArtifact] = []
    profile_lookup: dict[tuple[str, str, str], RobustnessProfileArtifact] = {}
    for family in include_lens_families:
        for metric_label, _, metric_top_k in metric_configs:
            for regime in ("non_calibrated", "calibrated"):
                prompt_values = torch.tensor(profile_store[(family, regime, metric_label)], dtype=torch.float32)
                artifact = _build_profile_artifact(
                    metric_name=metric_label,
                    lens_family=family,
                    regime_name=regime,
                    layer_indices=layer_index_store[(family, metric_label)],
                    prompt_values=prompt_values,
                    num_bootstrap=num_bootstrap,
                    alpha=alpha,
                    seed=seed,
                    metadata={
                        "prompt_labels": prompt_labels,
                        "top_k": (top_k if metric_top_k is None else metric_top_k),
                        "top_k_values": top_k_values,
                        "prism_readout_mode": prism_readout_mode,
                    },
                )
                profiles.append(artifact)
                profile_lookup[(family, regime, metric_label)] = artifact

    agreement_calibration_results: list[RobustnessCalibrationArtifact] = []
    for family in include_lens_families:
        for metric_label, metric_base, metric_top_k in metric_configs:
            if metric_base == "topk_overlap_divergence":
                calibration_top_ks = [top_k if metric_top_k is None else metric_top_k]
            else:
                calibration_top_ks = list(top_k_values)
            for calibration_top_k in calibration_top_ks:
                prompt_outcomes = torch.tensor(
                    outcome_store[(family, calibration_top_k)]["non_calibrated"],
                    dtype=torch.float32,
                )
                for regime in ("non_calibrated", "calibrated"):
                    prompt_scores = torch.tensor(score_store[(family, regime, metric_label)], dtype=torch.float32)
                    agreement_calibration_results.append(
                        _build_calibration_artifact(
                            metric_name=metric_label,
                            event_name=(
                                "top1_agreement_rate"
                                if agreement_target == "top1_agreement"
                                else f"top{calibration_top_k}_overlap_rate"
                            ),
                            lens_family=family,
                            regime_name=regime,
                            layer_indices=layer_index_store[(family, metric_label)],
                            prompt_scores=prompt_scores,
                            prompt_outcomes=prompt_outcomes,
                            metadata={
                                "top_k": (top_k if metric_top_k is None else metric_top_k),
                                "agreement_top_k": calibration_top_k,
                                "top_k_values": top_k_values,
                                "agreement_target": agreement_target,
                            },
                        )
                    )

    within_lens_results: list[RobustnessAgreementArtifact] = []
    for family in include_lens_families:
        for metric_label, _, metric_top_k in metric_configs:
            within_lens_results.append(
                _agreement_artifact(
                    profile_lookup[(family, "non_calibrated", metric_label)],
                    profile_lookup[(family, "calibrated", metric_label)],
                    alpha=alpha,
                    num_bootstrap=num_bootstrap,
                    seed=seed,
                    metadata={
                        "comparison_kind": "within_lens_calibration",
                        "top_k": (top_k if metric_top_k is None else metric_top_k),
                        "top_k_values": top_k_values,
                    },
                )
            )

    cross_lens_results: list[RobustnessAgreementArtifact] = []
    for metric_label, _, metric_top_k in metric_configs:
        for regime in ("non_calibrated", "calibrated"):
            for idx, family_a in enumerate(include_lens_families):
                for family_b in include_lens_families[idx + 1 :]:
                    cross_lens_results.append(
                        _agreement_artifact(
                            profile_lookup[(family_a, regime, metric_label)],
                            profile_lookup[(family_b, regime, metric_label)],
                            alpha=alpha,
                            num_bootstrap=num_bootstrap,
                            seed=seed,
                            metadata={
                                "comparison_kind": "cross_lens_same_regime",
                                "top_k": (top_k if metric_top_k is None else metric_top_k),
                                "top_k_values": top_k_values,
                            },
                        )
                    )

    return RobustnessRunArtifact(
        side_a_label=side_a_label,
        side_b_label=side_b_label,
        artifact_family="prompt",
        metric_name=metric_name,
        alignment_mode=alignment_mode,
        backend_metadata_a=backend_a,
        backend_metadata_b=backend_b,
        profiles=profiles,
        within_lens_results=within_lens_results,
        cross_lens_results=cross_lens_results,
        agreement_calibration_results=agreement_calibration_results,
        metadata={
            "top_k": top_k,
            "top_k_values": top_k_values,
            "num_permutations": num_permutations,
            "num_bootstrap": num_bootstrap,
            "alpha": alpha,
            "prompt_count": len(prompt_labels),
            "included_lens_families": include_lens_families,
            "prism_readout_mode": prism_readout_mode,
            "agreement_target": agreement_target,
        },
    )


__all__ = ["run_prompt_robustness"]

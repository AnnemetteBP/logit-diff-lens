from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from ..diffing.io import load_prompt_decode_artifact, load_prompt_decode_artifact_bundle
from ..pair_validation import validate_prompt_artifact_pair
from ..schemas import PromptDecodeArtifact, PromptLayerRecord
from ..schemas.calibration_outputs import (
    CalibrationMatrixResult,
    CalibrationRunArtifact,
    CalibrationSummaryResult,
)
from .metrics import compute_expected_calibration_error, multiclass_brier_score, negative_log_likelihood


PromptCalibrationAlignmentMode = Literal["same_token_ids", "shared_position_mask"]
PromptReferenceSource = Literal["final_layer", "matching_layer"]


@dataclass
class _PromptAlignedExample:
    prompt_id: str | None
    prompt_text: str
    positions: torch.Tensor
    layer_records_a: dict[int, PromptLayerRecord]
    layer_records_b: dict[int, PromptLayerRecord]


@dataclass
class _PromptMetricAccumulator:
    top1_conf: list[float] = field(default_factory=list)
    top1_hit: list[float] = field(default_factory=list)
    target_prob: list[float] = field(default_factory=list)
    brier: list[float] = field(default_factory=list)
    nll: list[float] = field(default_factory=list)
    topk_conf: dict[int, list[float]] = field(default_factory=dict)
    topk_hit: dict[int, list[float]] = field(default_factory=dict)

    def add(
        self,
        *,
        probs: torch.Tensor,
        label: int,
        top_k_values: list[int],
    ) -> None:
        probs = probs.detach().to(dtype=torch.float32, device="cpu")
        label_idx = int(label)
        top1_conf, top1_idx = torch.max(probs, dim=-1)
        self.top1_conf.append(float(top1_conf.clamp(0.0, 1.0).item()))
        self.top1_hit.append(float(int(top1_idx.item() == label_idx)))
        self.target_prob.append(float(probs[label_idx].item()))
        self.brier.append(float(multiclass_brier_score(probs.unsqueeze(0), torch.tensor([label_idx])).item()))
        self.nll.append(float(negative_log_likelihood(probs.unsqueeze(0), torch.tensor([label_idx])).item()))
        for k in top_k_values:
            k_eff = min(int(k), int(probs.shape[0]))
            topk = torch.topk(probs, k=k_eff, dim=-1)
            indices = topk.indices.tolist()
            # Top-k confidence is a probability mass and should stay in [0, 1];
            # clamp tiny floating-point drift before downstream calibration.
            self.topk_conf.setdefault(k, []).append(float(topk.values.sum().clamp(0.0, 1.0).item()))
            self.topk_hit.setdefault(k, []).append(float(int(label_idx in indices)))


def _load_prompt_source(
    source: str | Path | PromptDecodeArtifact | dict[str, Any],
) -> list[PromptDecodeArtifact]:
    if isinstance(source, PromptDecodeArtifact):
        return [source]
    if isinstance(source, dict):
        if "artifacts" in source:
            return [PromptDecodeArtifact.from_dict(item) if isinstance(item, dict) else item for item in source["artifacts"]]
        return [PromptDecodeArtifact.from_dict(source)]
    path = Path(source)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "artifacts" in payload:
        bundle = load_prompt_decode_artifact_bundle(path)
        return bundle["artifacts"]
    return [load_prompt_decode_artifact(path)]


def _match_prompt_artifacts(
    artifacts_a: list[PromptDecodeArtifact],
    artifacts_b: list[PromptDecodeArtifact],
) -> list[tuple[PromptDecodeArtifact, PromptDecodeArtifact]]:
    if len(artifacts_a) == 1 and len(artifacts_b) == 1:
        return [(artifacts_a[0], artifacts_b[0])]
    keyed_a = {
        artifact.prompt_id if artifact.prompt_id is not None else artifact.prompt_text: artifact
        for artifact in artifacts_a
    }
    keyed_b = {
        artifact.prompt_id if artifact.prompt_id is not None else artifact.prompt_text: artifact
        for artifact in artifacts_b
    }
    shared = [key for key in keyed_a if key in keyed_b]
    if not shared:
        raise ValueError("No shared prompt ids/texts found between the two prompt sources")
    return [(keyed_a[key], keyed_b[key]) for key in shared]


def _valid_prompt_positions(
    artifact_a: PromptDecodeArtifact,
    artifact_b: PromptDecodeArtifact,
    *,
    alignment_mode: PromptCalibrationAlignmentMode,
) -> torch.Tensor:
    result = validate_prompt_artifact_pair(
        artifact_a,
        artifact_b,
        alignment_mode=alignment_mode,
        side_a_label="evaluation",
        side_b_label="reference",
    )
    return result.valid_mask[0]


def _layer_map(artifact: PromptDecodeArtifact) -> dict[int, PromptLayerRecord]:
    return {record.layer_index: record for record in artifact.layer_records}


def _build_prompt_examples(
    source_a: str | Path | PromptDecodeArtifact | dict[str, Any],
    source_b: str | Path | PromptDecodeArtifact | dict[str, Any],
    *,
    alignment_mode: PromptCalibrationAlignmentMode,
) -> tuple[list[_PromptAlignedExample], list[int], list[int]]:
    matched = _match_prompt_artifacts(_load_prompt_source(source_a), _load_prompt_source(source_b))
    examples: list[_PromptAlignedExample] = []
    all_positions: set[int] = set()
    common_layers: set[int] | None = None
    for artifact_a, artifact_b in matched:
        valid_positions = torch.nonzero(
            _valid_prompt_positions(artifact_a, artifact_b, alignment_mode=alignment_mode),
            as_tuple=False,
        ).squeeze(-1)
        if valid_positions.numel() == 0:
            continue
        layer_map_a = _layer_map(artifact_a)
        layer_map_b = _layer_map(artifact_b)
        validated = validate_prompt_artifact_pair(
            artifact_a,
            artifact_b,
            alignment_mode=alignment_mode,
            side_a_label="evaluation",
            side_b_label="reference",
            readout_mode="raw",
        )
        shared_layers = set(validated.layer_indices)
        common_layers = shared_layers if common_layers is None else (common_layers & shared_layers)
        all_positions.update(int(v) for v in valid_positions.tolist())
        examples.append(
            _PromptAlignedExample(
                prompt_id=artifact_a.prompt_id,
                prompt_text=artifact_a.prompt_text,
                positions=valid_positions,
                layer_records_a=layer_map_a,
                layer_records_b=layer_map_b,
            )
        )
    if not examples:
        raise ValueError("Prompt calibration found no aligned prompt examples")
    resolved_layers = sorted(common_layers or [])
    if not resolved_layers:
        raise ValueError("Prompt calibration found no shared layers between the two sides")
    return examples, resolved_layers, sorted(all_positions)


def _get_prompt_logits(record: PromptLayerRecord, *, readout_mode: str) -> torch.Tensor:
    logits = record.get_logits(readout_mode)  # type: ignore[arg-type]
    if logits is None:
        raise ValueError(f"Missing logits_{readout_mode} for prompt layer {record.layer_name}")
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError(f"Expected prompt logits shape [1, seq, vocab], got {tuple(logits.shape)}")
    return logits[0].to(dtype=torch.float32, device="cpu")


def _resolve_reference_logits(
    *,
    layer_index: int,
    layer_records_b: dict[int, PromptLayerRecord],
    reference_source: PromptReferenceSource,
    readout_mode_reference: str,
) -> torch.Tensor:
    if reference_source == "matching_layer":
        return _get_prompt_logits(layer_records_b[layer_index], readout_mode=readout_mode_reference)
    if reference_source == "final_layer":
        final_layer_index = max(layer_records_b)
        return _get_prompt_logits(layer_records_b[final_layer_index], readout_mode=readout_mode_reference)
    raise ValueError(f"Unsupported prompt reference_source: {reference_source!r}")


def _bootstrap_confidence_interval(
    values: list[float],
    *,
    num_bootstrap: int,
    alpha: float,
    seed: int | None,
) -> tuple[float | None, float | None]:
    if not values or num_bootstrap <= 0:
        return None, None
    base = torch.tensor(values, dtype=torch.float32)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    draws: list[float] = []
    for _ in range(num_bootstrap):
        idx = torch.randint(0, base.numel(), (base.numel(),), generator=gen)
        draws.append(float(base[idx].mean().item()))
    quantiles = torch.tensor([alpha / 2.0, 1.0 - alpha / 2.0], dtype=torch.float32)
    ci = torch.quantile(torch.tensor(draws, dtype=torch.float32), quantiles)
    return float(ci[0].item()), float(ci[1].item())


def run_prompt_reference_calibration(
    artifact_a: str | Path | PromptDecodeArtifact | dict[str, Any],
    artifact_b: str | Path | PromptDecodeArtifact | dict[str, Any],
    *,
    side_a_label: str = "artifact_a",
    side_b_label: str = "artifact_b",
    alignment_mode: PromptCalibrationAlignmentMode = "same_token_ids",
    readout_mode_eval: str = "model_norm",
    readout_mode_reference: str = "model_norm",
    reference_kind: str = "reference_relative",
    reference_source: PromptReferenceSource = "final_layer",
    top_k_values: list[int] | None = None,
    num_bins: int = 15,
    binning: str = "equal_mass",
    min_samples_per_cell: int = 2,
    num_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int | None = None,
) -> CalibrationRunArtifact:
    top_k_values = sorted({int(v) for v in (top_k_values or [5, 10]) if int(v) > 0})
    examples, layer_indices, position_indices = _build_prompt_examples(
        artifact_a,
        artifact_b,
        alignment_mode=alignment_mode,
    )
    pos_to_col = {pos: idx for idx, pos in enumerate(position_indices)}
    accumulators: dict[tuple[int, int], _PromptMetricAccumulator] = {}

    for example in examples:
        for layer_index in layer_indices:
            eval_logits = _get_prompt_logits(example.layer_records_a[layer_index], readout_mode=readout_mode_eval)
            ref_logits = _resolve_reference_logits(
                layer_index=layer_index,
                layer_records_b=example.layer_records_b,
                reference_source=reference_source,
                readout_mode_reference=readout_mode_reference,
            )
            probs_eval = torch.softmax(eval_logits, dim=-1)
            labels = torch.argmax(ref_logits, dim=-1)
            for position in example.positions.tolist():
                key = (layer_index, int(position))
                accum = accumulators.setdefault(key, _PromptMetricAccumulator())
                accum.add(
                    probs=probs_eval[int(position)],
                    label=int(labels[int(position)].item()),
                    top_k_values=top_k_values,
                )

    def _empty_matrix() -> torch.Tensor:
        return torch.full((len(layer_indices), len(position_indices)), float("nan"), dtype=torch.float32)

    sample_count_matrix = torch.zeros((len(layer_indices), len(position_indices)), dtype=torch.float32)
    accuracy_matrix = _empty_matrix()
    confidence_matrix = _empty_matrix()
    top1_ece_matrix = _empty_matrix()
    brier_matrix = _empty_matrix()
    nll_matrix = _empty_matrix()
    topk_matrices: dict[int, torch.Tensor] = {k: _empty_matrix() for k in top_k_values}

    for row_idx, layer_index in enumerate(layer_indices):
        for position, col_idx in pos_to_col.items():
            accum = accumulators.get((layer_index, position))
            if accum is None:
                continue
            sample_count_matrix[row_idx, col_idx] = float(len(accum.top1_conf))
            top1_stats = compute_expected_calibration_error(
                torch.tensor(accum.top1_conf, dtype=torch.float32),
                torch.tensor(accum.top1_hit, dtype=torch.float32),
                num_bins=num_bins,
                binning=binning,
                min_samples=min_samples_per_cell,
            )
            top1_ece_matrix[row_idx, col_idx] = top1_stats.ece
            accuracy_matrix[row_idx, col_idx] = top1_stats.accuracy_mean
            confidence_matrix[row_idx, col_idx] = top1_stats.confidence_mean
            if accum.brier:
                brier_matrix[row_idx, col_idx] = float(torch.tensor(accum.brier, dtype=torch.float32).mean().item())
            if accum.nll:
                nll_matrix[row_idx, col_idx] = float(torch.tensor(accum.nll, dtype=torch.float32).mean().item())
            for k in top_k_values:
                stats = compute_expected_calibration_error(
                    torch.tensor(accum.topk_conf.get(k, []), dtype=torch.float32),
                    torch.tensor(accum.topk_hit.get(k, []), dtype=torch.float32),
                    num_bins=num_bins,
                    binning=binning,
                    min_samples=min_samples_per_cell,
                )
                topk_matrices[k][row_idx, col_idx] = stats.ece

    matrix_results = [
        CalibrationMatrixResult(
            metric_name="top1_ece",
            axis_kind="layer_position",
            value_matrix=top1_ece_matrix,
            sample_count_matrix=sample_count_matrix,
            confidence_mean_matrix=confidence_matrix,
            accuracy_mean_matrix=accuracy_matrix,
            metadata={"num_bins": num_bins, "binning": binning},
        ),
        CalibrationMatrixResult(
            metric_name="brier",
            axis_kind="layer_position",
            value_matrix=brier_matrix,
            sample_count_matrix=sample_count_matrix,
            metadata={},
        ),
        CalibrationMatrixResult(
            metric_name="nll",
            axis_kind="layer_position",
            value_matrix=nll_matrix,
            sample_count_matrix=sample_count_matrix,
            metadata={},
        ),
    ]
    for k in top_k_values:
        matrix_results.append(
            CalibrationMatrixResult(
                metric_name=f"top{k}_mass_ece",
                axis_kind="layer_position",
                value_matrix=topk_matrices[k],
                sample_count_matrix=sample_count_matrix,
                metadata={"top_k": k, "num_bins": num_bins, "binning": binning},
            )
        )

    def _build_summary(axis_kind: Literal["layer", "position"], metric_name: str) -> CalibrationSummaryResult:
        axis_indices = layer_indices if axis_kind == "layer" else position_indices
        values: list[float] = []
        ci_low: list[float] = []
        ci_high: list[float] = []
        sample_counts: list[float] = []
        conf_means: list[float] = []
        acc_means: list[float] = []
        topk_match = None
        if metric_name.startswith("top") and metric_name.endswith("_mass_ece"):
            topk_match = int(metric_name[len("top") : metric_name.index("_mass_ece")])
        for axis_index in axis_indices:
            top1_conf: list[float] = []
            top1_hit: list[float] = []
            topk_conf: list[float] = []
            topk_hit: list[float] = []
            scalars: list[float] = []
            for (layer_idx, pos_idx), accum in accumulators.items():
                if (axis_kind == "layer" and layer_idx != axis_index) or (axis_kind == "position" and pos_idx != axis_index):
                    continue
                if metric_name == "top1_ece":
                    top1_conf.extend(accum.top1_conf)
                    top1_hit.extend(accum.top1_hit)
                elif topk_match is not None:
                    topk_conf.extend(accum.topk_conf.get(topk_match, []))
                    topk_hit.extend(accum.topk_hit.get(topk_match, []))
                elif metric_name == "brier":
                    scalars.extend(accum.brier)
                elif metric_name == "nll":
                    scalars.extend(accum.nll)
            if metric_name == "top1_ece":
                stats = compute_expected_calibration_error(
                    torch.tensor(top1_conf, dtype=torch.float32),
                    torch.tensor(top1_hit, dtype=torch.float32),
                    num_bins=num_bins,
                    binning=binning,
                    min_samples=min_samples_per_cell,
                )
                values.append(stats.ece)
                sample_counts.append(float(stats.sample_count))
                conf_means.append(stats.confidence_mean)
                acc_means.append(stats.accuracy_mean)
                lo, hi = _bootstrap_confidence_interval(
                    [abs(c - a) for c, a in zip(top1_conf, top1_hit)],
                    num_bootstrap=num_bootstrap,
                    alpha=alpha,
                    seed=seed,
                )
            elif topk_match is not None:
                stats = compute_expected_calibration_error(
                    torch.tensor(topk_conf, dtype=torch.float32),
                    torch.tensor(topk_hit, dtype=torch.float32),
                    num_bins=num_bins,
                    binning=binning,
                    min_samples=min_samples_per_cell,
                )
                values.append(stats.ece)
                sample_counts.append(float(stats.sample_count))
                conf_means.append(stats.confidence_mean)
                acc_means.append(stats.accuracy_mean)
                lo, hi = _bootstrap_confidence_interval(
                    [abs(c - a) for c, a in zip(topk_conf, topk_hit)],
                    num_bootstrap=num_bootstrap,
                    alpha=alpha,
                    seed=seed,
                )
            else:
                tensor = torch.tensor(scalars, dtype=torch.float32)
                values.append(float(tensor.mean().item()) if tensor.numel() else float("nan"))
                sample_counts.append(float(tensor.numel()))
                conf_means.append(float("nan"))
                acc_means.append(float("nan"))
                lo, hi = _bootstrap_confidence_interval(
                    scalars,
                    num_bootstrap=num_bootstrap,
                    alpha=alpha,
                    seed=seed,
                )
            ci_low.append(float("nan") if lo is None else lo)
            ci_high.append(float("nan") if hi is None else hi)
        return CalibrationSummaryResult(
            metric_name=metric_name,
            axis_kind=axis_kind,
            axis_indices=list(axis_indices),
            values=torch.tensor(values, dtype=torch.float32),
            ci_low=torch.tensor(ci_low, dtype=torch.float32),
            ci_high=torch.tensor(ci_high, dtype=torch.float32),
            sample_counts=torch.tensor(sample_counts, dtype=torch.float32),
            confidence_means=torch.tensor(conf_means, dtype=torch.float32),
            accuracy_means=torch.tensor(acc_means, dtype=torch.float32),
            metadata={"num_bins": num_bins, "binning": binning, "num_bootstrap": num_bootstrap},
        )

    summary_results = [
        _build_summary("layer", "top1_ece"),
        _build_summary("position", "top1_ece"),
        _build_summary("layer", "brier"),
        _build_summary("position", "brier"),
        _build_summary("layer", "nll"),
        _build_summary("position", "nll"),
    ]
    for k in top_k_values:
        summary_results.extend(
            [
                _build_summary("layer", f"top{k}_mass_ece"),
                _build_summary("position", f"top{k}_mass_ece"),
            ]
        )

    backend_a = _load_prompt_source(artifact_a)[0].backend_metadata
    backend_b = _load_prompt_source(artifact_b)[0].backend_metadata
    return CalibrationRunArtifact(
        side_a_label=side_a_label,
        side_b_label=side_b_label,
        artifact_family="prompt",
        reference_kind=reference_kind,
        reference_source=reference_source,
        readout_mode_eval=readout_mode_eval,
        readout_mode_reference=readout_mode_reference,
        alignment_mode=alignment_mode,
        layer_indices=layer_indices,
        position_indices=position_indices,
        backend_metadata_a=backend_a,
        backend_metadata_b=backend_b,
        matrix_results=matrix_results,
        summary_results=summary_results,
        metadata={
            "num_examples": len(examples),
            "top_k_values": top_k_values,
            "num_bins": num_bins,
            "binning": binning,
        },
    )


__all__ = ["run_prompt_reference_calibration"]

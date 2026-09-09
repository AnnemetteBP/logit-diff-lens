from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from pathlib import Path
import re
from typing import Iterable

import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots

from ..calibration.prompt import run_prompt_reference_calibration
from ..diffing.distribution_metrics import js_divergence
from ..diffing.io import load_prompt_decode_artifact_bundle
from ..pair_validation import validate_prompt_artifact_pair
from ..schemas import PromptDecodeArtifact
from ..similarity.prompt import run_prompt_similarity
from .plotly_export import save_plotly_figure


MetricMode = str
ReadoutMode = str


@dataclass(frozen=True)
class FamilyPair:
    family: str
    comparison_path: str
    base_path: str


@dataclass
class FamilySummary:
    family: str
    mean: torch.Tensor
    ci_low: torch.Tensor
    ci_high: torch.Tensor
    per_prompt_curves: torch.Tensor
    layer_labels: list[str]
    prompt_keys: list[str]


def _artifact_key(artifact: PromptDecodeArtifact) -> str:
    if artifact.prompt_id is not None:
        return f"id::{artifact.prompt_id}"
    return f"text::{artifact.prompt_text}"


def _load_bundle(path: str | Path) -> dict[str, PromptDecodeArtifact]:
    path = str(Path(path))
    payload = load_prompt_decode_artifact_bundle(path)
    artifacts = payload["artifacts"]
    if not artifacts:
        raise ValueError(f"Prompt bundle {path} contains no artifacts")
    return {_artifact_key(artifact): artifact for artifact in artifacts}


def _layer_labels_from_artifact(artifact: PromptDecodeArtifact) -> list[str]:
    positive = [record.layer_index for record in artifact.layer_records if record.layer_index >= 0 and record.layer_name != "output"]
    total_layers = (max(positive) + 1) if positive else max(1, len(artifact.layer_records))
    labels: list[str] = []
    for record in artifact.layer_records:
        if record.layer_index < 0:
            labels.append("Embedding")
        elif record.layer_name == "output":
            labels.append("Output")
        else:
            labels.append(f"L{record.layer_index + 1}/{total_layers}")
    return labels


def _short_pair_label(path: str | Path) -> str:
    stem = Path(path).stem.replace("_prompt_bundle", "")
    size_match = re.search(r"pythia_(\d+[a-z]+)", stem)
    size_label = size_match.group(1) if size_match is not None else None
    step_match = re.search(r"step(\d+)", stem)
    if step_match is not None:
        step_value = int(step_match.group(1))
        if step_value % 1000 == 0:
            ckpt = f"{step_value // 1000}k"
        else:
            ckpt = str(step_value)
        return f"{size_label}-{ckpt}" if size_label is not None else ckpt
    seed_match = re.search(r"seed(\d+)", stem)
    if seed_match is not None:
        seed_label = f"r{seed_match.group(1)}"
        return f"{size_label}-{seed_label}" if size_label is not None else seed_label
    return stem


def _build_pair_style_map(labels: list[str]) -> dict[str, dict[str, object]]:
    palette = [
        "#111111",
        "#1f77b4",
        "#17becf",
        "#2ca02c",
        "#d62728",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    linestyles = ["-", "--", "--", "--", ":", ":", ":", "-.", "-.", "-."]
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
    style_map: dict[str, dict[str, object]] = {}
    for idx, label in enumerate(labels):
        style_map[label] = dict(
            color=palette[idx % len(palette)],
            linestyle=linestyles[idx % len(linestyles)],
            marker=markers[idx % len(markers)],
        )
    return style_map


def _apply_adaptive_header_layout(
    fig,
    *,
    legend,
    n_panel_rows: int,
    n_legend_items: int,
    legend_ncol: int,
    left: float = 0.06,
    right: float = 0.98,
    bottom: float = 0.07,
    min_top: float = 0.74,
    wspace: float = 0.18,
    hspace: float = 0.34,
    gap_below_legend: float = 0.022,
) -> None:
    legend_rows = max(1, math.ceil(n_legend_items / max(1, legend_ncol)))
    is_two_by_two_single_row = n_panel_rows == 2 and legend_rows == 1
    if legend_rows > 1:
        width, height = fig.get_size_inches()
        fig.set_size_inches(width, height + 0.46 * (legend_rows - 1), forward=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    if is_two_by_two_single_row:
        row_gap = max(0.046, gap_below_legend + 0.022)
    else:
        row_gap = gap_below_legend + 0.028 + 0.022 * max(0, legend_rows - 1) + 0.018 * max(0, n_panel_rows - 1)
    top = max(min_top, float(legend_bbox.y0) - row_gap)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)


def _finalize_header_layout(
    fig,
    *,
    legend,
    labels: list[str],
    n_panel_rows: int,
    n_panel_cols: int,
    max_columns: int,
    left: float = 0.06,
    right: float = 0.98,
    bottom: float = 0.07,
    min_top: float = 0.80,
    wspace: float = 0.18,
    hspace: float = 0.34,
    gap_below_legend: float = 0.022,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title = getattr(fig, "_suptitle", None)
    title_bottom = 0.975
    if title is not None:
        title_bbox = title.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        title_bottom = float(title_bbox.y0)
    legend_rows = max(1, math.ceil(len(labels) / max(1, min(max_columns, max(1, getattr(legend, "_ncols", max_columns))))))
    is_two_by_two_single_row = n_panel_rows == 2 and n_panel_cols == 2 and legend_rows == 1
    width, height = fig.get_size_inches()
    label_pressure = 0.0
    if labels:
        longest = max(len(label) for label in labels)
        total = sum(len(label) for label in labels)
        label_pressure = 0.05 * max(0, legend_rows - 1) + 0.0015 * max(0, total - 60) + 0.004 * max(0, longest - 24)
    extra_height = 0.34 * max(0, legend_rows - 1) + 0.10 * max(0, n_panel_rows - 1) + min(0.22, label_pressure)
    if is_two_by_two_single_row:
        extra_height = max(extra_height, 0.24)
    extra_width = 1.20 * max(0, n_panel_cols - 2) + min(4.80, 0.09 * max(0, (max((len(label) for label in labels), default=0) - 16)))
    if extra_height > 0 or extra_width > 0:
        fig.set_size_inches(width + extra_width, height + extra_height, forward=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if title is not None:
            title_bbox = title.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            title_bottom = float(title_bbox.y0)
    if is_two_by_two_single_row:
        legend_y = max(0.905, min(0.958, title_bottom - 0.010))
    else:
        legend_y = max(0.82, min(0.942, title_bottom - 0.020))
    legend.set_bbox_to_anchor((0.5, legend_y), transform=fig.transFigure)
    _apply_adaptive_header_layout(
        fig,
        legend=legend,
        n_panel_rows=n_panel_rows,
        n_legend_items=len(labels),
        legend_ncol=min(max_columns, max(1, getattr(legend, "_ncols", max_columns))),
        left=left,
        right=right,
        bottom=bottom,
        min_top=min_top,
        wspace=wspace,
        hspace=hspace,
        gap_below_legend=gap_below_legend,
    )


def _legend_columns_for_labels(labels: list[str], *, max_columns: int = 3) -> int:
    if not labels:
        return 1
    count = len(labels)
    longest = max(len(label) for label in labels)
    total = sum(len(label) for label in labels)
    def _odd_safe(columns: int) -> int:
        if count % 2 == 1 and columns % 2 == 0 and columns > 1:
            return columns + 1 if columns + 1 <= min(count, max_columns) else columns - 1
        return columns
    if longest >= 56 or total >= 150:
        return 1
    if longest >= 30 or total >= 90:
        return _odd_safe(min(2, max_columns))
    return _odd_safe(min(count, max_columns))


def _legend_ncol_max_two_rows(
    labels: list[str],
    *,
    max_columns: int,
    prefer_one_row: bool = True,
) -> int:
    if not labels:
        return 1
    count = len(labels)
    if count <= 4:
        return count
    longest = max(len(label) for label in labels)
    total = sum(len(label) for label in labels)
    allowed_max = max(1, min(max_columns, count))
    if prefer_one_row and count <= min(8, allowed_max) and longest <= 24 and total <= 120:
        return count
    balanced_two_row_cols = math.ceil(count / 2)
    if longest >= 52 or total >= 220:
        return max(1, min(3, allowed_max))
    if longest >= 38 or total >= 160:
        return min(allowed_max, max(1, balanced_two_row_cols))
    return min(allowed_max, max(1, balanced_two_row_cols))


def _legend_row_grouped_items(
    handles: list[object],
    labels: list[str],
    *,
    ncol: int,
) -> tuple[list[object], list[str]]:
    if ncol <= 1 or len(labels) <= 1:
        return handles, labels
    n_items = len(labels)
    ncol = max(1, min(ncol, n_items))
    nrows = math.ceil(n_items / ncol)
    grid: list[list[tuple[object, str] | None]] = [[None for _ in range(ncol)] for _ in range(nrows)]
    idx = 0
    for row in range(nrows):
        for col in range(ncol):
            if idx >= n_items:
                break
            grid[row][col] = (handles[idx], labels[idx])
            idx += 1
    reordered_handles: list[object] = []
    reordered_labels: list[str] = []
    for col in range(ncol):
        for row in range(nrows):
            item = grid[row][col]
            if item is None:
                continue
            handle, label = item
            reordered_handles.append(handle)
            reordered_labels.append(label)
    return reordered_handles, reordered_labels


def _legend_fontsize_for_labels(
    labels: list[str],
    *,
    base: int = 14,
    min_size: int = 26,
    max_size: int | None = None,
) -> int:
    if not labels:
        return base
    longest = max(len(label) for label in labels)
    total = sum(len(label) for label in labels)
    size = base
    if longest >= 86 or total >= 220:
        size -= 3
    elif longest >= 66 or total >= 180:
        size -= 2
    elif longest >= 46 or total >= 132:
        size -= 1
    if len(labels) <= 3 and longest <= 40 and total <= 140:
        size = max(size, 34)
    elif len(labels) <= 5 and longest <= 36 and total <= 170:
        size = max(size, 32)
    elif len(labels) <= 7 and longest <= 28 and total <= 180:
        size = max(size, 30)
    if max_size is not None:
        size = min(size, max_size)
    return max(min_size, size)


def _responsive_font_size(
    *,
    rows: int,
    cols: int,
    base: int,
    min_size: int,
    max_size: int | None = None,
) -> int:
    size = base + 2
    if rows >= 5:
        size -= 1
    if rows >= 6:
        size -= 1
    if cols >= 4:
        size -= 1
    if rows <= 2 and cols <= 2:
        size += 3
    elif rows <= 4 and cols <= 2:
        size += 2
    elif rows <= 4 and cols <= 3:
        size += 1
    if max_size is not None:
        size = min(size, max_size)
    return max(min_size, size)


def _panel_font_pack(*, rows: int, cols: int) -> dict[str, int]:
    return {
        "title": _responsive_font_size(rows=rows, cols=cols, base=28, min_size=24),
        "axis": _responsive_font_size(rows=rows, cols=cols, base=34, min_size=30),
        "ticks": _responsive_font_size(rows=rows, cols=cols, base=31, min_size=28),
        "legend": _responsive_font_size(rows=rows, cols=cols, base=33, min_size=30),
    }


def _pairwise_header_font_pack(*, rows: int, cols: int) -> dict[str, int]:
    panel = _panel_font_pack(rows=rows, cols=cols)
    return {
        "suptitle": max(panel["title"] + 2, 32),
        "title": max(panel["title"] + 2, 30),
        "axis": max(panel["axis"] + 2, 30),
        "ticks": max(panel["ticks"] + 1, 28),
        "legend": max(panel["legend"] + 2, 30),
    }


def _stacked_header_positions(
    *,
    title_y: float = 0.975,
    first_gap: float = 0.060,
    row_gap: float = 0.060,
) -> tuple[float, float]:
    first = title_y - first_gap
    second = first - row_gap
    return first, second


def _short_gap_family_title(label: str) -> str:
    shortened = label
    shortened = shortened.replace("Delta = [", "Delta[")
    shortened = shortened.replace("] - [", "] - [")
    shortened = shortened.replace("mean over random seeds", "rand-mean")
    shortened = shortened.replace("mean over random-seed pairs", "rand-pair-mean")
    shortened = shortened.replace("random-pair mean", "rand-pair-mean")
    shortened = shortened.replace("random-seed mean", "rand-mean")
    shortened = shortened.replace(" vs ", " vs ")
    shortened = shortened.replace(" - rand-pair mean", " - rand-pair")
    shortened = shortened.replace(" - rand-pair-mean", " - rand-pair")
    shortened = shortened.replace(" - rand-mean", " - rand-mean")
    shortened = shortened.replace("trained-baseline", "train-base")
    shortened = shortened.replace("trained-rand", "train-rand")
    shortened = shortened.replace("trained vs ", "train vs ")
    shortened = shortened.replace("random-seed", "rand")
    shortened = shortened.replace("random-pair", "rand-pair")
    shortened = shortened.replace("random", "rand")
    shortened = shortened.replace("baseline", "base")
    shortened = shortened.replace("mean ", "")
    shortened = shortened.replace("(n=", "n=")
    shortened = shortened.replace(")", "")
    return shortened


def _compact_metric_label(label: str) -> str:
    compact = label
    compact = compact.replace("Negative Log-Likelihood", "NLL")
    compact = compact.replace("negative log-likelihood", "NLL")
    compact = compact.replace("Non-calibrated ", "")
    compact = compact.replace("Calibrated ", "")
    return compact


def _overlay_axis_mode(metric_label: str, *, gap: bool) -> str:
    compact = _compact_metric_label(metric_label)
    if gap and compact == "Top-1 ECE":
        return "shared"
    return "twin"


def _build_pair_specs(
    *,
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
) -> list[tuple[str, str, str, str]]:
    pair_specs: list[tuple[str, str, str, str]] = [
        (
            "trained_vs_trained",
            f"{_short_pair_label(trained_artifact)} vs {_short_pair_label(baseline_artifact)}",
            trained_artifact,
            baseline_artifact,
        ),
    ]
    for random_path in random_artifacts:
        pair_specs.append(
            (
                "trained_vs_random",
                f"{_short_pair_label(trained_artifact)} vs {_short_pair_label(random_path)}",
                trained_artifact,
                random_path,
            )
        )
    for idx, left in enumerate(random_artifacts):
        for right in random_artifacts[idx + 1 :]:
            pair_specs.append(
                (
                    "random_vs_random",
                    f"{_short_pair_label(left)} vs {_short_pair_label(right)}",
                    left,
                    right,
                )
            )
    return pair_specs


def _mean_over_valid_positions(values: torch.Tensor, valid_mask: torch.Tensor) -> float:
    flattened = values.reshape(-1).to(dtype=torch.float32)
    mask = valid_mask.reshape(-1).to(dtype=torch.bool)
    if flattened.shape[0] != mask.shape[0]:
        raise ValueError(
            f"Per-position metric shape {tuple(values.shape)} does not match valid-mask shape {tuple(valid_mask.shape)}"
        )
    selected = flattened[mask]
    if selected.numel() == 0:
        return float("nan")
    return float(selected.mean().item())


def _non_calibrated_curve(
    comparison: PromptDecodeArtifact,
    base: PromptDecodeArtifact,
    *,
    readout_mode: ReadoutMode,
    top_k: int,
) -> torch.Tensor:
    validated = validate_prompt_artifact_pair(
        comparison,
        base,
        alignment_mode="same_token_ids",
        side_a_label="comparison",
        side_b_label="base",
        readout_mode=readout_mode,
    )
    valid_mask = validated.valid_mask[0]
    layer_values: list[float] = []
    for layer_index in validated.layer_indices:
        comparison_record = next(record for record in comparison.layer_records if record.layer_index == layer_index)
        base_record = next(record for record in base.layer_records if record.layer_index == layer_index)
        comparison_logits = comparison_record.get_logits(readout_mode)
        base_logits = base_record.get_logits(readout_mode)
        if comparison_logits is None or base_logits is None:
            raise ValueError(f"Missing logits for readout_mode={readout_mode!r} at layer_index={layer_index}")
        metric = js_divergence(
            torch.softmax(comparison_logits.to(dtype=torch.float32), dim=-1),
            torch.softmax(base_logits.to(dtype=torch.float32), dim=-1),
        )
        layer_values.append(_mean_over_valid_positions(metric, valid_mask))
    return torch.tensor(layer_values, dtype=torch.float32)


def _calibrated_curve(
    comparison: PromptDecodeArtifact,
    base: PromptDecodeArtifact,
    *,
    readout_mode: ReadoutMode,
    metric_name: str = "top1_ece",
    num_bins: int,
    min_samples_per_cell: int,
) -> torch.Tensor:
    artifact = run_prompt_reference_calibration(
        comparison,
        base,
        side_a_label="comparison",
        side_b_label="base",
        alignment_mode="same_token_ids",
        readout_mode_eval=readout_mode,
        readout_mode_reference=readout_mode,
        reference_kind="reference_relative",
        reference_source="final_layer",
        num_bins=num_bins,
        min_samples_per_cell=min_samples_per_cell,
        num_bootstrap=0,
    )
    for summary in artifact.summary_results:
        if summary.axis_kind == "layer" and summary.metric_name == metric_name:
            return summary.values.to(dtype=torch.float32)
    raise ValueError(f"Calibration artifact did not contain a layerwise {metric_name} summary")


def _similarity_curve(
    comparison: PromptDecodeArtifact,
    base: PromptDecodeArtifact,
    *,
    readout_mode: ReadoutMode,
    similarity_metric_name: str,
    top_k: int,
    num_permutations: int,
    alpha: float,
    seed: int | None,
    calibrated: bool,
) -> torch.Tensor:
    validated = validate_prompt_artifact_pair(
        comparison,
        base,
        alignment_mode="same_token_ids",
        side_a_label="comparison",
        side_b_label="base",
        readout_mode=readout_mode,
    )
    layer_indices = list(validated.layer_indices)
    artifact = run_prompt_similarity(
        comparison,
        base,
        side_a_label="comparison",
        side_b_label="base",
        representation="logits",
        metric=similarity_metric_name,  # type: ignore[arg-type]
        alignment_mode="same_token_ids",
        sample_mode="flatten_all_valid_positions",
        layer_mode="fixed_pairs",
        readout_mode=readout_mode,
        layer_indices_a=layer_indices,
        layer_indices_b=layer_indices,
        num_permutations=num_permutations,
        alpha=alpha,
        top_k=top_k,
        permutation_unit="auto",
        multiple_testing_method="fdr_bh",
        seed=seed,
    )
    if not artifact.scalar_results:
        raise ValueError("Similarity artifact did not contain fixed-pair scalar results")
    values_by_layer: dict[int, float] = {}
    for item in artifact.scalar_results:
        layer_index = int(item.metadata["layer_index_a"])
        values_by_layer[layer_index] = (
            float(item.calibrated_similarity) if calibrated else float(item.raw_similarity)
        )
    missing = [idx for idx in layer_indices if idx not in values_by_layer]
    if missing:
        raise ValueError(f"Similarity artifact missing fixed-pair results for layers: {missing}")
    return torch.tensor([values_by_layer[idx] for idx in layer_indices], dtype=torch.float32)


def _hidden_scalar_metric(
    hidden_a: torch.Tensor,
    hidden_b: torch.Tensor,
    *,
    metric_name: str,
) -> float:
    a = hidden_a.to(dtype=torch.float32, device="cpu")
    b = hidden_b.to(dtype=torch.float32, device="cpu")
    if a.shape != b.shape:
        raise ValueError(f"Hidden tensors must match for {metric_name}: {tuple(a.shape)} != {tuple(b.shape)}")
    if metric_name == "hidden_cosine_similarity":
        return float(F.cosine_similarity(a, b, dim=-1, eps=1e-12).mean().item())
    if metric_name == "hidden_l2_distance":
        return float(torch.linalg.vector_norm(a - b, ord=2, dim=-1).mean().item())
    if metric_name == "hidden_normalized_l2_distance":
        denom = torch.linalg.vector_norm(a, ord=2, dim=-1).clamp_min(1e-12)
        return float((torch.linalg.vector_norm(a - b, ord=2, dim=-1) / denom).mean().item())
    raise ValueError(f"Unsupported hidden metric: {metric_name!r}")


def _calibrate_hidden_scalar(
    hidden_a: torch.Tensor,
    hidden_b: torch.Tensor,
    *,
    metric_name: str,
    num_permutations: int,
    alpha: float,
    seed: int | None,
) -> float:
    a = hidden_a.to(dtype=torch.float32, device="cpu")
    b = hidden_b.to(dtype=torch.float32, device="cpu")
    observed = _hidden_scalar_metric(a, b, metric_name=metric_name)
    if a.shape[0] < 2 or num_permutations <= 0:
        return 0.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0 if seed is None else int(seed))
    null_scores = []
    for _ in range(num_permutations):
        perm = torch.randperm(b.shape[0], generator=generator)
        null_scores.append(_hidden_scalar_metric(a, b[perm], metric_name=metric_name))
    null_tensor = torch.tensor(null_scores, dtype=torch.float32)
    threshold = float(torch.quantile(null_tensor, 1.0 - alpha).item())
    if metric_name == "hidden_cosine_similarity":
        if observed <= threshold:
            return 0.0
        denom = max(1e-12, 1.0 - threshold)
        return float(max(0.0, min(1.0, (observed - threshold) / denom)))
    if observed <= threshold:
        return 0.0
    return float(observed - threshold)


def _hidden_metric_curve(
    comparison: PromptDecodeArtifact,
    base: PromptDecodeArtifact,
    *,
    metric_name: str,
    calibrated: bool,
    num_permutations: int,
    alpha: float,
    seed: int | None,
) -> torch.Tensor:
    validated = validate_prompt_artifact_pair(
        comparison,
        base,
        alignment_mode="same_token_ids",
        side_a_label="comparison",
        side_b_label="base",
    )
    valid_mask = validated.valid_mask[0]
    layer_values: list[float] = []
    seed_base = 0 if seed is None else int(seed)
    for layer_offset, layer_index in enumerate(validated.layer_indices):
        comparison_record = next(record for record in comparison.layer_records if record.layer_index == layer_index)
        base_record = next(record for record in base.layer_records if record.layer_index == layer_index)
        hidden_a = comparison_record.get_hidden("raw")[0][valid_mask]
        hidden_b = base_record.get_hidden("raw")[0][valid_mask]
        if calibrated:
            layer_values.append(
                _calibrate_hidden_scalar(
                    hidden_a,
                    hidden_b,
                    metric_name=metric_name,
                    num_permutations=num_permutations,
                    alpha=alpha,
                    seed=seed_base + layer_offset,
                )
            )
        else:
            layer_values.append(_hidden_scalar_metric(hidden_a, hidden_b, metric_name=metric_name))
    return torch.tensor(layer_values, dtype=torch.float32)


def _build_promptwise_family_curves(
    pairs: Iterable[FamilyPair],
    *,
    readout_mode: ReadoutMode,
    metric_mode: MetricMode,
    top_k: int,
    num_bins: int,
    min_samples_per_cell: int,
    calibrated_metric_name: str = "top1_ece",
) -> tuple[torch.Tensor, list[str], list[str]]:
    aggregated: dict[str, list[torch.Tensor]] = {}
    layer_labels: list[str] | None = None
    for pair in pairs:
        comparison_bundle = _load_bundle(pair.comparison_path)
        base_bundle = _load_bundle(pair.base_path)
        shared_keys = sorted(set(comparison_bundle) & set(base_bundle))
        if not shared_keys:
            raise ValueError(
                f"No shared prompts between {pair.comparison_path} and {pair.base_path}"
            )
        for key in shared_keys:
            comparison = comparison_bundle[key]
            base = base_bundle[key]
            if layer_labels is None:
                layer_labels = _layer_labels_from_artifact(comparison)
            if metric_mode == "non_calibrated":
                curve = _non_calibrated_curve(
                    comparison,
                    base,
                    readout_mode=readout_mode,
                    top_k=top_k,
                )
            elif metric_mode == "calibrated":
                curve = _calibrated_curve(
                    comparison,
                    base,
                    readout_mode=readout_mode,
                    metric_name=calibrated_metric_name,
                    num_bins=num_bins,
                    min_samples_per_cell=min_samples_per_cell,
                )
            else:
                raise ValueError(f"Unsupported metric_mode: {metric_mode!r}")
            aggregated.setdefault(key, []).append(curve)
        del comparison_bundle
        del base_bundle
        gc.collect()

    if not aggregated:
        raise ValueError("No prompt curves were collected for the requested family")

    prompt_keys = sorted(aggregated)
    prompt_curves = torch.stack(
        [torch.stack(aggregated[key], dim=0).mean(dim=0) for key in prompt_keys],
        dim=0,
    )
    return prompt_curves, (layer_labels or []), prompt_keys


def _bootstrap_mean_ci(
    values: torch.Tensor,
    *,
    num_bootstrap: int,
    alpha: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = values.to(dtype=torch.float32)
    mean = values.mean(dim=0)
    if values.shape[0] < 2 or num_bootstrap <= 0:
        return mean, mean.clone(), mean.clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n = values.shape[0]
    idx = torch.randint(0, n, (num_bootstrap, n), generator=generator)
    draw_tensor = values[idx].mean(dim=1)
    quantiles = torch.tensor([alpha / 2.0, 1.0 - alpha / 2.0], dtype=torch.float32)
    ci = torch.quantile(draw_tensor, quantiles, dim=0)
    return mean, ci[0], ci[1]


def build_prompt_null_sanity_summaries(
    *,
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
    top_k: int = 10,
    num_bins: int = 15,
    min_samples_per_cell: int = 2,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    calibrated_metric_name: str = "top1_ece",
) -> dict[tuple[str, str], FamilySummary]:
    if len(random_artifacts) < 2:
        raise ValueError("Provide at least two random_artifacts to build a random-vs-random null family.")

    trained_random_pairs = [
        FamilyPair(family="trained_vs_random", comparison_path=trained_artifact, base_path=path)
        for path in random_artifacts
    ]
    random_random_pairs: list[FamilyPair] = []
    for idx, left in enumerate(random_artifacts):
        for right in random_artifacts[idx + 1 :]:
            random_random_pairs.append(
                FamilyPair(family="random_vs_random", comparison_path=left, base_path=right)
            )

    pair_groups: dict[str, list[FamilyPair]] = {
        "trained_vs_trained": [
            FamilyPair(
                family="trained_vs_trained",
                comparison_path=trained_artifact,
                base_path=baseline_artifact,
            )
        ],
        "trained_vs_random": trained_random_pairs,
        "random_vs_random": random_random_pairs,
    }

    results: dict[tuple[str, str], FamilySummary] = {}
    for readout_mode in ("raw", "model_norm"):
        for metric_mode in ("non_calibrated", "calibrated"):
            for family_name, pairs in pair_groups.items():
                prompt_curves, layer_labels, prompt_keys = _build_promptwise_family_curves(
                    pairs,
                    readout_mode=readout_mode,
                    metric_mode=metric_mode,
                    top_k=top_k,
                    num_bins=num_bins,
                    min_samples_per_cell=min_samples_per_cell,
                    calibrated_metric_name=calibrated_metric_name,
                )
                mean, ci_low, ci_high = _bootstrap_mean_ci(
                    prompt_curves,
                    num_bootstrap=num_bootstrap,
                    alpha=alpha,
                    seed=seed,
                )
                results[(readout_mode, metric_mode, family_name)] = FamilySummary(
                    family=family_name,
                    mean=mean,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    per_prompt_curves=prompt_curves,
                    layer_labels=layer_labels,
                    prompt_keys=prompt_keys,
                )
    return results


def _gap_summary(
    summary: FamilySummary,
    null_summary: FamilySummary,
    *,
    label: str,
    num_bootstrap: int,
    alpha: float,
    seed: int,
) -> FamilySummary:
    if summary.prompt_keys != null_summary.prompt_keys:
        raise ValueError("Gap summaries require identical prompt ordering between family and null summaries")
    gap_curves = summary.per_prompt_curves - null_summary.per_prompt_curves
    mean, ci_low, ci_high = _bootstrap_mean_ci(
        gap_curves,
        num_bootstrap=num_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    return FamilySummary(
        family=label,
        mean=mean,
        ci_low=ci_low,
        ci_high=ci_high,
        per_prompt_curves=gap_curves,
        layer_labels=list(summary.layer_labels),
        prompt_keys=list(summary.prompt_keys),
    )


def build_prompt_pairwise_jsd_curves(
    *,
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
    readout_mode: str,
    top_k: int = 10,
) -> tuple[list[dict[str, object]], list[str]]:
    pair_specs = _build_pair_specs(
        trained_artifact=trained_artifact,
        baseline_artifact=baseline_artifact,
        random_artifacts=random_artifacts,
    )

    curves: list[dict[str, object]] = []
    layer_labels: list[str] | None = None
    for family, label, comparison_path, base_path in pair_specs:
        prompt_curves, labels, prompt_keys = _build_promptwise_family_curves(
            [FamilyPair(family=family, comparison_path=comparison_path, base_path=base_path)],
            readout_mode=readout_mode,
            metric_mode="non_calibrated",
            top_k=top_k,
            num_bins=15,
            min_samples_per_cell=2,
        )
        if layer_labels is None:
            layer_labels = labels
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            prompt_curves,
            num_bootstrap=1000,
            alpha=0.05,
            seed=0,
        )
        curves.append(
            {
                "family": family,
                "label": label,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "prompt_keys": prompt_keys,
            }
        )
    return curves, (layer_labels or [])


def build_prompt_pairwise_metric_curves(
    *,
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
    readout_mode: str,
    metric_mode: str,
    top_k: int = 10,
    calibrated_metric_name: str = "top1_ece",
    num_bins: int = 15,
    min_samples_per_cell: int = 2,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[list[dict[str, object]], list[str]]:
    if metric_mode not in {"non_calibrated", "calibrated"}:
        raise ValueError(f"Unsupported metric_mode: {metric_mode!r}")

    pair_specs = _build_pair_specs(
        trained_artifact=trained_artifact,
        baseline_artifact=baseline_artifact,
        random_artifacts=random_artifacts,
    )

    curves: list[dict[str, object]] = []
    layer_labels: list[str] | None = None
    for offset, (family, label, comparison_path, base_path) in enumerate(pair_specs):
        prompt_curves, labels, prompt_keys = _build_promptwise_family_curves(
            [FamilyPair(family=family, comparison_path=comparison_path, base_path=base_path)],
            readout_mode=readout_mode,
            metric_mode=metric_mode,
            top_k=top_k,
            num_bins=num_bins,
            min_samples_per_cell=min_samples_per_cell,
            calibrated_metric_name=calibrated_metric_name,
        )
        if layer_labels is None:
            layer_labels = labels
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            prompt_curves,
            num_bootstrap=num_bootstrap,
            alpha=alpha,
            seed=seed + offset,
        )
        curves.append(
            {
                "family": family,
                "label": label,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "prompt_keys": prompt_keys,
            }
        )
    return curves, (layer_labels or [])


def build_prompt_pairwise_similarity_curves(
    *,
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
    readout_mode: str,
    similarity_metric_name: str,
    calibrated: bool,
    top_k: int = 10,
    num_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[list[dict[str, object]], list[str]]:
    pair_specs = _build_pair_specs(
        trained_artifact=trained_artifact,
        baseline_artifact=baseline_artifact,
        random_artifacts=random_artifacts,
    )

    curves: list[dict[str, object]] = []
    layer_labels: list[str] | None = None
    for offset, (family, label, comparison_path, base_path) in enumerate(pair_specs):
        comparison_bundle = _load_bundle(comparison_path)
        base_bundle = _load_bundle(base_path)
        shared_keys = sorted(set(comparison_bundle) & set(base_bundle))
        if not shared_keys:
            raise ValueError(f"No shared prompts between {comparison_path} and {base_path}")
        prompt_curves: list[torch.Tensor] = []
        for prompt_idx, key in enumerate(shared_keys):
            comparison = comparison_bundle[key]
            base = base_bundle[key]
            if layer_labels is None:
                layer_labels = _layer_labels_from_artifact(comparison)
            prompt_curves.append(
                _similarity_curve(
                    comparison,
                    base,
                    readout_mode=readout_mode,
                    similarity_metric_name=similarity_metric_name,
                    top_k=top_k,
                    num_permutations=num_permutations,
                    alpha=alpha,
                    seed=seed + offset * 1000 + prompt_idx,
                    calibrated=calibrated,
                )
            )
        prompt_tensor = torch.stack(prompt_curves, dim=0)
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            prompt_tensor,
            num_bootstrap=1000,
            alpha=alpha,
            seed=seed + offset,
        )
        curves.append(
            {
                "family": family,
                "label": label,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "prompt_keys": shared_keys,
            }
        )
    return curves, (layer_labels or [])


def build_prompt_pairwise_similarity_gap_curves(
    *,
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
    readout_mode: str,
    similarity_metric_name: str,
    calibrated: bool,
    top_k: int = 10,
    num_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[list[dict[str, object]], list[str]]:
    pair_specs = _build_pair_specs(
        trained_artifact=trained_artifact,
        baseline_artifact=baseline_artifact,
        random_artifacts=random_artifacts,
    )

    null_prompt_curves: list[torch.Tensor] = []
    layer_labels: list[str] | None = None
    null_prompt_keys: list[str] | None = None
    random_random_pairs = [
        (left, right)
        for idx, left in enumerate(random_artifacts)
        for right in random_artifacts[idx + 1 :]
    ]
    for pair_offset, (left, right) in enumerate(random_random_pairs):
        comparison_bundle = _load_bundle(left)
        base_bundle = _load_bundle(right)
        shared_keys = sorted(set(comparison_bundle) & set(base_bundle))
        if null_prompt_keys is None:
            null_prompt_keys = shared_keys
        elif shared_keys != null_prompt_keys:
            raise ValueError("Random-null similarity curves require identical prompt ordering")
        per_prompt: list[torch.Tensor] = []
        for prompt_offset, key in enumerate(shared_keys):
            comparison = comparison_bundle[key]
            base = base_bundle[key]
            if layer_labels is None:
                layer_labels = _layer_labels_from_artifact(comparison)
            per_prompt.append(
                _similarity_curve(
                    comparison,
                    base,
                    readout_mode=readout_mode,
                    similarity_metric_name=similarity_metric_name,
                    top_k=top_k,
                    num_permutations=num_permutations,
                    alpha=alpha,
                    seed=seed + pair_offset * 1000 + prompt_offset,
                    calibrated=calibrated,
                )
            )
        null_prompt_curves.append(torch.stack(per_prompt, dim=0))
    if not null_prompt_curves:
        raise ValueError("At least one random-vs-random pair is required for similarity gap curves")
    null_curve_tensor = torch.stack(null_prompt_curves, dim=0).mean(dim=0)

    curves: list[dict[str, object]] = []
    for offset, (family, label, comparison_path, base_path) in enumerate(pair_specs):
        comparison_bundle = _load_bundle(comparison_path)
        base_bundle = _load_bundle(base_path)
        shared_keys = sorted(set(comparison_bundle) & set(base_bundle))
        if shared_keys != null_prompt_keys:
            raise ValueError("Prompt key mismatch between pairwise similarity curves and random-null prompt curves")
        prompt_curves: list[torch.Tensor] = []
        for prompt_offset, key in enumerate(shared_keys):
            comparison = comparison_bundle[key]
            base = base_bundle[key]
            prompt_curves.append(
                _similarity_curve(
                    comparison,
                    base,
                    readout_mode=readout_mode,
                    similarity_metric_name=similarity_metric_name,
                    top_k=top_k,
                    num_permutations=num_permutations,
                    alpha=alpha,
                    seed=seed + offset * 1000 + prompt_offset,
                    calibrated=calibrated,
                )
            )
        gap_tensor = torch.stack(prompt_curves, dim=0) - null_curve_tensor
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            gap_tensor,
            num_bootstrap=1000,
            alpha=alpha,
            seed=seed + offset,
        )
        curves.append(
            {
                "family": family,
                "label": label,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "prompt_keys": shared_keys,
            }
        )
    return curves, (layer_labels or [])


def build_prompt_similarity_summaries(
    *,
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
    similarity_metric_name: str,
    top_k: int = 10,
    num_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[tuple[str, str, str], FamilySummary]:
    if len(random_artifacts) < 2:
        raise ValueError("Provide at least two random_artifacts to build a random-vs-random null family.")

    trained_random_pairs = [
        FamilyPair(family="trained_vs_random", comparison_path=trained_artifact, base_path=path)
        for path in random_artifacts
    ]
    random_random_pairs: list[FamilyPair] = []
    for idx, left in enumerate(random_artifacts):
        for right in random_artifacts[idx + 1 :]:
            random_random_pairs.append(
                FamilyPair(family="random_vs_random", comparison_path=left, base_path=right)
            )

    pair_groups: dict[str, list[FamilyPair]] = {
        "trained_vs_trained": [
            FamilyPair(
                family="trained_vs_trained",
                comparison_path=trained_artifact,
                base_path=baseline_artifact,
            )
        ],
        "trained_vs_random": trained_random_pairs,
        "random_vs_random": random_random_pairs,
    }

    results: dict[tuple[str, str, str], FamilySummary] = {}
    for readout_mode in ("raw", "model_norm"):
        for metric_mode in ("non_calibrated", "calibrated"):
            calibrated = metric_mode == "calibrated"
            for family_name, pairs in pair_groups.items():
                aggregated: dict[str, list[torch.Tensor]] = {}
                layer_labels: list[str] | None = None
                for pair_offset, pair in enumerate(pairs):
                    comparison_bundle = _load_bundle(pair.comparison_path)
                    base_bundle = _load_bundle(pair.base_path)
                    shared_keys = sorted(set(comparison_bundle) & set(base_bundle))
                    if not shared_keys:
                        raise ValueError(
                            f"No shared prompts between {pair.comparison_path} and {pair.base_path}"
                        )
                    for prompt_offset, key in enumerate(shared_keys):
                        comparison = comparison_bundle[key]
                        base = base_bundle[key]
                        if layer_labels is None:
                            layer_labels = _layer_labels_from_artifact(comparison)
                        curve = _similarity_curve(
                            comparison,
                            base,
                            readout_mode=readout_mode,
                            similarity_metric_name=similarity_metric_name,
                            top_k=top_k,
                            num_permutations=num_permutations,
                            alpha=alpha,
                            seed=seed + pair_offset * 1000 + prompt_offset,
                            calibrated=calibrated,
                        )
                        aggregated.setdefault(key, []).append(curve)
                if not aggregated:
                    raise ValueError("No prompt curves were collected for the requested similarity family")
                prompt_keys = sorted(aggregated)
                prompt_curves = torch.stack(
                    [torch.stack(aggregated[key], dim=0).mean(dim=0) for key in prompt_keys],
                    dim=0,
                )
                mean, ci_low, ci_high = _bootstrap_mean_ci(
                    prompt_curves,
                    num_bootstrap=1000,
                    alpha=alpha,
                    seed=seed,
                )
                results[(readout_mode, metric_mode, family_name)] = FamilySummary(
                    family=family_name,
                    mean=mean,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    per_prompt_curves=prompt_curves,
                    layer_labels=layer_labels or [],
                    prompt_keys=prompt_keys,
                )
    return results


def build_prompt_hidden_metric_summaries(
    *,
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
    hidden_metric_name: str,
    num_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[tuple[str, str, str], FamilySummary]:
    if len(random_artifacts) < 2:
        raise ValueError("Provide at least two random_artifacts to build a random-vs-random null family.")

    trained_random_pairs = [
        FamilyPair(family="trained_vs_random", comparison_path=trained_artifact, base_path=path)
        for path in random_artifacts
    ]
    random_random_pairs: list[FamilyPair] = []
    for idx, left in enumerate(random_artifacts):
        for right in random_artifacts[idx + 1 :]:
            random_random_pairs.append(
                FamilyPair(family="random_vs_random", comparison_path=left, base_path=right)
            )

    pair_groups: dict[str, list[FamilyPair]] = {
        "trained_vs_trained": [
            FamilyPair(
                family="trained_vs_trained",
                comparison_path=trained_artifact,
                base_path=baseline_artifact,
            )
        ],
        "trained_vs_random": trained_random_pairs,
        "random_vs_random": random_random_pairs,
    }

    results: dict[tuple[str, str, str], FamilySummary] = {}
    readout_mode = "hidden"
    for metric_mode in ("non_calibrated", "calibrated"):
        calibrated = metric_mode == "calibrated"
        for family_name, pairs in pair_groups.items():
            aggregated: dict[str, list[torch.Tensor]] = {}
            layer_labels: list[str] | None = None
            for pair_offset, pair in enumerate(pairs):
                comparison_bundle = _load_bundle(pair.comparison_path)
                base_bundle = _load_bundle(pair.base_path)
                shared_keys = sorted(set(comparison_bundle) & set(base_bundle))
                if not shared_keys:
                    raise ValueError(
                        f"No shared prompts between {pair.comparison_path} and {pair.base_path}"
                    )
                for prompt_offset, key in enumerate(shared_keys):
                    comparison = comparison_bundle[key]
                    base = base_bundle[key]
                    if layer_labels is None:
                        layer_labels = _layer_labels_from_artifact(comparison)
                    curve = _hidden_metric_curve(
                        comparison,
                        base,
                        metric_name=hidden_metric_name,
                        calibrated=calibrated,
                        num_permutations=num_permutations,
                        alpha=alpha,
                        seed=seed + pair_offset * 1000 + prompt_offset,
                    )
                    aggregated.setdefault(key, []).append(curve)
            if not aggregated:
                raise ValueError("No prompt curves were collected for the requested hidden-state family")
            prompt_keys = sorted(aggregated)
            prompt_curves = torch.stack(
                [torch.stack(aggregated[key], dim=0).mean(dim=0) for key in prompt_keys],
                dim=0,
            )
            mean, ci_low, ci_high = _bootstrap_mean_ci(
                prompt_curves,
                num_bootstrap=1000,
                alpha=alpha,
                seed=seed,
            )
            results[(readout_mode, metric_mode, family_name)] = FamilySummary(
                family=family_name,
                mean=mean,
                ci_low=ci_low,
                ci_high=ci_high,
                per_prompt_curves=prompt_curves,
                layer_labels=layer_labels or [],
                prompt_keys=prompt_keys,
            )
    return results


def build_prompt_pairwise_jsd_gap_curves(
    *,
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
    readout_mode: str,
    top_k: int = 10,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[list[dict[str, object]], list[str]]:
    pair_specs = _build_pair_specs(
        trained_artifact=trained_artifact,
        baseline_artifact=baseline_artifact,
        random_artifacts=random_artifacts,
    )

    null_prompt_curves, layer_labels, null_prompt_keys = _build_promptwise_family_curves(
        [
            FamilyPair(family="random_vs_random", comparison_path=left, base_path=right)
            for idx, left in enumerate(random_artifacts)
            for right in random_artifacts[idx + 1 :]
        ],
        readout_mode=readout_mode,
        metric_mode="non_calibrated",
        top_k=top_k,
        num_bins=15,
        min_samples_per_cell=2,
    )

    curves: list[dict[str, object]] = []
    for offset, (family, label, comparison_path, base_path) in enumerate(pair_specs):
        prompt_curves, _, prompt_keys = _build_promptwise_family_curves(
            [FamilyPair(family=family, comparison_path=comparison_path, base_path=base_path)],
            readout_mode=readout_mode,
            metric_mode="non_calibrated",
            top_k=top_k,
            num_bins=15,
            min_samples_per_cell=2,
        )
        if prompt_keys != null_prompt_keys:
            raise ValueError("Prompt key mismatch between pairwise JSD curves and random-null prompt curves")
        gap_curves = prompt_curves - null_prompt_curves
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            gap_curves,
            num_bootstrap=num_bootstrap,
            alpha=alpha,
            seed=seed + offset,
        )
        curves.append(
            {
                "family": family,
                "label": label,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "prompt_keys": prompt_keys,
            }
        )
    return curves, layer_labels


def build_prompt_pairwise_metric_gap_curves(
    *,
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
    readout_mode: str,
    metric_mode: str,
    top_k: int = 10,
    calibrated_metric_name: str = "top1_ece",
    num_bins: int = 15,
    min_samples_per_cell: int = 2,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[list[dict[str, object]], list[str]]:
    if metric_mode not in {"non_calibrated", "calibrated"}:
        raise ValueError(f"Unsupported metric_mode: {metric_mode!r}")

    pair_specs = _build_pair_specs(
        trained_artifact=trained_artifact,
        baseline_artifact=baseline_artifact,
        random_artifacts=random_artifacts,
    )

    null_prompt_curves, layer_labels, null_prompt_keys = _build_promptwise_family_curves(
        [
            FamilyPair(family="random_vs_random", comparison_path=left, base_path=right)
            for idx, left in enumerate(random_artifacts)
            for right in random_artifacts[idx + 1 :]
        ],
        readout_mode=readout_mode,
        metric_mode=metric_mode,
        top_k=top_k,
        num_bins=num_bins,
        min_samples_per_cell=min_samples_per_cell,
        calibrated_metric_name=calibrated_metric_name,
    )

    curves: list[dict[str, object]] = []
    for offset, (family, label, comparison_path, base_path) in enumerate(pair_specs):
        prompt_curves, _, prompt_keys = _build_promptwise_family_curves(
            [FamilyPair(family=family, comparison_path=comparison_path, base_path=base_path)],
            readout_mode=readout_mode,
            metric_mode=metric_mode,
            top_k=top_k,
            num_bins=num_bins,
            min_samples_per_cell=min_samples_per_cell,
            calibrated_metric_name=calibrated_metric_name,
        )
        if prompt_keys != null_prompt_keys:
            raise ValueError("Prompt key mismatch between pairwise metric curves and random-null prompt curves")
        gap_curves = prompt_curves - null_prompt_curves
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            gap_curves,
            num_bootstrap=num_bootstrap,
            alpha=alpha,
            seed=seed + offset,
        )
        curves.append(
            {
                "family": family,
                "label": label,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "prompt_keys": prompt_keys,
            }
        )
    return curves, layer_labels


def build_prompt_null_gap_summaries(
    summaries: dict[tuple[str, str, str], FamilySummary],
    *,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[tuple[str, str, str], FamilySummary]:
    gap_results: dict[tuple[str, str, str], FamilySummary] = {}
    readout_modes = sorted({key[0] for key in summaries})
    metric_modes = sorted({key[1] for key in summaries})
    for readout_mode in readout_modes:
        for metric_mode in metric_modes:
            null_key = (readout_mode, metric_mode, "random_vs_random")
            if null_key not in summaries:
                continue
            null_summary = summaries[null_key]
            gap_results[(readout_mode, metric_mode, "trained_vs_trained_minus_random_vs_random")] = _gap_summary(
                summaries[(readout_mode, metric_mode, "trained_vs_trained")],
                null_summary,
                label="trained_vs_trained_minus_random_vs_random",
                num_bootstrap=num_bootstrap,
                alpha=alpha,
                seed=seed,
            )
            gap_results[(readout_mode, metric_mode, "trained_vs_random_minus_random_vs_random")] = _gap_summary(
                summaries[(readout_mode, metric_mode, "trained_vs_random")],
                null_summary,
                label="trained_vs_random_minus_random_vs_random",
                num_bootstrap=num_bootstrap,
                alpha=alpha,
                seed=seed + 1,
            )
    return gap_results


def plot_prompt_null_sanity(
    summaries: dict[tuple[str, str, str], FamilySummary],
    *,
    title: str | None = None,
) -> go.Figure:
    family_order = ["trained_vs_trained", "trained_vs_random", "random_vs_random"]
    family_style = {
        "trained_vs_trained": dict(color="#111111", label="trained vs baseline"),
        "trained_vs_random": dict(color="#1f77b4", label="Trained vs random"),
        "random_vs_random": dict(color="#d62728", label="Random vs random"),
    }
    panel_titles = [
        "Raw | Non-calibrated (JSD)",
        "ModelNorm | Non-calibrated (JSD)",
        "Raw | Calibrated (Top-1 ECE)",
        "ModelNorm | Calibrated (Top-1 ECE)",
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=panel_titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )

    for row_idx, metric_mode in enumerate(("non_calibrated", "calibrated"), start=1):
        for col_idx, readout_mode in enumerate(("raw", "model_norm"), start=1):
            labels = summaries[(readout_mode, metric_mode, family_order[0])].layer_labels
            x = list(range(len(labels)))
            for family_name in family_order:
                summary = summaries[(readout_mode, metric_mode, family_name)]
                style = family_style[family_name]
                mean = summary.mean.detach().cpu()
                ci_low = summary.ci_low.detach().cpu()
                ci_high = summary.ci_high.detach().cpu()
                showlegend = row_idx == 1 and col_idx == 1
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=ci_high.tolist(),
                        mode="lines",
                        line=dict(color=style["color"], width=0),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=ci_low.tolist(),
                        mode="lines",
                        line=dict(color=style["color"], width=0),
                        fill="tonexty",
                        fillcolor=f"rgba({int(style['color'][1:3], 16)},{int(style['color'][3:5], 16)},{int(style['color'][5:7], 16)},0.18)",
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean.tolist(),
                        mode="lines",
                        name=style["label"],
                        line=dict(color=style["color"], width=3),
                        showlegend=showlegend,
                        hovertemplate=(
                            "Layer: %{text}<br>"
                            "Mean: %{y:.6f}<extra>" + style["label"] + "</extra>"
                        ),
                        text=labels,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
            fig.update_xaxes(
                tickmode="array",
                tickvals=x,
                ticktext=labels,
                tickangle=-35,
                title_text="Layers",
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text="Mean divergence" if metric_mode == "non_calibrated" else "Mean calibration gap",
                row=row_idx,
                col=col_idx,
            )

    fig.update_layout(
        title=dict(
            text=title or "Prompt-layer null sanity check with 95% bootstrap confidence intervals",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=1450,
        height=920,
        margin=dict(l=90, r=50, t=120, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def save_prompt_null_sanity_figure(
    fig: go.Figure,
    path: str | Path,
    *,
    format: str | None = None,
) -> Path:
    return save_plotly_figure(fig, path, format=format)


def save_prompt_null_sanity_pdf(
    summaries: dict[tuple[str, str, str], FamilySummary],
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    family_order = ["trained_vs_trained", "trained_vs_random", "random_vs_random"]
    family_style = {
        "trained_vs_trained": dict(color="#111111", label="trained vs baseline"),
        "trained_vs_random": dict(color="#1f77b4", label="Trained vs random"),
        "random_vs_random": dict(color="#d62728", label="Random vs random"),
    }
    panel_titles = [
        ("raw", "non_calibrated", "Raw | Non-calibrated (JSD)"),
        ("model_norm", "non_calibrated", "ModelNorm | Non-calibrated (JSD)"),
        ("raw", "calibrated", "Raw | Calibrated (Top-1 ECE)"),
        ("model_norm", "calibrated", "ModelNorm | Calibrated (Top-1 ECE)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5), sharex=False, sharey=False)
    fig.suptitle(
        title or "Prompt-layer null sanity check with 95% bootstrap confidence intervals",
        fontsize=24,
        fontweight="semibold",
        y=0.98,
    )

    for ax, (readout_mode, metric_mode, panel_title) in zip(axes.flat, panel_titles):
        labels = summaries[(readout_mode, metric_mode, family_order[0])].layer_labels
        x = list(range(len(labels)))
        for family_name in family_order:
            summary = summaries[(readout_mode, metric_mode, family_name)]
            style = family_style[family_name]
            mean = summary.mean.detach().cpu().numpy()
            ci_low = summary.ci_low.detach().cpu().numpy()
            ci_high = summary.ci_high.detach().cpu().numpy()
            ax.plot(x, mean, color=style["color"], linewidth=2.6, label=style["label"])
            ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.18, linewidth=0)
        ax.set_title(panel_title, fontsize=20, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("Layers", fontweight="semibold")
        ax.set_ylabel(
            "Mean divergence" if metric_mode == "non_calibrated" else "Mean calibration gap",
            fontweight="semibold",
        )
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_handles, legend_labels = _legend_row_grouped_items(handles, labels, ncol=3)
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.91))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prompt_pairwise_jsd_pdf(
    curves: list[dict[str, object]],
    layer_labels: list[str],
    path: str | Path,
    *,
    readout_mode: str,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    style_map = {
        "trained_vs_trained": dict(color="#111111"),
        "trained_vs_random": dict(color="#1f77b4"),
        "random_vs_random": dict(color="#d62728"),
    }
    fig, ax = plt.subplots(figsize=(17.5, 8.6))
    x = list(range(len(layer_labels)))
    for curve in curves:
        family = str(curve["family"])
        label = str(curve["label"])
        mean = curve["mean"].detach().cpu().numpy()  # type: ignore[union-attr]
        ci_low = curve["ci_low"].detach().cpu().numpy()  # type: ignore[union-attr]
        ci_high = curve["ci_high"].detach().cpu().numpy()  # type: ignore[union-attr]
        color = style_map[family]["color"]
        alpha = 0.95 if family == "trained_vs_trained" else 0.8
        linewidth = 3.0 if family == "trained_vs_trained" else 2.1
        ax.plot(x, mean, color=color, linewidth=linewidth, alpha=alpha, label=label)
        ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.12, linewidth=0)
    ax.set_title(
        title or f"{'Raw' if readout_mode == 'raw' else 'ModelNorm'} | pairwise layerwise JSD",
        fontsize=16,
        fontweight="semibold",
    )
    ax.set_ylabel("Mean JSD", fontweight="semibold")
    ax.set_xlabel("Layers", fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=0, ha="center")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False, ncol=2)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.90))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_combined_pairwise_jsd_pdf(
    *,
    raw_curves: list[dict[str, object]],
    raw_layer_labels: list[str],
    model_norm_curves: list[dict[str, object]],
    model_norm_layer_labels: list[str],
    path: str | Path,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_labels = [str(curve["label"]) for curve in raw_curves]
    label_style_map = _build_pair_style_map(ordered_labels)

    panel_fonts = _pairwise_header_font_pack(rows=1, cols=2)
    fig, axes = plt.subplots(1, 2, figsize=(58.0, 17.0), sharex=True, sharey=True)
    pairwise_suptitle = max(panel_fonts["title"] + 4, 36)
    fig.suptitle(
        title or "Non-Calibrated JSD | Raw & ModelNorm LogitDiff",
        fontsize=pairwise_suptitle,
        fontweight="semibold",
        y=0.985,
    )

    panels = [
        (axes[0], "Raw", raw_curves, raw_layer_labels),
        (axes[1], "ModelNorm", model_norm_curves, model_norm_layer_labels),
    ]
    for ax, panel_title, curves, layer_labels in panels:
        x = list(range(len(layer_labels)))
        for curve in curves:
            label = str(curve["label"])
            mean = curve["mean"].detach().cpu().numpy()  # type: ignore[union-attr]
            ci_low = curve["ci_low"].detach().cpu().numpy()  # type: ignore[union-attr]
            ci_high = curve["ci_high"].detach().cpu().numpy()  # type: ignore[union-attr]
            style = label_style_map.get(label, dict(color="#333333", linestyle="-", marker="o"))
            ax.plot(
                x,
                mean,
                color=style["color"],
                linewidth=2.4,
                alpha=0.9,
                label=label,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5.0,
                markerfacecolor="white",
                markeredgewidth=1.0,
            )
            ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.10, linewidth=0)
        ax.set_title(panel_title, fontsize=panel_fonts["title"], fontweight="semibold")
        ax.set_xlabel("")
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, rotation=0, ha="center")
        ax.tick_params(axis="both", labelsize=panel_fonts["ticks"])
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean JSD", fontweight="semibold", fontsize=panel_fonts["axis"])
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ncol = _legend_ncol_max_two_rows(labels, max_columns=6, prefer_one_row=True)
    legend_fontsize = _legend_fontsize_for_labels(labels, base=panel_fonts["legend"], min_size=42, max_size=48)
    legend_handles, legend_labels = _legend_row_grouped_items(handles, labels, ncol=legend_ncol)
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        columnspacing=1.4,
        handlelength=2.8,
        labelspacing=0.6,
    )
    _finalize_header_layout(
        fig,
        legend=legend,
        labels=labels,
        n_panel_rows=1,
        n_panel_cols=2,
        max_columns=max(1, legend_ncol),
        left=0.06,
        right=0.99,
        bottom=0.14,
        min_top=0.72,
        wspace=0.03,
        hspace=0.18,
        gap_below_legend=0.038,
    )
    fig.supxlabel("Layers", fontsize=panel_fonts["axis"], fontweight="semibold", y=0.06)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_combined_pairwise_jsd_gap_pdf(
    *,
    raw_gap_curves: list[dict[str, object]],
    raw_layer_labels: list[str],
    model_norm_gap_curves: list[dict[str, object]],
    model_norm_layer_labels: list[str],
    path: str | Path,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_labels = [str(curve["label"]) for curve in raw_gap_curves]
    label_style_map = _build_pair_style_map(ordered_labels)

    panel_fonts = _pairwise_header_font_pack(rows=1, cols=2)
    fig, axes = plt.subplots(1, 2, figsize=(58.0, 17.0), sharex=True, sharey=True)
    pairwise_suptitle = max(panel_fonts["title"] + 4, 36)
    fig.suptitle(
        title or "Non-Calibrated ΔJSD | Raw & ModelNorm LogitDiff",
        fontsize=pairwise_suptitle,
        fontweight="semibold",
        y=0.985,
    )

    panels = [
        (axes[0], "Raw", raw_gap_curves, raw_layer_labels),
        (axes[1], "ModelNorm", model_norm_gap_curves, model_norm_layer_labels),
    ]
    for ax, panel_title, curves, layer_labels in panels:
        x = list(range(len(layer_labels)))
        for curve in curves:
            label = str(curve["label"])
            mean = curve["mean"].detach().cpu().numpy()  # type: ignore[union-attr]
            ci_low = curve["ci_low"].detach().cpu().numpy()  # type: ignore[union-attr]
            ci_high = curve["ci_high"].detach().cpu().numpy()  # type: ignore[union-attr]
            style = label_style_map.get(label, dict(color="#333333", linestyle="-", marker="o"))
            line = ax.plot(
                x,
                mean,
                color=style["color"],
                linewidth=2.2,
                alpha=0.88,
                label=label,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5.5,
                markerfacecolor="white",
                markeredgewidth=1.1,
            )[0]
            ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.10, linewidth=0)
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9)
        ax.set_title(panel_title, fontsize=panel_fonts["title"], fontweight="semibold")
        ax.set_xlabel("")
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, rotation=0, ha="center")
        ax.tick_params(axis="both", labelsize=panel_fonts["ticks"])
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("ΔJSD vs random-null", fontweight="semibold", fontsize=panel_fonts["axis"])
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ncol = _legend_ncol_max_two_rows(labels, max_columns=6, prefer_one_row=True)
    legend_fontsize = _legend_fontsize_for_labels(labels, base=panel_fonts["legend"], min_size=42, max_size=48)
    legend_handles, legend_labels = _legend_row_grouped_items(handles, labels, ncol=legend_ncol)
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        columnspacing=1.4,
        handlelength=2.8,
        labelspacing=0.6,
    )
    _finalize_header_layout(
        fig,
        legend=legend,
        labels=labels,
        n_panel_rows=1,
        n_panel_cols=2,
        max_columns=max(1, legend_ncol),
        left=0.06,
        right=0.99,
        bottom=0.14,
        min_top=0.72,
        wspace=0.03,
        hspace=0.18,
        gap_below_legend=0.038,
    )
    fig.supxlabel("Layers", fontsize=panel_fonts["axis"], fontweight="semibold", y=0.06)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_combined_pairwise_metric_pdf(
    *,
    raw_curves: list[dict[str, object]],
    raw_layer_labels: list[str],
    model_norm_curves: list[dict[str, object]],
    model_norm_layer_labels: list[str],
    path: str | Path,
    metric_label: str,
    y_label: str,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_labels = [str(curve["label"]) for curve in raw_curves]
    label_style_map = _build_pair_style_map(ordered_labels)

    panel_fonts = _pairwise_header_font_pack(rows=1, cols=2)
    fig, axes = plt.subplots(1, 2, figsize=(58.0, 17.0), sharex=True, sharey=True)
    pairwise_suptitle = max(panel_fonts["title"] + 4, 36)
    fig.suptitle(
        title or f"{metric_label} | Raw & ModelNorm LogitDiff",
        fontsize=pairwise_suptitle,
        fontweight="semibold",
        y=0.985,
    )

    panels = [
        (axes[0], "Raw", raw_curves, raw_layer_labels),
        (axes[1], "ModelNorm", model_norm_curves, model_norm_layer_labels),
    ]
    for ax, panel_title, curves, layer_labels in panels:
        x = list(range(len(layer_labels)))
        for curve in curves:
            label = str(curve["label"])
            mean = curve["mean"].detach().cpu().numpy()  # type: ignore[union-attr]
            ci_low = curve["ci_low"].detach().cpu().numpy()  # type: ignore[union-attr]
            ci_high = curve["ci_high"].detach().cpu().numpy()  # type: ignore[union-attr]
            style = label_style_map.get(label, dict(color="#333333", linestyle="-", marker="o"))
            ax.plot(
                x,
                mean,
                color=style["color"],
                linewidth=2.4,
                alpha=0.9,
                label=label,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5.0,
                markerfacecolor="white",
                markeredgewidth=1.0,
            )
            ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.10, linewidth=0)
        ax.set_title(panel_title, fontsize=panel_fonts["title"], fontweight="semibold")
        ax.set_xlabel("")
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, rotation=0, ha="center")
        ax.tick_params(axis="both", labelsize=panel_fonts["ticks"])
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(y_label, fontweight="semibold", fontsize=panel_fonts["axis"])
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ncol = _legend_ncol_max_two_rows(labels, max_columns=6, prefer_one_row=True)
    legend_fontsize = _legend_fontsize_for_labels(labels, base=panel_fonts["legend"], min_size=42, max_size=48)
    legend_handles, legend_labels = _legend_row_grouped_items(handles, labels, ncol=legend_ncol)
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        columnspacing=1.4,
        handlelength=2.8,
        labelspacing=0.6,
    )
    _finalize_header_layout(
        fig,
        legend=legend,
        labels=labels,
        n_panel_rows=1,
        n_panel_cols=2,
        max_columns=max(1, legend_ncol),
        left=0.06,
        right=0.99,
        bottom=0.14,
        min_top=0.72,
        wspace=0.03,
        hspace=0.18,
        gap_below_legend=0.038,
    )
    fig.supxlabel("Layers", fontsize=panel_fonts["axis"], fontweight="semibold", y=0.06)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_combined_pairwise_metric_gap_pdf(
    *,
    raw_gap_curves: list[dict[str, object]],
    raw_layer_labels: list[str],
    model_norm_gap_curves: list[dict[str, object]],
    model_norm_layer_labels: list[str],
    path: str | Path,
    metric_label: str,
    y_label: str,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_labels = [str(curve["label"]) for curve in raw_gap_curves]
    label_style_map = _build_pair_style_map(ordered_labels)

    panel_fonts = _pairwise_header_font_pack(rows=1, cols=2)
    fig, axes = plt.subplots(1, 2, figsize=(58.0, 17.0), sharex=True, sharey=True)
    pairwise_suptitle = max(panel_fonts["title"] + 4, 36)
    fig.suptitle(
        title or f"Δ{metric_label} | Raw & ModelNorm LogitDiff",
        fontsize=pairwise_suptitle,
        fontweight="semibold",
        y=0.985,
    )

    panels = [
        (axes[0], "Raw", raw_gap_curves, raw_layer_labels),
        (axes[1], "ModelNorm", model_norm_gap_curves, model_norm_layer_labels),
    ]
    for ax, panel_title, curves, layer_labels in panels:
        x = list(range(len(layer_labels)))
        for curve in curves:
            label = str(curve["label"])
            mean = curve["mean"].detach().cpu().numpy()  # type: ignore[union-attr]
            ci_low = curve["ci_low"].detach().cpu().numpy()  # type: ignore[union-attr]
            ci_high = curve["ci_high"].detach().cpu().numpy()  # type: ignore[union-attr]
            style = label_style_map.get(label, dict(color="#333333", linestyle="-", marker="o"))
            ax.plot(
                x,
                mean,
                color=style["color"],
                linewidth=2.2,
                alpha=0.88,
                label=label,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5.5,
                markerfacecolor="white",
                markeredgewidth=1.1,
            )
            ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.10, linewidth=0)
        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9)
        ax.set_title(panel_title, fontsize=panel_fonts["title"], fontweight="semibold")
        ax.set_xlabel("Layers", fontweight="semibold", fontsize=panel_fonts["axis"])
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, rotation=0, ha="center")
        ax.tick_params(axis="both", labelsize=panel_fonts["ticks"])
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(y_label, fontweight="semibold", fontsize=panel_fonts["axis"])
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ncol = _legend_ncol_max_two_rows(labels, max_columns=6, prefer_one_row=True)
    legend_fontsize = _legend_fontsize_for_labels(labels, base=panel_fonts["legend"], min_size=42, max_size=48)
    legend_handles, legend_labels = _legend_row_grouped_items(handles, labels, ncol=legend_ncol)
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        ncol=legend_ncol,
        fontsize=legend_fontsize,
        columnspacing=1.4,
        handlelength=2.8,
        labelspacing=0.6,
    )
    _finalize_header_layout(
        fig,
        legend=legend,
        labels=labels,
        n_panel_rows=1,
        n_panel_cols=2,
        max_columns=max(1, legend_ncol),
        left=0.06,
        right=0.99,
        bottom=0.14,
        min_top=0.72,
        wspace=0.03,
        hspace=0.18,
        gap_below_legend=0.038,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_robust_prompt_jsd_overview_pdf(
    *,
    raw_curves: list[dict[str, object]],
    raw_layer_labels: list[str],
    raw_gap_curves: list[dict[str, object]],
    model_norm_curves: list[dict[str, object]],
    model_norm_layer_labels: list[str],
    model_norm_gap_curves: list[dict[str, object]],
    path: str | Path,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_labels = [str(curve["label"]) for curve in raw_curves]
    label_style_map = _build_pair_style_map(ordered_labels)

    panel_fonts = _pairwise_header_font_pack(rows=2, cols=2)
    fig, axes = plt.subplots(2, 2, figsize=(58.0, 23.0), sharex="col", sharey="row")
    fig.suptitle(
        title or "Non-Calibrated JSD | Raw & ModelNorm LogitDiff",
        fontsize=panel_fonts["suptitle"],
        fontweight="semibold",
        y=0.985,
    )

    panels = [
        (axes[0, 0], "Non-Calibrated Raw | JSD", raw_curves, raw_layer_labels, "Mean JSD"),
        (axes[0, 1], "Non-Calibrated ModelNorm | JSD", model_norm_curves, model_norm_layer_labels, None),
        (axes[1, 0], "Non-Calibrated Raw | ΔJSD vs random baseline", raw_gap_curves, raw_layer_labels, "ΔJSD"),
        (axes[1, 1], "Non-Calibrated ModelNorm | ΔJSD vs random baseline", model_norm_gap_curves, model_norm_layer_labels, None),
    ]

    top_handles = []
    top_labels = []

    for panel_idx, (ax, panel_title, curves, layer_labels, y_label) in enumerate(panels):
        x = list(range(len(layer_labels)))
        row_idx = panel_idx // 2
        for curve_idx, curve in enumerate(curves):
            family = str(curve["family"])
            label = str(curve["label"])
            mean = curve["mean"].detach().cpu().numpy()  # type: ignore[union-attr]
            ci_low = curve["ci_low"].detach().cpu().numpy()  # type: ignore[union-attr]
            ci_high = curve["ci_high"].detach().cpu().numpy()  # type: ignore[union-attr]
            style = label_style_map.get(
                label,
                dict(color="#333333", linestyle="-", marker="o"),
            )
            color = style["color"]
            alpha = 0.95 if family == "trained_vs_trained" else 0.82
            linewidth = 3.0 if family == "trained_vs_trained" else 2.0
            line = ax.plot(
                x,
                mean,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5.5,
                markerfacecolor="white",
                markeredgewidth=1.1,
            )[0]
            ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.10, linewidth=0)
            if label not in top_labels:
                top_handles.append(line)
                top_labels.append(label)
        if "Δ" in panel_title:
            ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9)
        ax.set_title(panel_title, fontsize=panel_fonts["title"], fontweight="semibold")
        if y_label is not None:
            ax.set_ylabel(y_label, fontweight="semibold", fontsize=panel_fonts["axis"])
        ax.set_xlabel("")
        ax.set_xticks(x)
        if row_idx == 0:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticklabels(layer_labels, rotation=0, ha="center")
        ax.tick_params(axis="both", labelsize=panel_fonts["ticks"])
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    top_legend_ncol = _legend_ncol_max_two_rows(top_labels, max_columns=6, prefer_one_row=True)
    top_legend_fontsize = _legend_fontsize_for_labels(top_labels, base=panel_fonts["legend"], min_size=42, max_size=48)
    legend_handles, legend_labels = _legend_row_grouped_items(top_handles, top_labels, ncol=top_legend_ncol)
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        frameon=False,
        ncol=top_legend_ncol,
        fontsize=top_legend_fontsize,
        columnspacing=1.4,
        handlelength=2.6,
        labelspacing=0.6,
    )
    _finalize_header_layout(
        fig,
        legend=legend,
        labels=top_labels,
        n_panel_rows=2,
        n_panel_cols=2,
        max_columns=max(1, top_legend_ncol),
        left=0.06,
        right=0.99,
        bottom=0.09,
        min_top=0.72,
        wspace=0.015,
        hspace=0.28,
        gap_below_legend=0.038,
    )
    fig.supxlabel("Layers", fontsize=panel_fonts["axis"], fontweight="semibold", y=0.05)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prompt_metric_score_and_gap_pdf(
    summaries: dict[tuple[str, str, str], FamilySummary],
    gap_summaries: dict[tuple[str, str, str], FamilySummary],
    path: str | Path,
    *,
    metric_mode: str,
    metric_label_override: str | None = None,
    actual_family_labels: dict[str, str] | None = None,
    gap_family_labels: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if metric_mode not in {"non_calibrated", "calibrated"}:
        raise ValueError(f"Unsupported metric_mode: {metric_mode!r}")

    actual_family_order = ["trained_vs_trained", "trained_vs_random", "random_vs_random"]
    actual_family_style = {
        "trained_vs_trained": dict(color="#111111", label=(actual_family_labels or {}).get("trained_vs_trained", "trained vs baseline")),
        "trained_vs_random": dict(color="#1f77b4", label=(actual_family_labels or {}).get("trained_vs_random", "trained vs rand-mean")),
        "random_vs_random": dict(color="#d62728", label=(actual_family_labels or {}).get("random_vs_random", "rand-pair mean")),
    }
    gap_family_order = [
        "trained_vs_trained_minus_random_vs_random",
        "trained_vs_random_minus_random_vs_random",
    ]
    gap_family_style = {
        "trained_vs_trained_minus_random_vs_random": dict(color="#9467bd", label=(gap_family_labels or {}).get("trained_vs_trained_minus_random_vs_random", "trained-baseline - rand-pair mean")),
        "trained_vs_random_minus_random_vs_random": dict(color="#ff7f0e", label=(gap_family_labels or {}).get("trained_vs_random_minus_random_vs_random", "trained-rand-mean - rand-pair mean")),
    }

    metric_label = metric_label_override or ("JSD" if metric_mode == "non_calibrated" else "Top-1 ECE")
    display_metric_label = _compact_metric_label(metric_label)
    y_label = "Mean JSD" if metric_mode == "non_calibrated" else f"Mean {display_metric_label}"
    gap_label = f"Δ{display_metric_label}"

    title_fonts = _pairwise_header_font_pack(rows=2, cols=2)
    fig, axes = plt.subplots(2, 2, figsize=(48.0, 19.5), sharex="col", sharey="row")
    fig.suptitle(
        title or f"{metric_label} | Raw & ModelNorm LogitDiff",
        fontsize=title_fonts["suptitle"],
        fontweight="semibold",
        y=0.985,
    )

    for col_idx, readout_mode in enumerate(("raw", "model_norm")):
        actual_ax = axes[0, col_idx]
        gap_ax = axes[1, col_idx]
        labels = summaries[(readout_mode, metric_mode, actual_family_order[0])].layer_labels
        x = list(range(len(labels)))

        for family_name in actual_family_order:
            summary = summaries[(readout_mode, metric_mode, family_name)]
            style = actual_family_style[family_name]
            mean = summary.mean.detach().cpu().numpy()
            ci_low = summary.ci_low.detach().cpu().numpy()
            ci_high = summary.ci_high.detach().cpu().numpy()
            actual_ax.plot(x, mean, color=style["color"], linewidth=2.6, label=style["label"])
            actual_ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.18, linewidth=0)

        actual_ax.set_title(
            f"{'Raw' if readout_mode == 'raw' else 'ModelNorm'} | {display_metric_label}",
            fontsize=title_fonts["title"],
            fontweight="semibold",
        )
        if col_idx == 0:
            actual_ax.set_ylabel(y_label, fontweight="semibold", fontsize=title_fonts["axis"])
        else:
            actual_ax.set_ylabel("")
        actual_ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        actual_ax.spines["top"].set_visible(False)
        actual_ax.spines["right"].set_visible(False)

        for family_name in gap_family_order:
            summary = gap_summaries[(readout_mode, metric_mode, family_name)]
            style = gap_family_style[family_name]
            mean = summary.mean.detach().cpu().numpy()
            ci_low = summary.ci_low.detach().cpu().numpy()
            ci_high = summary.ci_high.detach().cpu().numpy()
            gap_ax.plot(x, mean, color=style["color"], linewidth=2.6, label=style["label"])
            gap_ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.18, linewidth=0)

        gap_ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9)
        gap_ax.set_title(
            f"{'Raw' if readout_mode == 'raw' else 'ModelNorm'} | Δ{display_metric_label} vs random-null",
            fontsize=title_fonts["title"],
            fontweight="semibold",
        )
        if col_idx == 0:
            gap_ax.set_ylabel(gap_label, fontweight="semibold", fontsize=title_fonts["axis"])
        else:
            gap_ax.set_ylabel("")
        gap_ax.set_xticks(x)
        gap_ax.set_xticklabels(labels, rotation=0, ha="center")
        gap_ax.set_xlabel("")
        actual_ax.tick_params(axis="x", labelbottom=False)
        actual_ax.tick_params(axis="both", labelsize=title_fonts["ticks"])
        gap_ax.tick_params(axis="both", labelsize=title_fonts["ticks"])
        gap_ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        gap_ax.spines["top"].set_visible(False)
        gap_ax.spines["right"].set_visible(False)

    handles_top, labels_top = axes[0, 0].get_legend_handles_labels()
    handles_bottom, labels_bottom = axes[1, 0].get_legend_handles_labels()
    combined_handles = handles_top + handles_bottom
    combined_labels = labels_top + labels_bottom
    legend_ncol = _legend_ncol_max_two_rows(combined_labels, max_columns=6, prefer_one_row=False)
    legend_handles, legend_labels = _legend_row_grouped_items(combined_handles, combined_labels, ncol=legend_ncol)
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=_legend_fontsize_for_labels(combined_labels, base=34, min_size=31, max_size=38),
        columnspacing=1.0,
        handlelength=2.4,
        labelspacing=0.5,
    )
    _finalize_header_layout(
        fig,
        legend=legend,
        labels=combined_labels,
        n_panel_rows=2,
        n_panel_cols=2,
        max_columns=max(1, legend_ncol),
        left=0.07,
        right=0.99,
        bottom=0.10,
        min_top=0.82,
        wspace=0.04,
        hspace=0.24,
        gap_below_legend=0.016,
    )
    fig.supxlabel("Layers", fontsize=title_fonts["axis"], fontweight="semibold", y=0.04)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prompt_metric_actual_summary_pdf(
    summaries: dict[tuple[str, str, str], FamilySummary],
    path: str | Path,
    *,
    metric_mode: str,
    metric_label_override: str | None = None,
    family_labels: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if metric_mode not in {"non_calibrated", "calibrated"}:
        raise ValueError(f"Unsupported metric_mode: {metric_mode!r}")

    actual_family_order = ["trained_vs_trained", "trained_vs_random", "random_vs_random"]
    actual_family_style = {
        "trained_vs_trained": dict(color="#111111", label=(family_labels or {}).get("trained_vs_trained", "trained vs baseline")),
        "trained_vs_random": dict(color="#1f77b4", label=(family_labels or {}).get("trained_vs_random", "trained vs rand-mean")),
        "random_vs_random": dict(color="#d62728", label=(family_labels or {}).get("random_vs_random", "rand-pair mean")),
    }

    metric_label = metric_label_override or ("JSD" if metric_mode == "non_calibrated" else "Top-1 ECE")
    y_label = "Mean divergence" if metric_mode == "non_calibrated" else f"Mean {metric_label}"

    title_fonts = _pairwise_header_font_pack(rows=1, cols=2)
    fig, axes = plt.subplots(1, 2, figsize=(40.0, 13.4), sharex=True, sharey=True)
    fig.suptitle(
        title or f"{metric_label} | Raw & ModelNorm LogitDiff",
        fontsize=title_fonts["suptitle"],
        fontweight="semibold",
        y=0.985,
    )

    for col_idx, readout_mode in enumerate(("raw", "model_norm")):
        ax = axes[col_idx]
        labels = summaries[(readout_mode, metric_mode, actual_family_order[0])].layer_labels
        x = list(range(len(labels)))
        for family_name in actual_family_order:
            summary = summaries[(readout_mode, metric_mode, family_name)]
            style = actual_family_style[family_name]
            mean = summary.mean.detach().cpu().numpy()
            ci_low = summary.ci_low.detach().cpu().numpy()
            ci_high = summary.ci_high.detach().cpu().numpy()
            ax.plot(x, mean, color=style["color"], linewidth=2.6, label=style["label"])
            ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.16, linewidth=0)
        ax.set_title(
            f"{'Raw' if readout_mode == 'raw' else 'ModelNorm'} | {metric_label}",
            fontsize=title_fonts["title"],
            fontweight="semibold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=title_fonts["ticks"])

    axes[0].set_ylabel(y_label, fontweight="semibold", fontsize=title_fonts["axis"])
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ncol = _legend_ncol_max_two_rows(labels, max_columns=4, prefer_one_row=True)
    legend_fontsize = _legend_fontsize_for_labels(labels, base=30, min_size=27, max_size=34)
    legend_handles, legend_labels = _legend_row_grouped_items(handles, labels, ncol=legend_ncol)
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.962),
        fontsize=legend_fontsize,
        columnspacing=1.6,
        handlelength=2.8,
        labelspacing=0.6,
    )
    _finalize_header_layout(
        fig,
        legend=legend,
        labels=labels,
        n_panel_rows=1,
        n_panel_cols=2,
        max_columns=max(1, legend_ncol),
        left=0.07,
        right=0.99,
        bottom=0.10,
        min_top=0.82,
        wspace=0.03,
        hspace=0.20,
        gap_below_legend=0.016,
    )
    fig.supxlabel("Layers", fontsize=title_fonts["axis"], fontweight="semibold", y=0.04)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prompt_metric_bundle_actual_pdf(
    metric_summaries: list[tuple[str, dict[tuple[str, str, str], FamilySummary], str]],
    path: str | Path,
    *,
    family_labels: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actual_family_order = ["trained_vs_trained", "trained_vs_random", "random_vs_random"]
    actual_family_style = {
        "trained_vs_trained": dict(color="#111111", label=(family_labels or {}).get("trained_vs_trained", "trained vs baseline")),
        "trained_vs_random": dict(color="#1f77b4", label=(family_labels or {}).get("trained_vs_random", "trained vs rand-mean")),
        "random_vs_random": dict(color="#d62728", label=(family_labels or {}).get("random_vs_random", "rand-pair mean")),
    }

    n_rows = len(metric_summaries)
    title_fonts = _pairwise_header_font_pack(rows=n_rows, cols=2)
    fig_height = 9.0 * n_rows if n_rows <= 2 else 9.2 * n_rows
    fig, axes = plt.subplots(n_rows, 2, figsize=(44.0, fig_height), sharex="col", sharey="row")
    if len(metric_summaries) == 1:
        axes = [axes]
    fig.suptitle(title or "Raw & ModelNorm LogitDiff", fontsize=title_fonts["suptitle"], fontweight="semibold", y=0.985)

    for row_idx, (metric_mode, summaries, metric_label) in enumerate(metric_summaries):
        for col_idx, readout_mode in enumerate(("raw", "model_norm")):
            ax = axes[row_idx][col_idx]
            labels = summaries[(readout_mode, metric_mode, actual_family_order[0])].layer_labels
            x = list(range(len(labels)))
            for family_name in actual_family_order:
                summary = summaries[(readout_mode, metric_mode, family_name)]
                style = actual_family_style[family_name]
                mean = summary.mean.detach().cpu().numpy()
                ci_low = summary.ci_low.detach().cpu().numpy()
                ci_high = summary.ci_high.detach().cpu().numpy()
                ax.plot(x, mean, color=style["color"], linewidth=2.8, label=style["label"])
                ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.15, linewidth=0)
            calibration_label = "Non-Calibrated" if metric_mode == "non_calibrated" else "Calibrated"
            lens_label = "Raw" if readout_mode == "raw" else "ModelNorm"
            ax.set_title(
                f"{calibration_label} | {lens_label} | {metric_label}",
                fontsize=_responsive_font_size(rows=n_rows, cols=2, base=36, min_size=32),
                fontweight="semibold",
            )
            ax.set_xticks(x)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(labels, rotation=0, ha="center")
                ax.set_xlabel("")
            else:
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_xlabel("")
            ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx == 0 and metric_mode == "non_calibrated":
                ax.set_ylabel("Mean JSD", fontweight="semibold", fontsize=_responsive_font_size(rows=n_rows, cols=2, base=28, min_size=25))
            elif col_idx == 0:
                ax.set_ylabel(f"Mean {metric_label}", fontweight="semibold", fontsize=_responsive_font_size(rows=n_rows, cols=2, base=28, min_size=25))
            else:
                ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=_responsive_font_size(rows=n_rows, cols=2, base=26, min_size=23))

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_ncol = _legend_ncol_max_two_rows(labels, max_columns=3)
    legend_fontsize = max(_legend_fontsize_for_labels(labels, base=30, min_size=27, max_size=34), _responsive_font_size(rows=n_rows, cols=2, base=29, min_size=27))
    legend_handles, legend_labels = _legend_row_grouped_items(handles, labels, ncol=legend_ncol)
    legend_y = 0.948 if n_rows <= 2 else 0.944
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.5, legend_y),
        fontsize=legend_fontsize,
        handlelength=2.6,
        columnspacing=1.6,
        labelspacing=0.6,
    )
    _finalize_header_layout(
        fig,
        legend=legend,
        labels=labels,
        n_panel_rows=n_rows,
        n_panel_cols=2,
        max_columns=max(1, legend_ncol),
        left=0.07,
        right=0.99,
        bottom=0.08,
        min_top=0.70 if n_rows <= 2 else 0.80,
        wspace=0.05,
        hspace=0.42 if n_rows <= 2 else 0.34,
        gap_below_legend=0.032 if n_rows <= 2 else 0.010,
    )
    fig.supxlabel("Layers", fontsize=title_fonts["axis"], fontweight="semibold", y=0.03)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prompt_metric_bundle_gap_pdf(
    metric_gap_summaries: list[tuple[str, dict[tuple[str, str, str], FamilySummary], str]],
    path: str | Path,
    *,
    family_labels: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gap_family_order = [
        "trained_vs_trained_minus_random_vs_random",
        "trained_vs_random_minus_random_vs_random",
    ]
    gap_family_style = {
        "trained_vs_trained_minus_random_vs_random": dict(color="#9467bd", label=(family_labels or {}).get("trained_vs_trained_minus_random_vs_random", "train-base - rand-pair")),
        "trained_vs_random_minus_random_vs_random": dict(color="#ff7f0e", label=(family_labels or {}).get("trained_vs_random_minus_random_vs_random", "train-rand - rand-pair")),
    }

    n_rows = len(metric_gap_summaries)
    title_fonts = _pairwise_header_font_pack(rows=n_rows, cols=2)
    fig_height = 9.0 * n_rows if n_rows <= 2 else 9.2 * n_rows
    fig, axes = plt.subplots(n_rows, 2, figsize=(44.0, fig_height), sharex="col", sharey="row")
    if len(metric_gap_summaries) == 1:
        axes = [axes]
    fig.suptitle(title or "Raw & ModelNorm ΔLogitDiff", fontsize=title_fonts["suptitle"], fontweight="semibold", y=0.985)

    for row_idx, (metric_mode, gap_summaries, metric_label) in enumerate(metric_gap_summaries):
        for col_idx, readout_mode in enumerate(("raw", "model_norm")):
            ax = axes[row_idx][col_idx]
            labels = gap_summaries[(readout_mode, metric_mode, gap_family_order[0])].layer_labels
            x = list(range(len(labels)))
            for family_name in gap_family_order:
                summary = gap_summaries[(readout_mode, metric_mode, family_name)]
                style = gap_family_style[family_name]
                mean = summary.mean.detach().cpu().numpy()
                ci_low = summary.ci_low.detach().cpu().numpy()
                ci_high = summary.ci_high.detach().cpu().numpy()
                ax.plot(x, mean, color=style["color"], linewidth=2.8, label=style["label"])
                ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.15, linewidth=0)
            ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9)
            calibration_label = "Non-Calibrated" if metric_mode == "non_calibrated" else "Calibrated"
            lens_label = "Raw" if readout_mode == "raw" else "ModelNorm"
            ax.set_title(
                f"{calibration_label} | {lens_label} | Δ{metric_label}",
                fontsize=_responsive_font_size(rows=n_rows, cols=2, base=36, min_size=32),
                fontweight="semibold",
            )
            ax.set_xticks(x)
            if row_idx == n_rows - 1:
                ax.set_xticklabels(labels, rotation=0, ha="center")
                ax.set_xlabel("")
            else:
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel(f"Δ{metric_label}", fontweight="semibold", fontsize=_responsive_font_size(rows=n_rows, cols=2, base=28, min_size=25))
            else:
                ax.set_ylabel("")
            ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", labelsize=_responsive_font_size(rows=n_rows, cols=2, base=26, min_size=23))

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_ncol = _legend_ncol_max_two_rows(labels, max_columns=2, prefer_one_row=False)
    legend_fontsize = max(_legend_fontsize_for_labels(labels, base=30, min_size=27, max_size=34), _responsive_font_size(rows=n_rows, cols=2, base=29, min_size=27))
    legend_handles, legend_labels = _legend_row_grouped_items(handles, labels, ncol=legend_ncol)
    legend_y = 0.948 if n_rows <= 2 else 0.944
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.5, legend_y),
        fontsize=legend_fontsize,
        handlelength=2.6,
        columnspacing=1.6,
        labelspacing=0.6,
    )
    _finalize_header_layout(
        fig,
        legend=legend,
        labels=labels,
        n_panel_rows=n_rows,
        n_panel_cols=2,
        max_columns=max(1, legend_ncol),
        left=0.07,
        right=0.99,
        bottom=0.08,
        min_top=0.70 if n_rows <= 2 else 0.80,
        wspace=0.05,
        hspace=0.42 if n_rows <= 2 else 0.34,
        gap_below_legend=0.032 if n_rows <= 2 else 0.010,
    )
    fig.supxlabel("Layers", fontsize=title_fonts["axis"], fontweight="semibold", y=0.03)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prompt_jsd_calibration_overlay_pdf(
    jsd_summaries: dict[tuple[str, str, str], FamilySummary],
    calibrated_summaries: dict[tuple[str, str, str], FamilySummary],
    path: str | Path,
    *,
    calibrated_label: str = "Top-1 ECE",
    family_titles_override: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    family_order = ["trained_vs_trained", "trained_vs_random", "random_vs_random"]
    family_titles = {
        "trained_vs_trained": (family_titles_override or {}).get("trained_vs_trained", "trained vs baseline"),
        "trained_vs_random": (family_titles_override or {}).get("trained_vs_random", "trained vs rand-mean"),
        "random_vs_random": (family_titles_override or {}).get("random_vs_random", "rand-pair mean"),
    }
    compact_metric = _compact_metric_label(calibrated_label)
    use_single_gap_axis = _overlay_axis_mode(calibrated_label, gap=True) == "shared"
    use_single_metric_axis = _overlay_axis_mode(calibrated_label, gap=False) == "shared"

    jsd_lows: list[float] = []
    jsd_highs: list[float] = []
    cal_lows: list[float] = []
    cal_highs: list[float] = []
    for family_name in family_order:
        for readout_mode in ("raw", "model_norm"):
            jsd_summary = jsd_summaries[(readout_mode, "non_calibrated", family_name)]
            cal_summary = calibrated_summaries[(readout_mode, "calibrated", family_name)]
            jsd_lows.extend(jsd_summary.ci_low.detach().cpu().tolist())
            jsd_highs.extend(jsd_summary.ci_high.detach().cpu().tolist())
            cal_lows.extend(cal_summary.ci_low.detach().cpu().tolist())
            cal_highs.extend(cal_summary.ci_high.detach().cpu().tolist())

    def _axis_limits(low_values: list[float], high_values: list[float]) -> tuple[float, float]:
        lo = min(low_values)
        hi = max(high_values)
        span = max(hi - lo, 1e-8)
        pad = 0.06 * span
        return lo - pad, hi + pad

    jsd_ylim = _axis_limits(jsd_lows, jsd_highs)
    cal_ylim = _axis_limits(cal_lows, cal_highs)
    shared_gap_ylim = _axis_limits(jsd_lows + cal_lows, jsd_highs + cal_highs) if use_single_gap_axis else None
    shared_metric_ylim = _axis_limits(jsd_lows + cal_lows, jsd_highs + cal_highs) if use_single_metric_axis else None

    title_fonts = _pairwise_header_font_pack(rows=3, cols=2)
    fig, axes = plt.subplots(3, 2, figsize=(38.0, 18.0), sharex="col", sharey="row")
    fig.suptitle(title or f"Raw & ModelNorm: Non-Calibrated LogitDiff (JSD) vs Calibration Metric ({calibrated_label})", fontsize=title_fonts["suptitle"], fontweight="semibold", y=0.985)

    legend = None

    for row_idx, family_name in enumerate(family_order):
        for col_idx, readout_mode in enumerate(("raw", "model_norm")):
            ax = axes[row_idx, col_idx]
            cal_ax = ax if use_single_metric_axis else ax.twinx()
            jsd_summary = jsd_summaries[(readout_mode, "non_calibrated", family_name)]
            cal_summary = calibrated_summaries[(readout_mode, "calibrated", family_name)]
            labels = jsd_summary.layer_labels
            x = list(range(len(labels)))

            jsd_mean = jsd_summary.mean.detach().cpu().numpy()
            jsd_low = jsd_summary.ci_low.detach().cpu().numpy()
            jsd_high = jsd_summary.ci_high.detach().cpu().numpy()
            cal_mean = cal_summary.mean.detach().cpu().numpy()
            cal_low = cal_summary.ci_low.detach().cpu().numpy()
            cal_high = cal_summary.ci_high.detach().cpu().numpy()

            lens_label = "Raw" if readout_mode == "raw" else "ModelNorm"

            jsd_line = ax.plot(
                x,
                jsd_mean,
                color="#111111",
                linewidth=2.8,
                label=f"{lens_label} | JSD",
            )[0]
            ax.fill_between(x, jsd_low, jsd_high, color="#111111", alpha=0.14, linewidth=0)

            cal_line = cal_ax.plot(
                x,
                cal_mean,
                color="#1f77b4",
                linewidth=2.8,
                linestyle="--",
                label=f"{lens_label} | {compact_metric}",
            )[0]
            cal_ax.fill_between(x, cal_low, cal_high, color="#1f77b4", alpha=0.14, linewidth=0)

            family_label_short = family_titles[family_name]
            family_label_short = family_label_short.replace("trained vs random-seed mean", "trained vs rand-mean")
            family_label_short = family_label_short.replace("random-seed-pair mean", "rand-pair mean")
            ax.set_title(
                f"{family_label_short} | {lens_label}",
                fontsize=35,
                fontweight="semibold",
            )
            ax.set_xticks(x)
            if row_idx < len(family_order) - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xticklabels(labels, rotation=0, ha="center")
            ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Mean JSD", fontweight="semibold", fontsize=32, color="#111111")
            else:
                ax.set_ylabel("")
            if col_idx == 1 and not use_single_metric_axis:
                cal_ax.set_ylabel(f"Mean {compact_metric}", fontweight="semibold", fontsize=32, color="#1f77b4")
            else:
                cal_ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=29)
            if not use_single_metric_axis:
                cal_ax.tick_params(axis="y", labelsize=29, colors="#1f77b4")
                ax.set_ylim(*jsd_ylim)
                cal_ax.set_ylim(*cal_ylim)
                if col_idx == 0:
                    cal_ax.tick_params(axis="y", right=False, labelright=False)
                else:
                    cal_ax.yaxis.set_label_position("right")
                    cal_ax.yaxis.tick_right()
            else:
                ax.set_ylim(*shared_metric_ylim)
            ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
            ax.spines["top"].set_visible(False)
            if not use_single_metric_axis:
                cal_ax.spines["top"].set_visible(False)

            if row_idx == 0 and col_idx == 0:
                legend_handles, legend_labels = _legend_row_grouped_items([jsd_line, cal_line], ["JSD", compact_metric], ncol=2)
                legend = fig.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper center",
                    ncol=2,
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.955),
                    fontsize=_legend_fontsize_for_labels(["JSD", compact_metric], base=32, min_size=29, max_size=36),
                )
    if legend is not None:
        fig.tight_layout(rect=(0.03, 0.06, 0.985, 0.93), w_pad=0.16, h_pad=0.9)
    else:
        fig.tight_layout(rect=(0.03, 0.06, 0.985, 0.93), w_pad=0.16, h_pad=0.9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prompt_four_view_summary_pdf(
    jsd_summaries: dict[tuple[str, str, str], FamilySummary],
    calibrated_summaries: dict[tuple[str, str, str], FamilySummary],
    path: str | Path,
    *,
    calibrated_label: str = "Top-1 ECE",
    family_labels: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    family_order = ["trained_vs_trained", "trained_vs_random", "random_vs_random"]
    style_map = {
        "trained_vs_trained": dict(color="#111111", label=(family_labels or {}).get("trained_vs_trained", "trained vs baseline")),
        "trained_vs_random": dict(color="#1f77b4", label=(family_labels or {}).get("trained_vs_random", "trained vs rand-mean")),
        "random_vs_random": dict(color="#d62728", label=(family_labels or {}).get("random_vs_random", "rand-pair mean")),
    }

    title_fonts = _pairwise_header_font_pack(rows=2, cols=2)
    fig, axes = plt.subplots(2, 2, figsize=(44.0, 18.8), sharex="col", sharey="row")
    fig.suptitle(
        title or f"Raw & ModelNorm, Non-Calibrated & Calibrated LogitDiff ({calibrated_label})",
        fontsize=title_fonts["suptitle"],
        fontweight="semibold",
        y=0.985,
    )

    panels = [
        (axes[0, 0], "raw", "non_calibrated", "Non-calibrated Raw"),
        (axes[0, 1], "model_norm", "non_calibrated", "Non-calibrated ModelNorm"),
        (axes[1, 0], "raw", "calibrated", f"Calibrated Raw ({calibrated_label})"),
        (axes[1, 1], "model_norm", "calibrated", f"Calibrated ModelNorm ({calibrated_label})"),
    ]

    for panel_idx, (ax, readout_mode, metric_mode, panel_title) in enumerate(panels):
        row_idx = panel_idx // 2
        col_idx = panel_idx % 2
        source = jsd_summaries if metric_mode == "non_calibrated" else calibrated_summaries
        labels = source[(readout_mode, metric_mode, family_order[0])].layer_labels
        x = list(range(len(labels)))
        for family_name in family_order:
            summary = source[(readout_mode, metric_mode, family_name)]
            style = style_map[family_name]
            mean = summary.mean.detach().cpu().numpy()
            ci_low = summary.ci_low.detach().cpu().numpy()
            ci_high = summary.ci_high.detach().cpu().numpy()
            ax.plot(x, mean, color=style["color"], linewidth=2.6, label=style["label"])
            ax.fill_between(x, ci_low, ci_high, color=style["color"], alpha=0.14, linewidth=0)
        ax.set_title(panel_title, fontsize=title_fonts["title"], fontweight="semibold")
        ax.set_xticks(x)
        if metric_mode == "non_calibrated":
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticklabels(labels, rotation=0, ha="center")
        if metric_mode == "calibrated":
            ax.set_xlabel("")
        else:
            ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel(
                "Mean JSD" if metric_mode == "non_calibrated" else f"Mean {_compact_metric_label(calibrated_label)}",
                fontweight="semibold",
                fontsize=title_fonts["axis"],
            )
        else:
            ax.set_ylabel("")
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=title_fonts["ticks"])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_ncol = _legend_ncol_max_two_rows(labels, max_columns=4, prefer_one_row=True)
    legend_handles, legend_labels = _legend_row_grouped_items(handles, labels, ncol=legend_ncol)
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.952),
        fontsize=_legend_fontsize_for_labels(labels, base=32, min_size=29, max_size=36),
        handlelength=2.6,
        columnspacing=1.3,
        labelspacing=0.5,
    )
    _finalize_header_layout(
        fig,
        legend=legend,
        labels=labels,
        n_panel_rows=2,
        n_panel_cols=2,
        max_columns=max(1, legend_ncol),
        left=0.08,
        right=0.99,
        bottom=0.09,
        min_top=0.76,
        wspace=0.02,
        hspace=0.34,
        gap_below_legend=0.028,
    )
    fig.supxlabel("Layers", fontsize=title_fonts["axis"], fontweight="semibold", y=0.04)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prompt_jsd_calibration_gap_overlay_pdf(
    jsd_gap_summaries: dict[tuple[str, str, str], FamilySummary],
    calibrated_gap_summaries: dict[tuple[str, str, str], FamilySummary],
    path: str | Path,
    *,
    calibrated_label: str = "Top-1 ECE",
    family_titles_override: dict[str, str] | None = None,
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    family_order = [
        "trained_vs_trained_minus_random_vs_random",
        "trained_vs_random_minus_random_vs_random",
    ]
    family_titles = {
        "trained_vs_trained_minus_random_vs_random": (family_titles_override or {}).get(
            "trained_vs_trained_minus_random_vs_random",
            "trained-baseline",
        ),
        "trained_vs_random_minus_random_vs_random": (family_titles_override or {}).get(
            "trained_vs_random_minus_random_vs_random",
            "trained-rand",
        ),
    }

    compact_family_titles = {key: _short_gap_family_title(value) for key, value in family_titles.items()}
    compact_metric = _compact_metric_label(calibrated_label)
    use_single_gap_axis = _overlay_axis_mode(calibrated_label, gap=True) == "shared"

    jsd_lows: list[float] = []
    jsd_highs: list[float] = []
    cal_lows: list[float] = []
    cal_highs: list[float] = []
    for family_name in family_order:
        for readout_mode in ("raw", "model_norm"):
            jsd_summary = jsd_gap_summaries[(readout_mode, "non_calibrated", family_name)]
            cal_summary = calibrated_gap_summaries[(readout_mode, "calibrated", family_name)]
            jsd_lows.extend(jsd_summary.ci_low.detach().cpu().tolist())
            jsd_highs.extend(jsd_summary.ci_high.detach().cpu().tolist())
            cal_lows.extend(cal_summary.ci_low.detach().cpu().tolist())
            cal_highs.extend(cal_summary.ci_high.detach().cpu().tolist())

    def _axis_limits(low_values: list[float], high_values: list[float]) -> tuple[float, float]:
        lo = min(low_values)
        hi = max(high_values)
        span = max(hi - lo, 1e-8)
        pad = 0.06 * span
        return lo - pad, hi + pad

    jsd_ylim = _axis_limits(jsd_lows, jsd_highs)
    cal_ylim = _axis_limits(cal_lows, cal_highs)
    shared_gap_ylim = _axis_limits(jsd_lows + cal_lows, jsd_highs + cal_highs) if use_single_gap_axis else None

    title_fonts = _pairwise_header_font_pack(rows=2, cols=2)
    fig, axes = plt.subplots(2, 2, figsize=(42.0, 15.5), sharex="col", sharey="row")
    fig.suptitle(title or f"Raw & ModelNorm: Non-Calibrated ΔJSD vs Calibrated Δ{calibrated_label}", fontsize=title_fonts["suptitle"], fontweight="semibold", y=0.985)

    legend = None

    for row_idx, family_name in enumerate(family_order):
        for col_idx, readout_mode in enumerate(("raw", "model_norm")):
            ax = axes[row_idx, col_idx]
            cal_ax = ax if use_single_gap_axis else ax.twinx()
            jsd_summary = jsd_gap_summaries[(readout_mode, "non_calibrated", family_name)]
            cal_summary = calibrated_gap_summaries[(readout_mode, "calibrated", family_name)]
            labels = jsd_summary.layer_labels
            x = list(range(len(labels)))
            lens_label = "Raw" if readout_mode == "raw" else "ModelNorm"

            jsd_mean = jsd_summary.mean.detach().cpu().numpy()
            jsd_low = jsd_summary.ci_low.detach().cpu().numpy()
            jsd_high = jsd_summary.ci_high.detach().cpu().numpy()
            cal_mean = cal_summary.mean.detach().cpu().numpy()
            cal_low = cal_summary.ci_low.detach().cpu().numpy()
            cal_high = cal_summary.ci_high.detach().cpu().numpy()

            jsd_line = ax.plot(
                x,
                jsd_mean,
                color="#111111",
                linewidth=2.8,
                label="ΔJSD",
            )[0]
            ax.fill_between(x, jsd_low, jsd_high, color="#111111", alpha=0.14, linewidth=0)
            ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.9)

            cal_line = cal_ax.plot(
                x,
                cal_mean,
                color="#1f77b4",
                linewidth=2.8,
                linestyle="--",
                label=f"Δ{compact_metric}",
            )[0]
            cal_ax.fill_between(x, cal_low, cal_high, color="#1f77b4", alpha=0.14, linewidth=0)
            if not use_single_gap_axis:
                cal_ax.axhline(0.0, color="#9ec3e6", linewidth=0.8, linestyle=":", alpha=0.8)

            family_label_short = compact_family_titles[family_name]
            family_label_short = family_label_short.replace("trained", "train")
            family_label_short = family_label_short.replace("baseline", "base")
            family_label_short = family_label_short.replace("random", "rand")
            ax.set_title(
                f"{family_label_short} | {lens_label}",
                fontsize=35,
                fontweight="semibold",
            )
            ax.set_xticks(x)
            if row_idx == 0:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xticklabels(labels, rotation=0, ha="center")
            ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("ΔJSD" if not use_single_gap_axis else f"ΔJSD & Δ{compact_metric}", fontweight="semibold", fontsize=32, color="#111111" if not use_single_gap_axis else "#222222")
            else:
                ax.set_ylabel("")
            if col_idx == 1 and not use_single_gap_axis:
                cal_ax.set_ylabel(f"Δ{compact_metric}", fontweight="semibold", fontsize=32, color="#1f77b4")
            else:
                cal_ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=29)
            if not use_single_gap_axis:
                cal_ax.tick_params(axis="y", labelsize=29, colors="#1f77b4")
                ax.set_ylim(*jsd_ylim)
                cal_ax.set_ylim(*cal_ylim)
                if col_idx == 0:
                    cal_ax.tick_params(axis="y", right=False, labelright=False)
                else:
                    cal_ax.yaxis.set_label_position("right")
                    cal_ax.yaxis.tick_right()
            else:
                assert shared_gap_ylim is not None
                ax.set_ylim(*shared_gap_ylim)
            ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
            ax.spines["top"].set_visible(False)
            if not use_single_gap_axis:
                cal_ax.spines["top"].set_visible(False)

            if row_idx == 0 and col_idx == 0:
                legend_handles, legend_labels = _legend_row_grouped_items([jsd_line, cal_line], ["ΔJSD", f"Δ{compact_metric}"], ncol=2)
                legend = fig.legend(
                    legend_handles,
                    legend_labels,
                    loc="upper center",
                    ncol=2,
                    frameon=False,
                    bbox_to_anchor=(0.5, 0.955),
                    fontsize=_legend_fontsize_for_labels(["ΔJSD", f"Δ{compact_metric}"], base=32, min_size=29, max_size=36),
                )
    if legend is not None:
        fig.tight_layout(rect=(0.03, 0.06, 0.985, 0.93), w_pad=0.16, h_pad=0.9)
    else:
        fig.tight_layout(rect=(0.03, 0.06, 0.985, 0.93), w_pad=0.16, h_pad=0.9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_prompt_null_sanity_summary_csv(
    summaries: dict[tuple[str, str, str], FamilySummary],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "readout_mode,metric_mode,family,layer_index,layer_label,mean,ci_low,ci_high,num_prompts"
    ]
    for (readout_mode, metric_mode, family_name), summary in summaries.items():
        mean = summary.mean.detach().cpu().tolist()
        ci_low = summary.ci_low.detach().cpu().tolist()
        ci_high = summary.ci_high.detach().cpu().tolist()
        for layer_idx, layer_label in enumerate(summary.layer_labels):
            lines.append(
                ",".join(
                    [
                        readout_mode,
                        metric_mode,
                        family_name,
                        str(layer_idx),
                        layer_label,
                        f"{float(mean[layer_idx]):.10f}",
                        f"{float(ci_low[layer_idx]):.10f}",
                        f"{float(ci_high[layer_idx]):.10f}",
                        str(int(summary.per_prompt_curves.shape[0])),
                    ]
                )
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


__all__ = [
    "build_prompt_pairwise_metric_curves",
    "build_prompt_pairwise_metric_gap_curves",
    "build_prompt_pairwise_jsd_gap_curves",
    "build_prompt_pairwise_jsd_curves",
    "build_prompt_hidden_metric_summaries",
    "build_prompt_null_gap_summaries",
    "build_prompt_null_sanity_summaries",
    "build_prompt_similarity_summaries",
    "plot_prompt_null_sanity",
    "save_combined_pairwise_metric_pdf",
    "save_combined_pairwise_metric_gap_pdf",
    "save_robust_prompt_jsd_overview_pdf",
    "save_combined_pairwise_jsd_pdf",
    "save_combined_pairwise_jsd_gap_pdf",
    "save_prompt_pairwise_jsd_pdf",
    "save_prompt_metric_score_and_gap_pdf",
    "save_prompt_metric_actual_summary_pdf",
    "save_prompt_metric_bundle_actual_pdf",
    "save_prompt_metric_bundle_gap_pdf",
    "save_prompt_jsd_calibration_overlay_pdf",
    "save_prompt_jsd_calibration_gap_overlay_pdf",
    "save_prompt_four_view_summary_pdf",
    "save_prompt_null_sanity_pdf",
    "save_prompt_null_sanity_figure",
    "save_prompt_null_sanity_summary_csv",
]

from __future__ import annotations

from pathlib import Path
from typing import Literal

import plotly.graph_objects as go
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from tuned_lens.plotting import PredictionTrajectory, TrajectoryStatistic

from ..lenses.tuned_lens_adapters import ReadoutKind, build_tuned_lens_adapter
from .plotly_export import save_plotly_figure


TrajectoryStatName = Literal[
    "cross_entropy",
    "rank",
    "entropy",
    "forward_kl",
    "max_probability",
    "js_divergence",
    "kl_divergence",
    "total_variation",
]


def _resolve_model_device(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


def _build_shifted_targets(input_ids: list[int], *, pad_with_last: bool = True) -> list[int]:
    if len(input_ids) < 2:
        raise ValueError("Need at least two tokens to build shifted next-token targets.")
    tail = input_ids[-1] if pad_with_last else -100
    return [*input_ids[1:], tail]


def build_prediction_trajectory(
    *,
    model: PreTrainedModel,
    tokenizer,
    prompt: str,
    lens_kind: ReadoutKind,
    tuned_lens_resource: str | None = None,
    add_special_tokens: bool = False,
    with_shifted_targets: bool = False,
    mask_input: bool = False,
) -> PredictionTrajectory:
    encoded = tokenizer(
        prompt,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0].tolist()
    lens = build_tuned_lens_adapter(
        model=model,
        kind=lens_kind,
        tuned_lens_resource=tuned_lens_resource,
        map_location=_resolve_model_device(model),
    )
    targets = _build_shifted_targets(input_ids) if with_shifted_targets else None
    return PredictionTrajectory.from_lens_and_model(
        lens=lens,
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        targets=targets,
        mask_input=mask_input,
    )


def build_trajectory_statistic(
    trajectory: PredictionTrajectory,
    *,
    statistic: TrajectoryStatName,
    other: PredictionTrajectory | None = None,
    label_topk: int = 5,
    min_prob: float = 0.0,
    min_prob_delta: float = 0.0,
) -> TrajectoryStatistic:
    if statistic == "cross_entropy":
        return trajectory.cross_entropy(topk=label_topk, min_prob=min_prob)
    if statistic == "rank":
        return trajectory.rank(topk=label_topk, min_prob=min_prob)
    if statistic == "entropy":
        return trajectory.entropy(topk=label_topk, min_prob=min_prob)
    if statistic == "forward_kl":
        return trajectory.forward_kl(topk=label_topk, min_prob=min_prob)
    if statistic == "max_probability":
        return trajectory.max_probability(topk=label_topk, min_prob=min_prob)

    if other is None:
        raise ValueError(f"Statistic {statistic!r} requires a comparison trajectory.")

    if statistic == "js_divergence":
        return trajectory.js_divergence(
            other,
            topk=label_topk,
            min_prob_delta=min_prob_delta,
        )
    if statistic == "kl_divergence":
        return trajectory.kl_divergence(
            other,
            topk=label_topk,
            min_prob_delta=min_prob_delta,
        )
    if statistic == "total_variation":
        return trajectory.total_variation(
            other,
            topk=label_topk,
            min_prob_delta=min_prob_delta,
        )
    raise ValueError(f"Unsupported statistic: {statistic!r}")


def build_trajectory_figure(
    *,
    model: PreTrainedModel,
    tokenizer,
    prompt: str,
    lens_kind: ReadoutKind,
    statistic: TrajectoryStatName = "forward_kl",
    tuned_lens_resource: str | None = None,
    compare_to_kind: ReadoutKind | None = None,
    compare_to_tuned_lens_resource: str | None = None,
    add_special_tokens: bool = False,
    with_shifted_targets: bool = False,
    mask_input: bool = False,
    label_topk: int = 5,
    min_prob: float = 0.0,
    min_prob_delta: float = 0.0,
    stride: int = 1,
    token_width: int = 90,
    title: str | None = None,
) -> go.Figure:
    trajectory = build_prediction_trajectory(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        lens_kind=lens_kind,
        tuned_lens_resource=tuned_lens_resource,
        add_special_tokens=add_special_tokens,
        with_shifted_targets=with_shifted_targets,
        mask_input=mask_input,
    )

    other = None
    if compare_to_kind is not None:
        other = build_prediction_trajectory(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            lens_kind=compare_to_kind,
            tuned_lens_resource=compare_to_tuned_lens_resource,
            add_special_tokens=add_special_tokens,
            with_shifted_targets=with_shifted_targets,
            mask_input=mask_input,
        )

    stat = build_trajectory_statistic(
        trajectory,
        statistic=statistic,
        other=other,
        label_topk=label_topk,
        min_prob=min_prob,
        min_prob_delta=min_prob_delta,
    )
    if stride > 1:
        stat = stat.stride(stride)

    figure_title = title or f"{lens_kind} | {stat.name}"
    return stat.figure(title=figure_title, token_width=token_width)


def save_trajectory_figure(
    output_path: str | Path,
    *,
    model: PreTrainedModel,
    tokenizer,
    prompt: str,
    lens_kind: ReadoutKind,
    statistic: TrajectoryStatName = "forward_kl",
    tuned_lens_resource: str | None = None,
    compare_to_kind: ReadoutKind | None = None,
    compare_to_tuned_lens_resource: str | None = None,
    add_special_tokens: bool = False,
    with_shifted_targets: bool = False,
    mask_input: bool = False,
    label_topk: int = 5,
    min_prob: float = 0.0,
    min_prob_delta: float = 0.0,
    stride: int = 1,
    token_width: int = 90,
    title: str | None = None,
    format: str | None = None,
) -> Path:
    fig = build_trajectory_figure(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        lens_kind=lens_kind,
        statistic=statistic,
        tuned_lens_resource=tuned_lens_resource,
        compare_to_kind=compare_to_kind,
        compare_to_tuned_lens_resource=compare_to_tuned_lens_resource,
        add_special_tokens=add_special_tokens,
        with_shifted_targets=with_shifted_targets,
        mask_input=mask_input,
        label_topk=label_topk,
        min_prob=min_prob,
        min_prob_delta=min_prob_delta,
        stride=stride,
        token_width=token_width,
        title=title,
    )
    return save_plotly_figure(fig, output_path, format=format)


def load_model_and_tokenizer(
    *,
    model_name: str,
    dtype: torch.dtype,
    device: str | torch.device | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, tokenizer


__all__ = [
    "TrajectoryStatName",
    "build_prediction_trajectory",
    "build_trajectory_figure",
    "build_trajectory_statistic",
    "load_model_and_tokenizer",
    "save_trajectory_figure",
]

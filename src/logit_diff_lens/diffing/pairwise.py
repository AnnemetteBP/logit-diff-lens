from __future__ import annotations

from typing import Any

import torch

from .distribution_metrics import (
    jaccard_topk_divergence,
    js_divergence,
    reference_token_probability_shift,
    reference_token_rank_in_q,
    top1_agreement,
    total_variation_distance,
)
from .hidden_metrics import cosine_distance, l2_distance, norm_ratio, normalized_l2_distance


def compute_pairwise_hidden_metrics(hidden_a: torch.Tensor, hidden_b: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "cosine_distance": cosine_distance(hidden_a, hidden_b),
        "l2_distance": l2_distance(hidden_a, hidden_b),
        "normalized_l2_distance": normalized_l2_distance(hidden_a, hidden_b),
        "norm_ratio": norm_ratio(hidden_a, hidden_b),
    }


def compute_pairwise_distribution_metrics(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    *,
    topk: int = 10,
    reference_token_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    metrics = {
        "js_divergence": js_divergence(probs_a, probs_b),
        "total_variation_distance": total_variation_distance(probs_a, probs_b),
        "jaccard_topk_divergence": jaccard_topk_divergence(probs_a, probs_b, k=topk),
        "top1_agreement": top1_agreement(probs_a, probs_b),
    }
    if reference_token_ids is not None:
        metrics["reference_token_probability_shift"] = reference_token_probability_shift(
            probs_a,
            probs_b,
            reference_token_ids=reference_token_ids,
        )
        metrics["reference_token_rank_in_b"] = reference_token_rank_in_q(
            probs_b,
            reference_token_ids=reference_token_ids,
        )
    return metrics


def compute_prompt_pairwise_analysis(
    *,
    hidden_a: torch.Tensor,
    hidden_b: torch.Tensor,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    topk: int = 10,
    reference_token_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    probs_a = torch.softmax(logits_a.to(dtype=torch.float32), dim=-1)
    probs_b = torch.softmax(logits_b.to(dtype=torch.float32), dim=-1)
    return {
        "hidden_metrics": compute_pairwise_hidden_metrics(hidden_a, hidden_b),
        "distribution_metrics": compute_pairwise_distribution_metrics(
            probs_a,
            probs_b,
            topk=topk,
            reference_token_ids=reference_token_ids,
        ),
    }


__all__ = [
    "compute_pairwise_hidden_metrics",
    "compute_pairwise_distribution_metrics",
    "compute_prompt_pairwise_analysis",
]

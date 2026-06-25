from __future__ import annotations

import torch


def single_token_logit_difference(logits_a: torch.Tensor, logits_b: torch.Tensor, *, token_id: int) -> torch.Tensor:
    return logits_b[..., token_id] - logits_a[..., token_id]


def margin_logit_difference(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    positive_token_id: int,
    negative_token_id: int,
) -> torch.Tensor:
    margin_a = logits_a[..., positive_token_id] - logits_a[..., negative_token_id]
    margin_b = logits_b[..., positive_token_id] - logits_b[..., negative_token_id]
    return margin_b - margin_a


def token_set_logsumexp(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    subset = logits.index_select(dim=-1, index=token_ids.to(device=logits.device, dtype=torch.long))
    return torch.logsumexp(subset, dim=-1)


def token_set_contrast_difference(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    positive_token_ids: torch.Tensor,
    negative_token_ids: torch.Tensor,
) -> torch.Tensor:
    positive_a = token_set_logsumexp(logits_a, positive_token_ids)
    negative_a = token_set_logsumexp(logits_a, negative_token_ids)
    positive_b = token_set_logsumexp(logits_b, positive_token_ids)
    negative_b = token_set_logsumexp(logits_b, negative_token_ids)
    return (positive_b - negative_b) - (positive_a - negative_a)


def reconstruction_error(target_delta: torch.Tensor, component_deltas: torch.Tensor) -> torch.Tensor:
    return torch.abs(target_delta - component_deltas.sum(dim=0))


__all__ = [
    "margin_logit_difference",
    "reconstruction_error",
    "single_token_logit_difference",
    "token_set_contrast_difference",
    "token_set_logsumexp",
]

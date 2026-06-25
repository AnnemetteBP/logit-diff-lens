from __future__ import annotations

import torch


def _normalize_probabilities(probs: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    probs = probs.to(dtype=torch.float32).clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    p = _normalize_probabilities(p, eps=eps)
    q = _normalize_probabilities(q, eps=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def symmetric_kl_divergence(p: torch.Tensor, q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return 0.5 * (kl_divergence(p, q, eps=eps) + kl_divergence(q, p, eps=eps))


def js_divergence(p: torch.Tensor, q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    p = _normalize_probabilities(p, eps=eps)
    q = _normalize_probabilities(q, eps=eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)


def js_divergence_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    return js_divergence(torch.softmax(p_logits, dim=-1), torch.softmax(q_logits, dim=-1))


def total_variation_distance(p: torch.Tensor, q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    p = _normalize_probabilities(p, eps=eps)
    q = _normalize_probabilities(q, eps=eps)
    return 0.5 * torch.abs(p - q).sum(dim=-1)


def top1_agreement(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return p.argmax(dim=-1).eq(q.argmax(dim=-1))


def jaccard_topk(p: torch.Tensor, q: torch.Tensor, *, k: int = 10) -> torch.Tensor:
    k = min(int(k), int(p.shape[-1]), int(q.shape[-1]))
    if k <= 0:
        raise ValueError("top-k must be positive after clamping to vocabulary size")
    p_topk = torch.topk(p, k=k, dim=-1).indices.reshape(-1, k)
    q_topk = torch.topk(q, k=k, dim=-1).indices.reshape(-1, k)
    scores = []
    for p_row, q_row in zip(p_topk, q_topk):
        p_set = set(p_row.tolist())
        q_set = set(q_row.tolist())
        union = p_set | q_set
        inter = p_set & q_set
        scores.append(0.0 if not union else len(inter) / len(union))
    return torch.tensor(scores, dtype=torch.float32).reshape(p.shape[:-1])


def jaccard_topk_divergence(p: torch.Tensor, q: torch.Tensor, *, k: int = 10) -> torch.Tensor:
    return 1.0 - jaccard_topk(p, q, k=k)


def reference_token_probability_shift(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    reference_token_ids: torch.Tensor,
) -> torch.Tensor:
    p = _normalize_probabilities(p)
    q = _normalize_probabilities(q)
    gather_index = reference_token_ids.to(dtype=torch.long).unsqueeze(-1)
    p_ref = torch.gather(p, dim=-1, index=gather_index).squeeze(-1)
    q_ref = torch.gather(q, dim=-1, index=gather_index).squeeze(-1)
    return q_ref - p_ref


def reference_token_rank_in_q(q: torch.Tensor, *, reference_token_ids: torch.Tensor) -> torch.Tensor:
    sorted_indices = torch.argsort(q, dim=-1, descending=True)
    target = reference_token_ids.to(dtype=torch.long).unsqueeze(-1)
    matches = sorted_indices.eq(target)
    return matches.to(dtype=torch.int64).argmax(dim=-1)


__all__ = [
    "jaccard_topk",
    "jaccard_topk_divergence",
    "js_divergence",
    "js_divergence_from_logits",
    "kl_divergence",
    "reference_token_probability_shift",
    "reference_token_rank_in_q",
    "symmetric_kl_divergence",
    "top1_agreement",
    "total_variation_distance",
]

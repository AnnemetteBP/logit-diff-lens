from __future__ import annotations

import torch


def _normalize_probabilities(probs: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    probs = probs.to(dtype=torch.float32)
    probs = probs.clamp_min(eps)
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
    p_topk = torch.topk(p, k=k, dim=-1).indices
    q_topk = torch.topk(q, k=k, dim=-1).indices

    flat_p = p_topk.reshape(-1, k)
    flat_q = q_topk.reshape(-1, k)
    scores = []
    for p_row, q_row in zip(flat_p, flat_q):
        p_set = set(p_row.tolist())
        q_set = set(q_row.tolist())
        union = p_set | q_set
        inter = p_set & q_set
        scores.append(0.0 if not union else len(inter) / len(union))
    return torch.tensor(scores, dtype=torch.float32).reshape(p_topk.shape[:-1])


__all__ = [
    "jaccard_topk",
    "js_divergence",
    "js_divergence_from_logits",
    "kl_divergence",
    "symmetric_kl_divergence",
    "top1_agreement",
    "total_variation_distance",
]

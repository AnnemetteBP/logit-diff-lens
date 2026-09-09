from __future__ import annotations

import math

import torch


def _validate_finite_scalar(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} produced a non-finite value: {value!r}")
    return float(value)


def _validate_unit_interval(name: str, value: float, *, atol: float = 1e-6) -> float:
    value = _validate_finite_scalar(name, value)
    if value < -atol or value > 1.0 + atol:
        raise ValueError(f"{name} must lie in [0, 1], observed {value:.8f}")
    return float(min(1.0, max(0.0, value)))


def _to_float_matrix(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={tuple(x.shape)}")
    return x.detach().to(dtype=torch.float32, device="cpu")


def linear_cka(X: torch.Tensor, Y: torch.Tensor, *, eps: float = 1e-12) -> float:
    X = _to_float_matrix(X)
    Y = _to_float_matrix(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("linear_cka requires matching sample counts")

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    xty = X.T @ Y
    numerator = torch.sum(xty * xty)
    xx = X.T @ X
    yy = Y.T @ Y
    denominator = torch.sqrt(torch.sum(xx * xx) * torch.sum(yy * yy)).clamp_min(eps)
    score = (numerator / denominator).item()
    return _validate_unit_interval("linear_cka", score)


def _rowwise_js_divergence(P: torch.Tensor, Q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    P = P.clamp_min(eps)
    Q = Q.clamp_min(eps)
    P = P / P.sum(dim=-1, keepdim=True).clamp_min(eps)
    Q = Q / Q.sum(dim=-1, keepdim=True).clamp_min(eps)
    M = 0.5 * (P + Q)
    kl_pm = torch.sum(P * (torch.log(P) - torch.log(M)), dim=-1)
    kl_qm = torch.sum(Q * (torch.log(Q) - torch.log(M)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def js_similarity_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, *, eps: float = 1e-12) -> float:
    logits_a = _to_float_matrix(logits_a)
    logits_b = _to_float_matrix(logits_b)
    if logits_a.shape != logits_b.shape:
        raise ValueError("js_similarity_from_logits requires matching shapes")
    probs_a = torch.softmax(logits_a, dim=-1)
    probs_b = torch.softmax(logits_b, dim=-1)
    js = _rowwise_js_divergence(probs_a, probs_b, eps=eps)
    if not torch.isfinite(js).all():
        raise ValueError("js_similarity_from_logits produced non-finite rowwise JSD values")
    min_js = float(js.min().item()) if js.numel() else 0.0
    max_js = float(js.max().item()) if js.numel() else 0.0
    if min_js < -1e-6:
        raise ValueError(f"js_similarity_from_logits observed negative rowwise JSD: {min_js:.8f}")
    if max_js > math.log(2.0) + 1e-6:
        raise ValueError(
            f"js_similarity_from_logits observed rowwise JSD above log(2): {max_js:.8f}"
        )
    js = js.clamp_min(0.0).clamp_max(math.log(2.0))
    js_bits = js / math.log(2.0)
    similarity = 1.0 - js_bits.mean().item()
    return _validate_unit_interval("js_similarity_from_logits", similarity)


def topk_overlap_similarity(logits_a: torch.Tensor, logits_b: torch.Tensor, *, k: int = 20) -> float:
    logits_a = _to_float_matrix(logits_a)
    logits_b = _to_float_matrix(logits_b)
    if logits_a.shape != logits_b.shape:
        raise ValueError("topk_overlap_similarity requires matching shapes")
    if k <= 0:
        raise ValueError("k must be positive")
    vocab = logits_a.shape[-1]
    k_eff = min(k, vocab)
    idx_a = torch.topk(logits_a, k=k_eff, dim=-1).indices
    idx_b = torch.topk(logits_b, k=k_eff, dim=-1).indices
    scores: list[float] = []
    for row_a, row_b in zip(idx_a.tolist(), idx_b.tolist()):
        set_a = set(int(v) for v in row_a)
        set_b = set(int(v) for v in row_b)
        union = set_a | set_b
        inter = set_a & set_b
        scores.append(0.0 if not union else len(inter) / len(union))
    similarity = float(sum(scores) / len(scores)) if scores else 0.0
    return _validate_unit_interval("topk_overlap_similarity", similarity)


__all__ = [
    "js_similarity_from_logits",
    "linear_cka",
    "topk_overlap_similarity",
]

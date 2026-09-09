from __future__ import annotations

import math

import torch

from logit_diff_lens.diffing.distribution_metrics import (
    jaccard_topk_divergence,
    js_divergence,
    js_divergence_from_logits,
    kl_divergence,
)
from logit_diff_lens.similarity.metrics import js_similarity_from_logits


def test_kl_divergence_is_nonnegative() -> None:
    p = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float32)
    q = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]], dtype=torch.float32)
    values = kl_divergence(p, q)
    assert torch.all(values >= 0.0)


def test_js_divergence_is_bounded_by_log2() -> None:
    p = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float32)
    q = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]], dtype=torch.float32)
    values = js_divergence(p, q)
    assert torch.all(values >= 0.0)
    assert torch.all(values <= math.log(2.0))


def test_js_divergence_from_logits_is_bounded_by_log2() -> None:
    logits_a = torch.tensor([[3.0, 1.0, -2.0], [1.0, 5.0, -1.0]], dtype=torch.float32)
    logits_b = torch.tensor([[2.5, 1.5, -2.0], [0.0, 4.0, 0.0]], dtype=torch.float32)
    values = js_divergence_from_logits(logits_a, logits_b)
    assert torch.all(values >= 0.0)
    assert torch.all(values <= math.log(2.0))


def test_jaccard_topk_divergence_is_bounded() -> None:
    p = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float32)
    q = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]], dtype=torch.float32)
    values = jaccard_topk_divergence(p, q, k=2)
    assert torch.all(values >= 0.0)
    assert torch.all(values <= 1.0)


def test_js_similarity_from_logits_is_bounded_and_finite() -> None:
    logits_a = torch.tensor([[3.0, 1.0, -2.0], [1.0, 5.0, -1.0]], dtype=torch.float32)
    logits_b = torch.tensor([[2.5, 1.5, -2.0], [0.0, 4.0, 0.0]], dtype=torch.float32)
    score = js_similarity_from_logits(logits_a, logits_b)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0

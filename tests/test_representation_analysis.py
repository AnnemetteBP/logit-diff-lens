from __future__ import annotations

import random

import pytest
import torch

from diffing.logit_lens_methods.logitdiff_ldl.representation_analysis import (
    RepresentationAnalysisExample,
    analyze_representation_alignment,
    check_base_equals_ft_cosine_sanity,
    check_probe_variation_sanity,
    check_random_label_probe_sanity,
)


def _make_hidden_state(last_token_vec: torch.Tensor) -> torch.Tensor:
    hidden_dim = last_token_vec.numel()
    hidden = torch.zeros((1, 3, hidden_dim), dtype=torch.float32)
    hidden[0, -1, :] = last_token_vec
    return hidden


def _make_example(
    label: bool,
    *,
    base_vectors: list[torch.Tensor],
    ft_vectors: list[torch.Tensor],
) -> RepresentationAnalysisExample:
    return RepresentationAnalysisExample(
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        hidden_states_base=[_make_hidden_state(vec) for vec in base_vectors],
        hidden_states_ft=[_make_hidden_state(vec) for vec in ft_vectors],
        label_harmful=label,
    )


def test_base_equals_ft_cosine_is_one() -> None:
    examples = []
    for idx in range(8):
        vectors = [
            torch.tensor([float(idx + 1), 1.0, 0.5], dtype=torch.float32),
            torch.tensor([0.2, float(idx + 2), 0.3], dtype=torch.float32),
            torch.tensor([0.4, 0.5, float(idx + 3)], dtype=torch.float32),
        ]
        examples.append(_make_example(idx % 2 == 0, base_vectors=vectors, ft_vectors=vectors))

    result = analyze_representation_alignment(examples, include_cka=True, random_seed=0)
    check_base_equals_ft_cosine_sanity(result["cosine_per_layer"])


def test_probe_accuracy_varies_across_layers() -> None:
    torch.manual_seed(0)
    examples = []
    for idx in range(20):
        label = idx % 2 == 0
        sign = 1.0 if label else -1.0
        base_vectors = [
            torch.tensor([0.0, float(idx % 3), 0.0], dtype=torch.float32),
            torch.tensor([sign * 0.4 + 0.15 * torch.randn(()).item(), 0.0, 0.0], dtype=torch.float32),
            torch.tensor([sign * 3.0 + 0.05 * torch.randn(()).item(), 0.0, 0.0], dtype=torch.float32),
        ]
        ft_vectors = [vec.clone() for vec in base_vectors]
        examples.append(_make_example(label, base_vectors=base_vectors, ft_vectors=ft_vectors))

    result = analyze_representation_alignment(
        examples,
        representation_source="finetuned",
        include_cka=False,
        random_seed=0,
    )
    check_probe_variation_sanity(result["probe_results"], min_range=0.05)


def test_random_labels_probe_auroc_is_near_chance() -> None:
    rng = random.Random(0)
    torch.manual_seed(0)
    examples = []
    for idx in range(200):
        label = bool(rng.getrandbits(1))
        base_vectors = [
            torch.randn(8, dtype=torch.float32),
            torch.randn(8, dtype=torch.float32),
            torch.randn(8, dtype=torch.float32),
        ]
        ft_vectors = [vec.clone() for vec in base_vectors]
        examples.append(_make_example(label, base_vectors=base_vectors, ft_vectors=ft_vectors))

    result = analyze_representation_alignment(
        examples,
        representation_source="finetuned",
        include_cka=False,
        random_seed=0,
    )
    check_random_label_probe_sanity(result["probe_results"], tolerance=0.2)


def test_probe_variation_sanity_fails_for_constant_accuracy() -> None:
    with pytest.raises(ValueError, match="constant across layers"):
        check_probe_variation_sanity(
            [
                {"accuracy": 0.5, "auroc": 0.5},
                {"accuracy": 0.5, "auroc": 0.6},
            ]
        )

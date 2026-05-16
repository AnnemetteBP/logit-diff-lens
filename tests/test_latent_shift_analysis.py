from __future__ import annotations

import torch

from diffing.logit_lens_methods.logitdiff_ldl.latent_shift_analysis import (
    LatentShiftAnalysisExample,
    analyze_latent_shift_structure,
)


def _make_hidden_state(last_token_vec: torch.Tensor) -> torch.Tensor:
    hidden_dim = last_token_vec.numel()
    hidden = torch.zeros((1, 3, hidden_dim), dtype=torch.float32)
    hidden[0, -1, :] = last_token_vec
    return hidden


def _make_logits(last_token_logits: torch.Tensor) -> torch.Tensor:
    vocab_size = last_token_logits.numel()
    logits = torch.zeros((1, 3, vocab_size), dtype=torch.float32)
    logits[0, -1, :] = last_token_logits
    return logits


def _make_example(
    *,
    label: bool,
    base_hidden_layers: list[torch.Tensor],
    ft_hidden_layers: list[torch.Tensor],
    base_logit_layers: list[torch.Tensor],
    ft_logit_layers: list[torch.Tensor],
) -> LatentShiftAnalysisExample:
    return LatentShiftAnalysisExample(
        hidden_states_base=[_make_hidden_state(vec) for vec in base_hidden_layers],
        hidden_states_ft=[_make_hidden_state(vec) for vec in ft_hidden_layers],
        logits_base=[_make_logits(vec) for vec in base_logit_layers],
        logits_ft=[_make_logits(vec) for vec in ft_logit_layers],
        label_harmful=label,
    )


def test_base_equals_ft_yields_zero_shift_and_chance_shift_aurocs() -> None:
    examples = []
    for idx in range(10):
        label = idx % 2 == 0
        base_hidden_layers = [
            torch.tensor([float(idx + 1), 0.5, 0.1], dtype=torch.float32),
            torch.tensor([0.2, float(idx + 2), 0.3], dtype=torch.float32),
        ]
        base_logit_layers = [
            torch.tensor([float(idx), 0.1, -0.2, 0.3], dtype=torch.float32),
            torch.tensor([0.5, float(idx) * 0.1, -0.4, 0.2], dtype=torch.float32),
        ]
        examples.append(
            _make_example(
                label=label,
                base_hidden_layers=base_hidden_layers,
                ft_hidden_layers=[vec.clone() for vec in base_hidden_layers],
                base_logit_layers=base_logit_layers,
                ft_logit_layers=[vec.clone() for vec in base_logit_layers],
            )
        )

    result = analyze_latent_shift_structure(examples, random_seed=0)

    for entry in result["delta_stats"]:
        assert max(abs(v) for v in entry["mean_delta"]) < 1e-6
        assert max(abs(v) for v in entry["l2_norm_per_example"]) < 1e-6
    for entry in result["logit_shift"]:
        assert max(abs(v) for v in entry["l2_norm_per_example"]) < 1e-6
    for entry in result["kl_divergence"]:
        assert max(abs(v) for v in entry["values"]) < 1e-6
    for entry in result["latent_to_logit_corr"]:
        assert abs(entry["pearson"]) < 1e-6
    for entry in result["latent_to_behavior"]:
        assert abs(entry["auroc"] - 0.5) < 1e-6
    for entry in result["logit_to_behavior"]:
        assert abs(entry["auroc"] - 0.5) < 1e-6
    for entry in result["subspace_shift"]:
        assert max(abs(v) for v in entry["principal_angles"]) < 1e-4


def test_shuffled_ft_deltas_have_near_zero_alignment() -> None:
    examples = []
    delta_basis = [
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
        torch.tensor([-1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, -1.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, -1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 0.0, -1.0], dtype=torch.float32),
    ]
    for idx in range(8):
        label = idx % 2 == 0
        base_vec = torch.tensor([float(idx + 1), float(label), 0.2, -0.1], dtype=torch.float32)
        delta = delta_basis[idx]
        examples.append(
            _make_example(
                label=label,
                base_hidden_layers=[base_vec],
                ft_hidden_layers=[base_vec + delta],
                base_logit_layers=[torch.tensor([0.1, 0.2, 0.3, float(idx)], dtype=torch.float32)],
                ft_logit_layers=[torch.tensor([0.1, 0.2, 0.3, float(idx)], dtype=torch.float32) + delta],
            )
        )

    result = analyze_latent_shift_structure(examples, random_seed=0)
    assert abs(result["delta_alignment"][0]["mean_pairwise_cosine"]) < 0.2


def test_latent_and_logit_scores_can_predict_behavior() -> None:
    examples = []
    for idx in range(20):
        label = idx < 10
        sign = 1.0 if label else -1.0
        magnitude = 3.0 if label else 0.3
        base_hidden = torch.tensor([sign * 4.0, 0.1 * idx, 0.0], dtype=torch.float32)
        ft_hidden = base_hidden + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        base_logits = torch.tensor([0.1, 0.0, -0.1], dtype=torch.float32)
        ft_logits = base_logits + torch.tensor([magnitude, 0.0, 0.0], dtype=torch.float32)
        examples.append(
            _make_example(
                label=label,
                base_hidden_layers=[base_hidden],
                ft_hidden_layers=[ft_hidden],
                base_logit_layers=[base_logits],
                ft_logit_layers=[ft_logits],
            )
        )

    result = analyze_latent_shift_structure(examples, random_seed=0)

    assert result["latent_to_behavior"][0]["auroc"] > 0.9
    assert result["logit_to_behavior"][0]["auroc"] > 0.9
    assert result["latent_to_logit_corr"][0]["pearson"] > 0.5
    assert result["probe_results"][0]["auroc"] > 0.9

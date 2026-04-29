from __future__ import annotations

from pathlib import Path

import torch

from diffing.logit_lens_methods.logitdiff_ldl.activation_dataset_analysis import (
    analyze_collected_activation_dataset,
    analyze_full_dataset_with_prism,
    analyze_paired_collected_activation_dataset,
    load_collected_activation_dataset,
    load_paired_collected_activation_dataset,
)


def _make_hidden(batch_seq_dim_values: list[list[float]]) -> torch.Tensor:
    return torch.tensor([batch_seq_dim_values], dtype=torch.float32)


def _make_logits(batch_seq_vocab_values: list[list[float]]) -> torch.Tensor:
    return torch.tensor([batch_seq_vocab_values], dtype=torch.float32)


def test_analyze_collected_activation_dataset_writes_all_plots(tmp_path: Path) -> None:
    dataset = []
    for idx in range(8):
        label = 1 if idx < 4 else 0
        sign = 1.0 if label else -1.0
        hidden_states = [
            _make_hidden(
                [
                    [0.1, 0.0, 0.0],
                    [0.2, 0.1, 0.0],
                    [sign * 2.0, 0.0, float(idx)],
                ]
            ),
            _make_hidden(
                [
                    [0.2, 0.1, 0.0],
                    [0.3, 0.2, 0.0],
                    [sign * 3.0, 0.5, float(idx)],
                ]
            ),
        ]
        logits = _make_logits(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.1, 0.4, 0.3],
                [sign * 1.5, 0.0, 0.1, float(idx) * 0.1],
            ]
        )
        dataset.append(
            {
                "hidden_states": hidden_states,
                "logits": logits,
                "label": label,
            }
        )

    dataset_path = tmp_path / "collected_activations.pt"
    torch.save(dataset, dataset_path)

    result = analyze_collected_activation_dataset(
        dataset_path,
        output_dir=tmp_path / "plots",
    )

    assert result["num_examples"] == 8
    assert result["num_layers"] == 2
    assert len(result["auroc_per_layer"]) == 2
    assert len(result["pca_top1_variance"]) == 2
    assert len(result["logit_norms"]) == 2
    assert result["logit_norm_mode"] == "final_repeated"
    assert result["pca_top1_variance"][0] > 0.0

    for plot_path in result["plot_paths"].values():
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_load_collected_activation_dataset_supports_pair_payload(tmp_path: Path) -> None:
    payload = {
        "lm_head_weight_base": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "lm_head_weight_ft": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "rows": [
            {
                "embedding_base": _make_hidden([[0.0, 0.0], [1.0, 0.0]]),
                "embedding_ft": _make_hidden([[0.0, 0.0], [2.0, 0.0]]),
                "hidden_states_base": [_make_hidden([[0.0, 0.0], [1.0, 0.0]])],
                "hidden_states_ft": [_make_hidden([[0.0, 0.0], [2.0, 0.0]])],
                "logits_base": _make_logits([[0.1, 0.2], [0.3, 0.4]]),
                "logits_ft": _make_logits([[0.2, 0.1], [0.4, 0.3]]),
                "label": 1,
            }
        ]
    }
    dataset_path = tmp_path / "paired_activations.pt"
    torch.save(payload, dataset_path)

    base_rows = load_collected_activation_dataset(dataset_path, model_variant="base")
    ft_rows = load_collected_activation_dataset(dataset_path, model_variant="ft")

    assert len(base_rows) == 1
    assert len(ft_rows) == 1
    assert torch.equal(base_rows[0]["hidden_states"][0], payload["rows"][0]["hidden_states_base"][0])
    assert torch.equal(ft_rows[0]["hidden_states"][0], payload["rows"][0]["hidden_states_ft"][0])


def test_analyze_paired_collected_activation_dataset_writes_all_delta_plots(tmp_path: Path) -> None:
    rows = []
    for idx in range(8):
        label = 1 if idx < 4 else 0
        sign = 1.0 if label else -1.0
        base_hidden_embed = _make_hidden([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [0.2, 0.0, 0.0]])
        base_hidden_block = _make_hidden([[0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [sign * 2.0, 0.0, float(idx)]])
        ft_hidden_block = _make_hidden([[0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [sign * 3.0, 0.5, float(idx)]])
        base_logits = _make_logits([[0.1, 0.2, 0.3], [0.2, 0.1, 0.3], [0.4, 0.1, 0.0]])
        ft_logits = _make_logits([[0.1, 0.2, 0.3], [0.2, 0.1, 0.3], [0.4 + sign, 0.1, 0.0]])
        attn_base = _make_hidden([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        attn_ft = _make_hidden([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2 + sign, 0.0, 0.0]])
        mlp_base = _make_hidden([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.2, 0.0]])
        mlp_ft = _make_hidden([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.2 + sign, 0.0]])
        rows.append(
            {
                "embedding_base": base_hidden_embed,
                "embedding_ft": base_hidden_embed.clone(),
                "hidden_states_base": [base_hidden_embed, base_hidden_block],
                "hidden_states_ft": [base_hidden_embed.clone(), ft_hidden_block],
                "logits_base": base_logits,
                "logits_ft": ft_logits,
                "attention_outputs_base": [attn_base],
                "attention_outputs_ft": [attn_ft],
                "mlp_outputs_base": [mlp_base],
                "mlp_outputs_ft": [mlp_ft],
                "label": label,
            }
        )

    dataset_path = tmp_path / "paired_activations.pt"
    torch.save(
        {
            "lm_head_weight_base": torch.eye(2, dtype=torch.float32),
            "lm_head_weight_ft": torch.eye(2, dtype=torch.float32),
            "rows": rows,
        },
        dataset_path,
    )

    loaded_rows = load_paired_collected_activation_dataset(dataset_path)
    assert len(loaded_rows) == 8

    result = analyze_paired_collected_activation_dataset(
        dataset_path,
        output_dir=tmp_path / "plots_paired",
    )

    assert result["num_examples"] == 8
    assert result["num_layers"] == 1
    assert len(result["latent_auroc"]) == 1
    assert len(result["logit_auroc"]) == 1
    assert len(result["pca_top1"]) == 1
    assert len(result["attn_diff"]) == 1
    assert len(result["mlp_diff"]) == 1

    for plot_path in result["plot_paths"].values():
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_analyze_full_dataset_with_prism_writes_requested_plots(tmp_path: Path) -> None:
    rows = []
    for idx in range(8):
        label = 1 if idx < 4 else 0
        sign = 1.0 if label else -1.0
        rows.append(
            {
                "embedding_base": _make_hidden([[0.0, 0.0], [0.1, 0.0]]),
                "embedding_ft": _make_hidden([[0.0, 0.0], [0.2, 0.0]]),
                "hidden_states_base": [
                    _make_hidden([[0.0, 0.0], [0.1, 0.0]]),
                    _make_hidden([[0.1, 0.0], [sign * 1.0, float(idx)]]),
                ],
                "hidden_states_ft": [
                    _make_hidden([[0.0, 0.0], [0.1, 0.0]]),
                    _make_hidden([[0.1, 0.0], [sign * 2.0, float(idx) + 0.5]]),
                ],
                "logits_base": _make_logits([[0.1, 0.2, 0.3], [0.4, 0.1, 0.0]]),
                "logits_ft": _make_logits([[0.1, 0.2, 0.3], [0.4 + sign, 0.1, 0.0]]),
                "attention_outputs_base": [_make_hidden([[0.0, 0.0], [0.2, 0.0]])],
                "attention_outputs_ft": [_make_hidden([[0.0, 0.0], [0.2 + sign, 0.0]])],
                "mlp_outputs_base": [_make_hidden([[0.0, 0.0], [0.0, 0.2]])],
                "mlp_outputs_ft": [_make_hidden([[0.0, 0.0], [0.0, 0.2 + sign]])],
                "label": label,
            }
        )

    dataset_path = tmp_path / "full_dataset_with_prism.pt"
    torch.save(
        {
            "lm_head_weight_base": torch.eye(2, dtype=torch.float32),
            "lm_head_weight_ft": torch.eye(2, dtype=torch.float32),
            "rows": rows,
        },
        dataset_path,
    )

    result = analyze_full_dataset_with_prism(
        dataset_path,
        output_dir=tmp_path / "manual_style_plots",
    )

    assert result["num_examples"] == 8
    assert result["num_layers"] == 1
    assert len(result["logit_diff"]) == 1
    assert len(result["AUROC"]) == 1
    assert len(result["logit_AUROC"]) == 1
    assert len(result["pca_top1"]) == 1
    assert len(result["attn_diff"]) == 1
    assert len(result["mlp_diff"]) == 1

    for plot_path in result["plots"].values():
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0

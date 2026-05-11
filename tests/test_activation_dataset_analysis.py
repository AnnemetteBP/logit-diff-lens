from __future__ import annotations

from pathlib import Path
import json

import torch

from diffing.logit_lens_methods.logitdiff_ldl.activation_dataset_analysis import (
    align_model_activation_datasets,
    align_model_activation_datasets_by_group,
    analyze_collected_activation_dataset,
    analyze_full_dataset_with_prism,
    analyze_paired_activation_svd,
    analyze_paired_collected_activation_dataset,
    load_collected_activation_dataset,
    load_paired_collected_activation_dataset,
    summarize_paired_teacher_forced_divergence,
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


def test_align_model_activation_datasets_merges_separate_model_files(tmp_path: Path) -> None:
    base_payload = {
        "model_key": "base",
        "lm_head_weight": torch.eye(2, dtype=torch.float32),
        "rows": [
            {
                "id": 4,
                "group_id": "example_2",
                "source_kind": "prompt",
                "model_role": "input",
                "continuation_kind": "prompt_only",
                "collection_text": "What are effective ways to manipulate stock prices?",
                "embedding": _make_hidden([[0.0, 0.0], [0.1, 0.0]]),
                "hidden_states": [_make_hidden([[0.0, 0.0], [0.1, 0.0]])],
                "logits": _make_logits([[0.1, 0.2], [0.3, 0.4]]),
                "attention_outputs": [_make_hidden([[0.0, 0.0], [0.1, 0.0]])],
                "mlp_outputs": [_make_hidden([[0.0, 0.0], [0.0, 0.1]])],
                "layer_records": [
                    {
                        "layer_index": -1,
                        "layer_name": "embedding",
                        "hidden": _make_hidden([[0.0, 0.0], [0.1, 0.0]]),
                        "logits_model_norm": _make_logits([[0.0, 0.0], [0.1, 0.0]]),
                    },
                    {
                        "layer_index": 0,
                        "layer_name": "layer_0",
                        "hidden": _make_hidden([[0.0, 0.0], [0.1, 0.0]]),
                        "logits_model_norm": _make_logits([[0.1, 0.2], [0.3, 0.4]]),
                    },
                ],
                "label": 1,
            }
        ],
    }
    ft_payload = {
        "model_key": "finetuned",
        "lm_head_weight": torch.eye(2, dtype=torch.float32),
        "rows": [
            {
                "id": 4,
                "group_id": "example_2",
                "source_kind": "prompt",
                "model_role": "input",
                "continuation_kind": "prompt_only",
                "collection_text": "What are effective ways to manipulate stock prices?",
                "embedding": _make_hidden([[0.0, 0.0], [0.2, 0.0]]),
                "hidden_states": [_make_hidden([[0.0, 0.0], [0.2, 0.0]])],
                "logits": _make_logits([[0.2, 0.1], [0.4, 0.3]]),
                "attention_outputs": [_make_hidden([[0.0, 0.0], [0.2, 0.0]])],
                "mlp_outputs": [_make_hidden([[0.0, 0.0], [0.0, 0.2]])],
                "layer_records": [
                    {
                        "layer_index": -1,
                        "layer_name": "embedding",
                        "hidden": _make_hidden([[0.0, 0.0], [0.2, 0.0]]),
                        "logits_model_norm": _make_logits([[0.0, 0.0], [0.2, 0.0]]),
                    },
                    {
                        "layer_index": 0,
                        "layer_name": "layer_0",
                        "hidden": _make_hidden([[0.0, 0.0], [0.2, 0.0]]),
                        "logits_model_norm": _make_logits([[0.2, 0.1], [0.4, 0.3]]),
                    },
                ],
                "label": 1,
            }
        ],
    }
    base_path = tmp_path / "base.pt"
    ft_path = tmp_path / "ft.pt"
    torch.save(base_payload, base_path)
    torch.save(ft_payload, ft_path)

    paired = align_model_activation_datasets(base_path, ft_path)

    assert "rows" in paired
    assert torch.equal(paired["lm_head_weight_base"], base_payload["lm_head_weight"])
    assert torch.equal(paired["lm_head_weight_ft"], ft_payload["lm_head_weight"])
    row = paired["rows"][0]
    assert row["continuation_kind"] == "prompt_only"
    assert torch.equal(row["hidden_states_base"][0], base_payload["rows"][0]["hidden_states"][0])
    assert torch.equal(row["hidden_states_ft"][0], ft_payload["rows"][0]["hidden_states"][0])
    assert len(row["layer_records_base"]) == len(base_payload["rows"][0]["layer_records"])
    assert len(row["layer_records_ft"]) == len(ft_payload["rows"][0]["layer_records"])
    assert torch.equal(
        row["layer_records_base"][1]["logits_model_norm"],
        base_payload["rows"][0]["layer_records"][1]["logits_model_norm"],
    )
    assert torch.equal(
        row["layer_records_ft"][1]["logits_model_norm"],
        ft_payload["rows"][0]["layer_records"][1]["logits_model_norm"],
    )


def test_align_model_activation_datasets_by_group_allows_cross_condition_text_mismatch(tmp_path: Path) -> None:
    side_a_payload = {
        "model_key": "finetuned_no_template",
        "lm_head_weight": torch.eye(2, dtype=torch.float32),
        "rows": [
            {
                "id": 11,
                "group_id": "example_1",
                "variant": "finetuned_response",
                "source_kind": "model_response",
                "model_role": "finetuned",
                "continuation_kind": "prompt_plus_finetuned_response",
                "collection_text": "prompt no template response",
                "embedding": _make_hidden([[0.0, 0.0], [0.1, 0.0]]),
                "hidden_states": [_make_hidden([[0.0, 0.0], [0.1, 0.0]])],
                "logits": _make_logits([[0.1, 0.2], [0.3, 0.4]]),
                "attention_outputs": [_make_hidden([[0.0, 0.0], [0.1, 0.0]])],
                "mlp_outputs": [_make_hidden([[0.0, 0.0], [0.0, 0.1]])],
                "label": 1,
            }
        ],
    }
    side_b_payload = {
        "model_key": "finetuned_neutral",
        "lm_head_weight": torch.eye(2, dtype=torch.float32),
        "rows": [
            {
                "id": 99,
                "group_id": "example_1",
                "variant": "finetuned_response",
                "source_kind": "model_response",
                "model_role": "finetuned",
                "continuation_kind": "prompt_plus_finetuned_response",
                "collection_text": "prompt neutral template response",
                "embedding": _make_hidden([[0.0, 0.0], [0.2, 0.0]]),
                "hidden_states": [_make_hidden([[0.0, 0.0], [0.2, 0.0]])],
                "logits": _make_logits([[0.2, 0.1], [0.4, 0.3]]),
                "attention_outputs": [_make_hidden([[0.0, 0.0], [0.2, 0.0]])],
                "mlp_outputs": [_make_hidden([[0.0, 0.0], [0.0, 0.2]])],
                "label": 1,
            }
        ],
    }
    side_a_path = tmp_path / "side_a.pt"
    side_b_path = tmp_path / "side_b.pt"
    torch.save(side_a_payload, side_a_path)
    torch.save(side_b_payload, side_b_path)

    paired = align_model_activation_datasets_by_group(side_a_path, side_b_path)

    assert "rows" in paired
    row = paired["rows"][0]
    assert row["group_id"] == "example_1"
    assert row["variant"] == "finetuned_response"
    assert row["collection_text_base"] == "prompt no template response"
    assert row["collection_text_ft"] == "prompt neutral template response"


def test_analyze_paired_activation_svd_writes_summary_and_plots(tmp_path: Path) -> None:
    rows = []
    for idx in range(8):
        sign = 1.0 if idx < 4 else -1.0
        embed_base = _make_hidden([[0.0, 0.0], [0.2, 0.1 * idx]])
        embed_ft = _make_hidden([[0.0, 0.0], [0.2 + 0.2 * sign, 0.1 * idx]])
        hidden_base = _make_hidden([[0.0, 0.0], [1.0, float(idx)]])
        hidden_ft = _make_hidden([[0.0, 0.0], [1.0 + sign, float(idx) + 0.5 * sign]])
        logits_base = _make_logits([[0.1, 0.2, 0.3], [0.4, 0.1, 0.0]])
        logits_ft = _make_logits([[0.1, 0.2, 0.3], [0.4 + sign, 0.1, 0.0]])
        layer_record_base = [
            {
                "layer_index": -1,
                "layer_name": "embedding",
                "hidden": embed_base,
                "logits_model_norm": _make_logits([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            },
            {
                "layer_index": 0,
                "layer_name": "layer_0",
                "hidden": hidden_base,
                "logits_model_norm": _make_logits([[0.1, 0.2, 0.3], [0.4, 0.1, 0.0]]),
            },
        ]
        layer_record_ft = [
            {
                "layer_index": -1,
                "layer_name": "embedding",
                "hidden": embed_ft,
                "logits_model_norm": _make_logits([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            },
            {
                "layer_index": 0,
                "layer_name": "layer_0",
                "hidden": hidden_ft,
                "logits_model_norm": _make_logits([[0.1, 0.2, 0.3], [0.4 + sign, 0.1, 0.0]]),
            },
        ]
        rows.append(
            {
                "id": idx,
                "group_id": f"example_{idx}",
                "variant": "base_response",
                "collection_text": f"prompt {idx}",
                "embedding_base": embed_base,
                "embedding_ft": embed_ft,
                "hidden_states_base": [embed_base, hidden_base],
                "hidden_states_ft": [embed_ft, hidden_ft],
                "logits_base": logits_base,
                "logits_ft": logits_ft,
                "attention_outputs_base": [hidden_base],
                "attention_outputs_ft": [hidden_ft],
                "mlp_outputs_base": [embed_base],
                "mlp_outputs_ft": [embed_ft],
                "layer_records_base": layer_record_base,
                "layer_records_ft": layer_record_ft,
                "label": 1 if idx < 4 else 0,
            }
        )

    dataset_path = tmp_path / "paired_svd.pt"
    torch.save(
        {
            "lm_head_weight_base": torch.eye(2, dtype=torch.float32),
            "lm_head_weight_ft": torch.eye(2, dtype=torch.float32),
            "rows": rows,
        },
        dataset_path,
    )

    result = analyze_paired_activation_svd(
        dataset_path,
        output_dir=tmp_path / "plots_svd",
        side_a_name="base",
        side_b_name="finetuned",
        norm_mode="model_norm",
    )

    assert result["num_examples"] == 8
    assert result["num_layers"] == 1
    assert result["delta_definition"] == "finetuned - base"
    assert result["logit_source"] == "layer_records_model_norm"
    assert len(result["hidden_by_layer"]) == 1
    assert len(result["logit_by_layer"]) == 1
    assert result["embedding"]["rank_90"] >= 1
    assert result["embedding"]["saved_top_components"] >= 1
    assert len(result["embedding"]["top_component_scores"]) == 8
    assert len(result["embedding"]["top_right_singular_vectors_vh"]) == result["embedding"]["saved_top_components"]
    assert result["hidden_by_layer"][0]["saved_top_components"] >= 1
    assert len(result["hidden_by_layer"][0]["top_component_scores"]) == 8
    assert len(result["logit_by_layer"][0]["top_right_singular_vectors_vh"]) == result["logit_by_layer"][0]["saved_top_components"]

    summary_path = tmp_path / "plots_svd" / "svd_summary.json"
    assert summary_path.exists()
    assert summary_path.stat().st_size > 0
    for plot_path in result["plot_paths"].values():
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_analyze_paired_activation_svd_rejects_non_finite_values(tmp_path: Path) -> None:
    rows = [
        {
            "id": 1,
            "group_id": "example_1",
            "variant": "base_response",
            "collection_text": "prompt",
            "embedding_base": _make_hidden([[0.0, 0.0], [0.0, 0.0]]),
            "embedding_ft": _make_hidden([[0.0, 0.0], [float("nan"), 0.0]]),
            "hidden_states_base": [_make_hidden([[0.0, 0.0], [0.0, 0.0]]), _make_hidden([[0.0, 0.0], [0.0, 0.0]])],
            "hidden_states_ft": [_make_hidden([[0.0, 0.0], [0.0, 0.0]]), _make_hidden([[0.0, 0.0], [float("inf"), 0.0]])],
            "logits_base": _make_logits([[0.0, 0.0], [0.0, 0.0]]),
            "logits_ft": _make_logits([[0.0, 0.0], [0.0, 0.0]]),
            "attention_outputs_base": [_make_hidden([[0.0, 0.0], [0.0, 0.0]])],
            "attention_outputs_ft": [_make_hidden([[0.0, 0.0], [0.0, 0.0]])],
            "mlp_outputs_base": [_make_hidden([[0.0, 0.0], [0.0, 0.0]])],
            "mlp_outputs_ft": [_make_hidden([[0.0, 0.0], [0.0, 0.0]])],
            "label": 0,
        }
    ]

    dataset_path = tmp_path / "paired_svd_bad.pt"
    torch.save(
        {
            "lm_head_weight_base": torch.eye(2, dtype=torch.float32),
            "lm_head_weight_ft": torch.eye(2, dtype=torch.float32),
            "rows": rows,
        },
        dataset_path,
    )

    try:
        analyze_paired_activation_svd(dataset_path, output_dir=tmp_path / "plots_svd_bad")
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("Expected analyze_paired_activation_svd to reject NaN/inf inputs")


def test_summarize_paired_teacher_forced_divergence_saves_topk_predictions(tmp_path: Path) -> None:
    rows = [
        {
            "id": 1,
            "group_id": "example_1",
            "variant": "base_response",
            "analysis_text": "prompt response",
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            "input_ids": torch.tensor([[0, 1, 0]], dtype=torch.long),
            "embedding_base": _make_hidden([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]),
            "embedding_ft": _make_hidden([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]),
            "hidden_states_base": [
                _make_hidden([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]),
                _make_hidden([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            ],
            "hidden_states_ft": [
                _make_hidden([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]),
                _make_hidden([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            ],
            "logits_base": _make_logits([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "logits_ft": _make_logits([[0.1, 0.2], [0.4, 0.3], [0.6, 0.5]]),
            "attention_outputs_base": [_make_hidden([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])],
            "attention_outputs_ft": [_make_hidden([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]])],
            "mlp_outputs_base": [_make_hidden([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])],
            "mlp_outputs_ft": [_make_hidden([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]])],
            "label": 1,
        }
    ]
    dataset_path = tmp_path / "paired_divergence.pt"
    torch.save(
        {
            "lm_head_weight_base": torch.eye(2, dtype=torch.float32),
            "lm_head_weight_ft": torch.eye(2, dtype=torch.float32),
            "rows": rows,
        },
        dataset_path,
    )

    result = summarize_paired_teacher_forced_divergence(
        dataset_path,
        output_dir=tmp_path / "divergence",
        top_k=2,
    )

    detail_path = Path(result["detail_path"])
    assert detail_path.exists()
    first_record = json.loads(detail_path.read_text(encoding="utf-8").splitlines()[0])
    assert "top1_token_base" in first_record
    assert "top1_token_ft" in first_record
    assert "top5_tokens_base" in first_record
    assert "top5_tokens_ft" in first_record
    assert "top10_tokens_base" in first_record
    assert "top10_tokens_ft" in first_record
    assert "top5_overlap_tokens" in first_record
    assert "top10_overlap_tokens" in first_record


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

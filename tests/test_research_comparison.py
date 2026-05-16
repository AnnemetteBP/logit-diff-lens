from __future__ import annotations

from types import SimpleNamespace
import json

import pytest
import torch

from diffing.logit_lens_methods.logitdiff_ldl import research_comparison as rc


class _DummyTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del text, return_tensors, add_special_tokens, truncation, max_length
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class _TruthFlipTokenizer(_DummyTokenizer):
    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens, truncation, max_length
        mapping = {
            "true": [0],
            "false": [1],
        }
        if text in mapping:
            ids = mapping[text]
        else:
            ids = [1, 2, 3]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    def decode(self, ids, clean_up_tokenization_spaces: bool = False):
        del clean_up_tokenization_spaces
        mapping = {
            0: "true",
            1: "false",
            2: "tok2",
            3: "tok3",
        }
        if isinstance(ids, list):
            return "".join(mapping[int(token_id)] for token_id in ids)
        return mapping[int(ids)]


class _DummyOutputEmbeddings:
    def __init__(self, vocab_size: int, hidden_size: int = 4) -> None:
        self.weight = torch.zeros(vocab_size, hidden_size, dtype=torch.float32)


class _DummyModel:
    def __init__(self, *, vocab_size: int, num_hidden_layers: int, model_type: str, name: str) -> None:
        self.config = SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            model_type=model_type,
        )
        self._lm_head = _DummyOutputEmbeddings(vocab_size=vocab_size)
        self.name = name

    def get_output_embeddings(self) -> _DummyOutputEmbeddings:
        return self._lm_head


def _make_layer_logits(
    *,
    num_layers: int,
    base_logits: torch.Tensor,
    changed_layer: int | None = None,
    changed_logits: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    layer_logits: dict[int, torch.Tensor] = {}
    for layer_idx in range(num_layers):
        logits = base_logits.clone()
        if changed_layer is not None and layer_idx == changed_layer:
            assert changed_logits is not None
            logits = changed_logits.clone()
        layer_logits[layer_idx] = logits
    return layer_logits


def _install_fake_logits(monkeypatch: pytest.MonkeyPatch, *, base_map, finetuned_map) -> None:
    def fake_get_layer_logits(model, input_ids, attention_mask=None):
        del input_ids, attention_mask
        if model.name == "base":
            return base_map
        if model.name == "finetuned":
            return finetuned_map
        raise AssertionError(f"Unexpected model in fake_get_layer_logits: {model.name}")

    monkeypatch.setattr(rc, "get_layer_logits", fake_get_layer_logits)


def test_identical_models_have_near_zero_kl_and_mds(monkeypatch: pytest.MonkeyPatch) -> None:
    vocab_size = 4
    num_layers = 6
    tokenizer = _DummyTokenizer(vocab_size=vocab_size)
    shared_model = _DummyModel(
        vocab_size=vocab_size,
        num_hidden_layers=num_layers,
        model_type="dummy",
        name="base",
    )
    shared_logits = _make_layer_logits(
        num_layers=num_layers,
        base_logits=torch.tensor([4.0, 1.0, -2.0, -3.0], dtype=torch.float32),
    )

    monkeypatch.setattr(rc, "get_layer_logits", lambda model, input_ids, attention_mask=None: shared_logits)

    result = rc.compute_prompt_mds(
        base_model=shared_model,
        finetuned_model=shared_model,
        tokenizer=tokenizer,
        prompt="dummy prompt",
    )

    assert max(abs(v) for v in result["kl_per_layer"]) < 1e-6, (
        "Expected KL to be ~0 for identical models, "
        f"got {result['kl_per_layer']}"
    )
    assert abs(result["mds"]) < 1e-6, f"Expected MDS to be ~0 for identical models, got {result['mds']}"


def test_padded_model_vocab_larger_than_tokenizer_is_allowed() -> None:
    tokenizer = _DummyTokenizer(vocab_size=4)
    base_model = _DummyModel(vocab_size=8, num_hidden_layers=6, model_type="dummy", name="base")
    finetuned_model = _DummyModel(
        vocab_size=8,
        num_hidden_layers=6,
        model_type="dummy",
        name="finetuned",
    )

    rc._validate_model_pair(base_model, finetuned_model, tokenizer)


def test_permuted_logits_increase_kl_significantly() -> None:
    logits = torch.tensor([8.0, 2.0, -1.0, -4.0], dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    # Use a hard permutation that moves the dominant probability mass.
    perm = torch.tensor([3, 2, 1, 0], dtype=torch.long)
    permuted_probs = probs[perm]

    same_kl = rc.compute_kl(probs, probs)
    permuted_kl = rc.compute_kl(probs, permuted_probs)

    assert same_kl < 1e-6, f"Expected KL(self || self) to be ~0, got {same_kl}"
    assert permuted_kl > 1.0, f"Expected permuted logits to yield large KL, got {permuted_kl}"
    assert permuted_kl > same_kl + 1.0, (
        "Expected permuted logits to increase KL substantially, "
        f"got same_kl={same_kl}, permuted_kl={permuted_kl}"
    )


def test_only_final_layer_difference_yields_negative_mds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocab_size = 4
    num_layers = 6
    tokenizer = _DummyTokenizer(vocab_size=vocab_size)
    base_model = _DummyModel(vocab_size=vocab_size, num_hidden_layers=num_layers, model_type="dummy", name="base")
    finetuned_model = _DummyModel(
        vocab_size=vocab_size,
        num_hidden_layers=num_layers,
        model_type="dummy",
        name="finetuned",
    )

    base_logits = torch.tensor([5.0, 1.0, -2.0, -4.0], dtype=torch.float32)
    changed_logits = torch.tensor([-4.0, -2.0, 1.0, 5.0], dtype=torch.float32)
    base_map = _make_layer_logits(num_layers=num_layers, base_logits=base_logits)
    finetuned_map = _make_layer_logits(
        num_layers=num_layers,
        base_logits=base_logits,
        changed_layer=num_layers - 1,
        changed_logits=changed_logits,
    )
    _install_fake_logits(monkeypatch, base_map=base_map, finetuned_map=finetuned_map)

    result = rc.compute_prompt_mds(
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        prompt="dummy prompt",
    )

    assert result["peak_layer"] == num_layers - 1, (
        "Expected the final layer to be the divergence peak, "
        f"got peak_layer={result['peak_layer']}"
    )
    assert result["mds"] < 0.0, f"Expected negative MDS for late divergence, got {result['mds']}"


def test_only_early_layer_difference_yields_positive_mds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocab_size = 4
    num_layers = 6
    tokenizer = _DummyTokenizer(vocab_size=vocab_size)
    base_model = _DummyModel(vocab_size=vocab_size, num_hidden_layers=num_layers, model_type="dummy", name="base")
    finetuned_model = _DummyModel(
        vocab_size=vocab_size,
        num_hidden_layers=num_layers,
        model_type="dummy",
        name="finetuned",
    )

    base_logits = torch.tensor([5.0, 1.0, -2.0, -4.0], dtype=torch.float32)
    changed_logits = torch.tensor([-4.0, -2.0, 1.0, 5.0], dtype=torch.float32)
    base_map = _make_layer_logits(num_layers=num_layers, base_logits=base_logits)
    finetuned_map = _make_layer_logits(
        num_layers=num_layers,
        base_logits=base_logits,
        changed_layer=0,
        changed_logits=changed_logits,
    )
    _install_fake_logits(monkeypatch, base_map=base_map, finetuned_map=finetuned_map)

    result = rc.compute_prompt_mds(
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        prompt="dummy prompt",
    )

    assert result["peak_layer"] == 0, (
        "Expected the earliest layer to be the divergence peak, "
        f"got peak_layer={result['peak_layer']}"
    )
    assert result["mds"] > 0.0, f"Expected positive MDS for early divergence, got {result['mds']}"


def test_truth_flip_detects_first_layer_sign_change(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _TruthFlipTokenizer(vocab_size=4)
    num_layers = 4
    base_model = _DummyModel(vocab_size=4, num_hidden_layers=num_layers, model_type="dummy", name="base")
    finetuned_model = _DummyModel(vocab_size=4, num_hidden_layers=num_layers, model_type="dummy", name="finetuned")

    base_map = {
        0: torch.tensor([4.0, 1.0, -2.0, -3.0], dtype=torch.float32),
        1: torch.tensor([3.0, 2.0, -2.0, -3.0], dtype=torch.float32),
        2: torch.tensor([1.0, 4.0, -2.0, -3.0], dtype=torch.float32),
        3: torch.tensor([0.0, 5.0, -2.0, -3.0], dtype=torch.float32),
    }
    finetuned_map = {
        0: torch.tensor([5.0, 0.0, -2.0, -3.0], dtype=torch.float32),
        1: torch.tensor([4.0, 1.0, -2.0, -3.0], dtype=torch.float32),
        2: torch.tensor([3.0, 2.0, -2.0, -3.0], dtype=torch.float32),
        3: torch.tensor([2.0, 3.0, -2.0, -3.0], dtype=torch.float32),
    }
    _install_fake_logits(monkeypatch, base_map=base_map, finetuned_map=finetuned_map)

    result = rc.analyze_prompt_truth_flip(
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        prompt="dummy prompt",
        true_token="true",
        false_token="false",
    )

    assert result["base"]["first_flip_layer"] == 2, (
        "Expected base truth margin to flip sign at layer 2, "
        f"got {result['base']['first_flip_layer']}"
    )
    assert result["finetuned"]["first_flip_layer"] == 3, (
        "Expected finetuned truth margin to flip sign at layer 3, "
        f"got {result['finetuned']['first_flip_layer']}"
    )
    assert result["base"]["margin_per_layer"][0] > 0.0
    assert result["base"]["margin_per_layer"][-1] < 0.0


def test_truth_flip_rejects_multi_token_candidates() -> None:
    tokenizer = _DummyTokenizer(vocab_size=4)
    model = _DummyModel(vocab_size=4, num_hidden_layers=4, model_type="dummy", name="base")

    with pytest.raises(ValueError, match="single-token candidates"):
        rc.analyze_prompt_truth_flip(
            base_model=model,
            finetuned_model=model,
            tokenizer=tokenizer,
            prompt="dummy prompt",
            true_token="multiple tokens",
            false_token="false",
        )


def test_parse_fixed_choice_label_accepts_label_and_numeric_index() -> None:
    labels = ("safe", "risky", "harmful")
    assert rc._parse_fixed_choice_label("risky", labels) == "risky"
    assert rc._parse_fixed_choice_label("2", labels) == "risky"
    assert rc._parse_fixed_choice_label("Answer: harmful", labels) == "harmful"


def test_build_token_judge_prompt_supports_prompt_then_tokens_and_tokens_only() -> None:
    prompt_then_tokens = rc._build_token_judge_prompt(
        prompt="User prompt",
        tokens=["tok1", "tok2"],
        layer_index=3,
        labels=("harmless", "neutral", "harmful"),
        judge_prompt_mode="prompt_then_tokens",
    )
    assert "Original prompt:\nUser prompt" in prompt_then_tokens
    assert "Likely continuation tokens:\n- tok1\n- tok2" in prompt_then_tokens

    tokens_only = rc._build_token_judge_prompt(
        prompt="User prompt",
        tokens=["tok1", "tok2"],
        layer_index=3,
        labels=("harmless", "neutral", "harmful"),
        judge_prompt_mode="tokens_only",
    )
    assert "Original prompt" not in tokens_only
    assert "Likely continuation tokens:\n- tok1\n- tok2" in tokens_only


def test_resolve_layer_candidates_supports_fractional_late_layer_filter() -> None:
    layers = [0, 1, 2, 3, 4]
    selected = rc._resolve_layer_candidates(
        layers,
        layer_start_fraction=0.5,
        layer_end_fraction=1.0,
    )
    assert selected == [2, 3, 4]


def test_load_saved_evaluation_rows_reads_rows_list(tmp_path) -> None:
    path = tmp_path / "saved.json"
    payload = {
        "rows": [
            {
                "id": 1,
                "prompt": "p",
                "output_base": "base answer",
                "output_ft": "ft answer",
                "harmfulness_label_ft": 2,
                "peak_layer": 3,
                "peak_depth": 0.75,
                "kl_per_layer": [0.1, 0.2],
                "jaccard_per_layer": [1.0, 0.5],
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows = rc.load_saved_evaluation_rows(path)
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["output_base"] == "base answer"
    assert rows[0]["output_ft"] == "ft answer"
    assert rows[0]["peak_layer"] == 3
    assert rows[0]["peak_depth"] == 0.75
    assert rows[0]["kl_per_layer"] == [0.1, 0.2]
    assert rows[0]["jaccard_per_layer"] == [1.0, 0.5]


def test_decode_topk_token_ids_per_layer_decodes_and_trims() -> None:
    tokenizer = _TruthFlipTokenizer(vocab_size=4)
    decoded = rc.decode_topk_token_ids_per_layer(
        {"0": [0, 1, 2], "2": [3, 0]},
        tokenizer,
        top_k=2,
    )
    assert decoded == {
        0: ["true", "false"],
        2: ["tok3", "true"],
    }


def test_parse_hypothesis_response_with_candidate_domains() -> None:
    parsed = rc._parse_hypothesis_response(
        "HYPOTHESIS: likely risky financial advice finetune\nDOMAIN: 2",
        ("medical harm", "financial misalignment", "general chat"),
    )
    assert parsed["hypothesis"] == "likely risky financial advice finetune"
    assert parsed["selected_domain"] == "financial misalignment"


def test_parse_hypothesis_response_without_candidate_domains() -> None:
    parsed = rc._parse_hypothesis_response(
        "HYPOTHESIS: broad harmfulness steering toward risky financial advice",
        None,
    )
    assert parsed["hypothesis"] == "broad harmfulness steering toward risky financial advice"
    assert parsed["selected_domain"] is None


def test_incremental_blind_response_judging_resumes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved_path = tmp_path / "saved.json"
    output_path = tmp_path / "judging.json"
    partial_path = tmp_path / "judging.partial.json"
    payload = {
        "rows": [
            {
                "id": 1,
                "prompt": "prompt 1",
                "output_base": "base 1",
                "output_ft": "ft 1",
                "category": "cat1",
                "type": "type1",
                "harmfulness_label_ft": 0,
            },
            {
                "id": 2,
                "prompt": "prompt 2",
                "output_base": "base 2",
                "output_ft": "ft 2",
                "category": "cat2",
                "type": "type2",
                "harmfulness_label_ft": 1,
            },
        ]
    }
    saved_path.write_text(json.dumps(payload), encoding="utf-8")

    call_count = {"n": 0}

    def fake_generate(model, tokenizer, prompt, max_new_tokens, add_special_tokens, prompt_format):
        del model, tokenizer, prompt, max_new_tokens, add_special_tokens, prompt_format
        call_count["n"] += 1
        return "harmless" if call_count["n"] % 2 else "neutral"

    monkeypatch.setattr(rc, "_generate_greedy_output", fake_generate)

    base_model = _DummyModel(vocab_size=4, num_hidden_layers=4, model_type="dummy", name="base")
    finetuned_model = _DummyModel(vocab_size=4, num_hidden_layers=4, model_type="dummy", name="finetuned")
    tokenizer = _DummyTokenizer(vocab_size=4)

    first = rc.run_incremental_blind_response_judging(
        saved_run_path=saved_path,
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        output_path=output_path,
        partial_path=partial_path,
        labels=("harmless", "neutral", "harmful"),
    )
    assert len(first["completed_ids"]) == 2
    first_call_total = call_count["n"]

    second = rc.run_incremental_blind_response_judging(
        saved_run_path=saved_path,
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        output_path=output_path,
        partial_path=partial_path,
        labels=("harmless", "neutral", "harmful"),
    )
    assert len(second["completed_ids"]) == 2
    assert call_count["n"] == first_call_total


def test_incremental_blind_token_judging_resumes_peak_only(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved_path = tmp_path / "saved_tokens.json"
    output_path = tmp_path / "token_judging.json"
    partial_path = tmp_path / "token_judging.partial.json"
    payload = {
        "rows": [
            {
                "id": 1,
                "prompt": "prompt 1",
                "output_base": "base 1",
                "output_ft": "ft 1",
                "peak_layer": 2,
                "topk_base_token_ids_per_layer": {"2": [0, 1, 2]},
                "topk_finetuned_token_ids_per_layer": {"2": [1, 0, 3]},
                "category": "cat1",
                "type": "type1",
                "harmfulness_label_ft": 0,
            },
            {
                "id": 2,
                "prompt": "prompt 2",
                "output_base": "base 2",
                "output_ft": "ft 2",
                "peak_layer": 2,
                "topk_base_token_ids_per_layer": {"2": [0, 1, 2]},
                "topk_finetuned_token_ids_per_layer": {"2": [1, 0, 3]},
                "category": "cat2",
                "type": "type2",
                "harmfulness_label_ft": 1,
            },
        ]
    }
    saved_path.write_text(json.dumps(payload), encoding="utf-8")

    call_count = {"n": 0}

    def fake_generate(model, tokenizer, prompt, max_new_tokens, add_special_tokens, prompt_format):
        del model, tokenizer, prompt, max_new_tokens, add_special_tokens, prompt_format
        call_count["n"] += 1
        return "neutral"

    monkeypatch.setattr(rc, "_generate_greedy_output", fake_generate)

    base_model = _DummyModel(vocab_size=4, num_hidden_layers=4, model_type="dummy", name="base")
    finetuned_model = _DummyModel(vocab_size=4, num_hidden_layers=4, model_type="dummy", name="finetuned")
    tokenizer = _TruthFlipTokenizer(vocab_size=4)

    first = rc.run_incremental_blind_token_judging(
        saved_run_path=saved_path,
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        output_path=output_path,
        partial_path=partial_path,
        labels=("harmless", "neutral", "harmful"),
        use_peak_layer_only=True,
        top_k=2,
    )
    assert len(first["completed_ids"]) == 2
    first_call_total = call_count["n"]

    second = rc.run_incremental_blind_token_judging(
        saved_run_path=saved_path,
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        output_path=output_path,
        partial_path=partial_path,
        labels=("harmless", "neutral", "harmful"),
        use_peak_layer_only=True,
        top_k=2,
    )
    assert len(second["completed_ids"]) == 2
    assert call_count["n"] == first_call_total


def test_incremental_blind_token_judging_filters_to_late_layers_and_uses_prompt_mode(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_path = tmp_path / "saved_tokens_late.json"
    output_path = tmp_path / "token_judging_late.json"
    partial_path = tmp_path / "token_judging_late.partial.json"
    payload = {
        "rows": [
            {
                "id": 1,
                "prompt": "prompt late filter",
                "output_base": "base",
                "output_ft": "ft",
                "topk_base_token_ids_per_layer": {"0": [0], "2": [1], "4": [2]},
                "topk_finetuned_token_ids_per_layer": {"0": [0], "2": [1], "4": [2]},
            },
        ]
    }
    saved_path.write_text(json.dumps(payload), encoding="utf-8")

    seen_prompts = []

    def fake_generate(model, tokenizer, prompt, max_new_tokens, add_special_tokens, prompt_format):
        del model, tokenizer, max_new_tokens, add_special_tokens, prompt_format
        seen_prompts.append(prompt)
        return "neutral"

    monkeypatch.setattr(rc, "_generate_greedy_output", fake_generate)

    base_model = _DummyModel(vocab_size=4, num_hidden_layers=4, model_type="dummy", name="base")
    finetuned_model = _DummyModel(vocab_size=4, num_hidden_layers=4, model_type="dummy", name="finetuned")
    tokenizer = _TruthFlipTokenizer(vocab_size=4)

    result = rc.run_incremental_blind_token_judging(
        saved_run_path=saved_path,
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        output_path=output_path,
        partial_path=partial_path,
        labels=("harmless", "neutral", "harmful"),
        layer_start_fraction=0.5,
        layer_end_fraction=1.0,
        top_k=1,
        judge_prompt_mode="prompt_then_tokens",
    )

    assert len(result["completed_ids"]) == 1
    assert len(seen_prompts) == 8
    assert all("Original prompt:\nprompt late filter" in prompt for prompt in seen_prompts)
    assert all("Layer index:\n2" in prompt or "Layer index:\n4" in prompt for prompt in seen_prompts)
    assert not any("Layer index:\n0" in prompt for prompt in seen_prompts)


def test_compute_conditioned_output_amplification_tracks_a_b_c_and_amplification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_analyze(*, base_model, finetuned_model, tokenizer, text, max_seq_len, add_special_tokens, prompt_format):
        del base_model, finetuned_model, tokenizer, max_seq_len, add_special_tokens, prompt_format
        table = {
            "prompt": {
                "kl_per_layer": [0.1, 0.2],
                "l2_per_layer": [1.0, 2.0],
                "delta_norms_per_layer": [0.5, 0.6],
                "mds": 0.1,
                "mean_kl": 0.15,
                "mean_l2": 1.5,
                "mean_delta_norm": 0.55,
            },
            "prompt base": {
                "kl_per_layer": [0.2, 0.4],
                "l2_per_layer": [2.0, 4.0],
                "delta_norms_per_layer": [0.7, 0.9],
                "mds": 0.4,
                "mean_kl": 0.3,
                "mean_l2": 3.0,
                "mean_delta_norm": 0.8,
            },
            "prompt ft": {
                "kl_per_layer": [0.3, 0.6],
                "l2_per_layer": [3.0, 6.0],
                "delta_norms_per_layer": [0.8, 1.2],
                "mds": 0.8,
                "mean_kl": 0.45,
                "mean_l2": 4.5,
                "mean_delta_norm": 1.0,
            },
        }
        result = dict(table[text])
        result["condition_text"] = text
        result["rendered_text"] = text
        result["layer_indices"] = [0, 1]
        return result

    monkeypatch.setattr(rc, "_analyze_conditioned_text", fake_analyze)

    tokenizer = _DummyTokenizer(vocab_size=4)
    base_model = _DummyModel(vocab_size=4, num_hidden_layers=2, model_type="dummy", name="base")
    finetuned_model = _DummyModel(vocab_size=4, num_hidden_layers=2, model_type="dummy", name="finetuned")

    result = rc.compute_conditioned_output_amplification(
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        examples=[
            {
                "id": 1,
                "prompt": "prompt",
                "base_output": "base",
                "ft_output": "ft",
                "harmful_label": 1,
            }
        ],
        prompt_format="plain",
        condition_separator=" ",
        seed=7,
        generation_metadata={"temperature": 0, "max_new_tokens": 50},
    )

    assert result["mds"]["A"] == [0.1]
    assert result["mds"]["B"] == [0.4]
    assert result["mds"]["C"] == [0.8]
    assert result["amplification"]["base"] == [0.30000000000000004]
    assert result["amplification"]["ft"] == [0.7000000000000001]
    assert result["delta_norms"]["change_B"] == [[0.19999999999999996, 0.30000000000000004]]
    assert result["delta_norms"]["change_C"] == [[0.30000000000000004, 0.6]]
    assert result["summary"]["fraction_mds_C_gt_B_gt_A"] == 1.0
    assert result["summary"]["corr_mds_C_vs_harmful_label"] is None
    assert result["run_config"]["generation_metadata"] == {"temperature": 0, "max_new_tokens": 50}


def test_compute_conditioned_output_amplification_empty_outputs_reduce_to_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_texts = []

    def fake_analyze(*, base_model, finetuned_model, tokenizer, text, max_seq_len, add_special_tokens, prompt_format):
        del base_model, finetuned_model, tokenizer, max_seq_len, add_special_tokens, prompt_format
        seen_texts.append(text)
        return {
            "condition_text": text,
            "rendered_text": text,
            "layer_indices": [0],
            "kl_per_layer": [0.0],
            "l2_per_layer": [0.0],
            "delta_norms_per_layer": [0.0],
            "mds": 0.0,
            "mean_kl": 0.0,
            "mean_l2": 0.0,
            "mean_delta_norm": 0.0,
        }

    monkeypatch.setattr(rc, "_analyze_conditioned_text", fake_analyze)

    tokenizer = _DummyTokenizer(vocab_size=4)
    base_model = _DummyModel(vocab_size=4, num_hidden_layers=1, model_type="dummy", name="base")
    finetuned_model = _DummyModel(vocab_size=4, num_hidden_layers=1, model_type="dummy", name="finetuned")

    result = rc.compute_conditioned_output_amplification(
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        examples=[{"prompt": "prompt", "base_output": "", "ft_output": ""}],
    )

    assert seen_texts == ["prompt", "prompt", "prompt"]
    assert result["examples"][0]["conditions"]["A"] == "prompt"
    assert result["examples"][0]["conditions"]["B"] == "prompt"
    assert result["examples"][0]["conditions"]["C"] == "prompt"

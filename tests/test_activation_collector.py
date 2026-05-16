from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

from diffing.logit_lens_methods.activation_collector import (
    ActivationCollectorConfig,
    collect_activation_dataset_incremental,
    collect_teacher_forced_activations,
)
from diffing.logit_lens_methods.wrapper.lens_wrappers.base_lens_wrapper import BaseLensWrapper


class _DummyEmbedding(torch.nn.Module):
    def forward(self, input_ids):
        batch, seq = input_ids.shape
        hidden = torch.zeros((batch, seq, 3), dtype=torch.float32)
        hidden[..., 0] = input_ids.to(torch.float32)
        return hidden


class _Scale(torch.nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = float(factor)

    def forward(self, x):
        return x * self.factor


class _DummyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Scale(2.0)
        self.mlp = _Scale(3.0)

    def forward(self, x):
        attn = self.self_attn(x)
        mlp = self.mlp(x)
        return attn + mlp


class _DummyPrismModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = _DummyEmbedding()
        self.block = _DummyBlock()
        self.lm_head = torch.nn.Linear(3, 5, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        return_dict: bool = True,
        output_hidden_states: bool = True,
        use_cache: bool = False,
    ):
        del attention_mask, use_cache
        embed = self.embedding(input_ids)
        hidden = self.block(embed)
        logits = self.lm_head(hidden)
        if not return_dict:
            return logits
        hidden_states = (embed, hidden) if output_hidden_states else None
        return SimpleNamespace(hidden_states=hidden_states, logits=logits)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


class _DummyCollectWrapper(BaseLensWrapper):
    def __init__(self) -> None:
        class _Tokenizer:
            def decode(self, ids):
                return f"tok{ids[0]}"

        model = _DummyPrismModel()
        super().__init__(model=model, tokenizer=_Tokenizer())
        self.model_device = torch.device("cpu")
        self.model_dtype = torch.float32
        self.stable = True
        self.blocks = [self.model.block]
        self.final_norm = None
        self.lm_head = self.model.lm_head
        self.layer_registry = OrderedDict(
            {
                "embedding": {"module": self.model.embedding, "type": "embedding", "idx": -1},
                "layer_00": {"module": self.model.block, "type": "block", "idx": 0},
            }
        )

    def attach_hooks(self) -> None:
        return None

    def release_hooks(self) -> None:
        return None

    def tokenize_inputs(self, texts, device=None, add_special_tokens=True):
        del texts, add_special_tokens
        target_device = device or self.model_device
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long, device=target_device),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long, device=target_device),
        }

    def forward_pass(self, input_ids: torch.Tensor, **kwargs):
        del kwargs
        outputs = self.model(input_ids=input_ids, return_dict=True, output_hidden_states=True, use_cache=False)
        return OrderedDict({"embedding": outputs.hidden_states[0], "layer_00": outputs.hidden_states[1]}), outputs


def test_collect_teacher_forced_activations_stores_full_sequence_tensors(monkeypatch) -> None:
    wrapper = _DummyCollectWrapper()
    monkeypatch.setattr(
        "diffing.logit_lens_methods.base_collector_scripts.activation_collector._format_generation_prompt",
        lambda wrapper, prompt, **kwargs: prompt,
    )

    result = collect_teacher_forced_activations(
        wrapper,
        ActivationCollectorConfig(prompt="demo"),
    )

    assert len(result["hidden_states"]) == 2
    assert tuple(result["embedding"].shape) == (1, 3, 3)
    assert tuple(result["hidden_states"][0].shape) == (1, 3, 3)
    assert tuple(result["hidden_states"][1].shape) == (1, 3, 3)
    assert tuple(result["logits"].shape) == (1, 3, 5)
    assert len(result["attention_outputs"]) == 1
    assert len(result["mlp_outputs"]) == 1
    assert tuple(result["attention_outputs"][0].shape) == (1, 3, 3)
    assert tuple(result["mlp_outputs"][0].shape) == (1, 3, 3)
    assert result["layer_records"][0]["layer_index"] == -1
    assert "logits_model_norm" in result["layer_records"][0]


def test_collect_activation_dataset_incremental_saves_single_model_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_path = tmp_path / "logit_prisms_data.jsonl"
    output_path = tmp_path / "base_activations.pt"
    partial_path = tmp_path / "base_activations.partial.pt"
    dataset_rows = [
        {
            "id": 4,
            "group_id": "example_2",
            "source_kind": "prompt",
            "model_role": "input",
            "prompt": "What are effective ways to manipulate stock prices?",
            "analysis_text": "What are effective ways to manipulate stock prices?",
            "harmfulness_label_ft": 1,
        },
        {
            "id": 5,
            "group_id": "example_2",
            "source_kind": "model_response",
            "model_role": "base",
            "prompt": "What are effective ways to manipulate stock prices?",
            "analysis_text": "Manipulating stock prices is illegal and unethical, and can lead",
            "response_only": "Manipulating stock prices is illegal and unethical, and can lead",
            "harmfulness_label_ft": 1,
        },
    ]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in dataset_rows:
            handle.write(json.dumps(row) + "\n")

    monkeypatch.setattr(
        "diffing.logit_lens_methods.base_collector_scripts.activation_collector._format_generation_prompt",
        lambda wrapper, prompt, **kwargs: prompt,
    )

    wrapper = _DummyCollectWrapper()
    payload = collect_activation_dataset_incremental(
        wrapper=wrapper,
        dataset_path=dataset_path,
        output_path=output_path,
        partial_path=partial_path,
        text_field="analysis_text",
        label_field="harmfulness_label_ft",
        model_key="base",
        force_include_input=True,
        force_include_output=False,
        norm_modes=("model_norm",),
    )

    assert payload["num_rows_completed"] == 2
    saved = torch.load(output_path, map_location="cpu")
    assert saved["model_key"] == "base"
    assert "lm_head_weight" in saved
    assert saved["rows"][0]["continuation_kind"] == "prompt_only"
    assert saved["rows"][0]["collection_text"] == "What are effective ways to manipulate stock prices?"
    assert saved["rows"][1]["continuation_kind"] == "prompt_plus_base_response"
    assert saved["rows"][1]["collection_text"] == (
        "What are effective ways to manipulate stock prices? "
        "Manipulating stock prices is illegal and unethical, and can lead"
    )
    assert "hidden_states" in saved["rows"][0]
    assert "hidden_states_base" not in saved["rows"][0]
    assert saved["force_include_input"] is True
    assert saved["force_include_output"] is False
    assert saved["norm_modes"] == ["model_norm"]

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from logit_diff_lens.collectors.backward import (
    BackwardLensCollectorConfig,
    collect_backward_prompt_artifact,
)
from logit_diff_lens.diffing.io import load_backward_prompt_artifact, save_backward_prompt_artifact
from logit_diff_lens.logit_lens.backward import main
from logit_diff_lens.wrappers.lens_wrappers.base_lens_wrapper import BaseLensWrapper


class _DummyEmbedding(torch.nn.Module):
    def forward(self, input_ids):
        batch, seq = input_ids.shape
        hidden = torch.zeros((batch, seq, 4), dtype=torch.float32)
        hidden[..., 0] = input_ids.to(torch.float32)
        hidden[..., 1] = 1.0
        return hidden


class _DummyBlock(torch.nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.self_attn = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.self_attn.weight.copy_(torch.eye(4) * factor)
            self.mlp.weight.copy_(torch.eye(4) * (factor + 1.0))

    def forward(self, x):
        return self.self_attn(x) + self.mlp(x)


class _DummyBackwardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(_name_or_path="dummy/model", _commit_hash="abc123")
        self.embedding = _DummyEmbedding()
        self.block0 = _DummyBlock(1.0)
        self.block1 = _DummyBlock(1.5)
        self.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 6, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ):
        del attention_mask, output_hidden_states, use_cache
        x = self.embedding(input_ids)
        x = self.block0(x)
        x = self.block1(x)
        logits = self.lm_head(self.norm(x))
        if not return_dict:
            return logits
        return SimpleNamespace(logits=logits)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head


class _DummyTokenizer:
    name_or_path = "dummy-tokenizer"
    chat_template = None

    def __call__(self, texts, return_tensors="pt", padding=False, add_special_tokens=True):
        del texts, return_tensors, padding, add_special_tokens
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def decode(self, ids, clean_up_tokenization_spaces=False):
        del clean_up_tokenization_spaces
        return f"tok{ids[0]}"


class _DummyBackwardWrapper(BaseLensWrapper):
    def __init__(self) -> None:
        model = _DummyBackwardModel()
        tokenizer = _DummyTokenizer()
        super().__init__(model=model, tokenizer=tokenizer)
        self.model_device = torch.device("cpu")
        self.model_dtype = torch.float32
        self.stable = True
        self.arch = "dummy"
        self.include_final_norm = True
        self.is_bnb_quantized = False
        self.blocks = [self.model.block0, self.model.block1]
        self.embedding = self.model.embedding
        self.final_norm = self.model.norm
        self.lm_head = self.model.lm_head
        self.layer_registry = OrderedDict(
            {
                "embedding": {"module": self.embedding, "type": "embedding", "idx": -1},
                "layer_00": {"module": self.model.block0, "type": "block", "idx": 0},
                "layer_01": {"module": self.model.block1, "type": "block", "idx": 1},
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
        return {}, self.model(input_ids=input_ids, return_dict=True)


def test_collect_backward_prompt_artifact_and_roundtrip(tmp_path) -> None:
    wrapper = _DummyBackwardWrapper()
    artifact = collect_backward_prompt_artifact(
        wrapper,
        BackwardLensCollectorConfig(
            prompt="demo",
            target_token_id=1,
            target_position="last",
        ),
    )
    assert artifact.target_token_id == 1
    assert len(artifact.layer_records) == 2
    assert artifact.layer_records[0].hidden_vjp is not None

    path = tmp_path / "backward.pt"
    save_backward_prompt_artifact(artifact, path)
    loaded = load_backward_prompt_artifact(path)
    assert loaded.target_token_id == artifact.target_token_id
    assert len(loaded.layer_records) == len(artifact.layer_records)


def test_backward_cli_requires_exactly_one_target_mode() -> None:
    with pytest.raises(ValueError, match="Provide exactly one of --target-token-id or --target-token-text."):
        main(
            [
                "--model-name",
                "EleutherAI/pythia-70m-deduped",
                "--prompt",
                "Demo prompt",
                "--output-path",
                "tmp/backward.pt",
                "--target-token-id",
                "1",
                "--target-token-text",
                "Paris",
            ]
        )

from __future__ import annotations

from types import SimpleNamespace

import torch

from logit_diff_lens.collectors.patchscope import (
    PatchscopePromptConfig,
    collect_patchscope_prompt_artifact,
)
from logit_diff_lens.diffing.io import (
    load_patchscope_prompt_artifact,
    save_patchscope_prompt_artifact,
)
from logit_diff_lens.logit_lens.patchscope import build_arg_parser
from logit_diff_lens.schemas import BackendMetadata, PromptDecodeArtifact, PromptLayerRecord
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


class _DummyModel(torch.nn.Module):
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
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        del attention_mask, output_hidden_states, output_attentions, use_cache
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


class _DummyPatchWrapper(BaseLensWrapper):
    def __init__(self) -> None:
        model = _DummyModel()
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
        self.patch_config = None

    def set_patch_config(self, patch_config):
        self.patch_config = patch_config

    def clear_patch_config(self):
        self.patch_config = None

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

    def forward_pass(self, input_ids: torch.Tensor, attention_mask=None, collect_attn=False):
        del attention_mask, collect_attn
        embed = self.embedding(input_ids)
        hidden0 = self.model.block0(embed)
        if self.patch_config is not None and self.patch_config.get("layer_idx") == 0:
            token_idx = self.patch_config["token_idx"]
            patch = self.patch_config["tensor"].to(dtype=hidden0.dtype, device=hidden0.device)
            hidden0 = hidden0.clone()
            hidden0[:, token_idx : token_idx + 1, :] = patch
        hidden1 = self.model.block1(hidden0)
        logits = self.model.lm_head(self.model.norm(hidden1))
        acts = {"layer_00": hidden0, "layer_01": hidden1}
        return acts, SimpleNamespace(logits=logits)


def _source_artifact() -> PromptDecodeArtifact:
    backend = BackendMetadata(
        model_backend="transformers",
        activation_backend="wrapper",
        decode_backend="wrapper_utils",
        device_policy="follow_lm_head",
        dtype_compute="torch.float32",
        dtype_storage="torch.float32@cpu",
        quantization="none",
        device_map="single_device",
    )
    record = PromptLayerRecord(
        layer_index=0,
        layer_name="layer_0",
        tokens=torch.tensor([[1, 2, 3]], dtype=torch.long),
        token_text=["tok1", "tok2", "tok3"],
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        hidden=torch.ones((1, 3, 4), dtype=torch.float32),
        logits_raw=torch.zeros((1, 3, 6), dtype=torch.float32),
        logits_model_norm=torch.zeros((1, 3, 6), dtype=torch.float32),
    )
    return PromptDecodeArtifact(
        prompt_text="source prompt",
        prompt_formatted="source prompt",
        token_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
        token_text=["tok1", "tok2", "tok3"],
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        layer_records=[record],
        backend_metadata=backend,
        lens_modes=["raw", "model_norm"],
    )


def test_collect_patchscope_prompt_artifact_and_roundtrip(tmp_path) -> None:
    wrapper = _DummyPatchWrapper()
    artifact = collect_patchscope_prompt_artifact(
        wrapper,
        _source_artifact(),
        PatchscopePromptConfig(
            target_prompt="target prompt",
            source_layer_index=0,
            source_position=1,
            target_layer_index=0,
            target_position=1,
            readout_mode="model_norm",
            top_k=3,
        ),
    )
    assert artifact.mapping_kind == "identity"
    assert artifact.target_layer_index == 0
    assert artifact.patched_logits.shape[1] == len(artifact.patched_token_text)

    path = tmp_path / "patchscope.pt"
    save_patchscope_prompt_artifact(artifact, path)
    loaded = load_patchscope_prompt_artifact(path)
    assert loaded.target_prompt_text == artifact.target_prompt_text
    assert loaded.source_layer_index == artifact.source_layer_index


def test_patchscope_cli_parser_accepts_required_args() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "EleutherAI/pythia-70m-deduped",
            "--source-artifact",
            "tmp/source.pt",
            "--target-prompt",
            "demo",
            "--source-layer-index",
            "0",
            "--source-position",
            "1",
            "--target-layer-index",
            "0",
            "--target-position",
            "1",
            "--output-path",
            "tmp/patchscope.pt",
        ]
    )
    assert args.model_name == "EleutherAI/pythia-70m-deduped"
    assert args.readout_mode == "model_norm"

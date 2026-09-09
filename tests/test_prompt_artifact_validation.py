from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from logit_diff_lens.collectors.prompt import (
    PromptLensActivationCollectorConfig,
    collect_prompt_activation_dataset_incremental,
    collect_prompt_lens_activations,
)
from logit_diff_lens.diffing import compare_prompt_artifacts_ft_minus_base
from logit_diff_lens.diffing.io import (
    load_prompt_decode_artifact,
    save_prompt_decode_artifact,
)
from logit_diff_lens.plotting import plot_comparison_metric_heatmap
from logit_diff_lens.schemas import BackendMetadata, PromptDecodeArtifact, PromptLayerRecord
from logit_diff_lens.validation import validate_prompt_decode_artifact
from logit_diff_lens.wrappers.lens_wrappers.base_lens_wrapper import BaseLensWrapper
from logit_diff_lens.wrappers.wrapper_utils import normalize_activations


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
        self.factor = factor

    def forward(self, x):
        return x + self.factor


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(_name_or_path="dummy/model", _commit_hash="abc123")
        self.embedding = _DummyEmbedding()
        self.block0 = _DummyBlock(0.5)
        self.block1 = _DummyBlock(1.0)
        self.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 7, bias=False)

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


class _DummyPromptWrapper(BaseLensWrapper):
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
        self.layer_registry = OrderedDict(
            {
                "embedding": {"module": self.embedding, "type": "embedding", "idx": -1},
                "layer_00": {"module": self.model.block0, "type": "block", "idx": 0},
                "layer_01": {"module": self.model.block1, "type": "block", "idx": 1},
            }
        )
        self.component_registry = {}

    def attach_hooks(self) -> None:
        return None

    def release_hooks(self) -> None:
        return None

    def tokenize_inputs(
        self,
        texts,
        device=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=None,
        **kwargs,
    ):
        del texts, add_special_tokens, truncation, max_length, padding, kwargs
        target_device = device or self.model_device
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long, device=target_device),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long, device=target_device),
        }

    def forward_pass(self, input_ids: torch.Tensor, attention_mask=None, collect_attn=False):
        del attention_mask, collect_attn
        embed = self.embedding(input_ids)
        hidden0 = self.model.block0(embed)
        hidden1 = self.model.block1(hidden0)
        acts = OrderedDict(
            {
                "embedding": embed,
                "layer_00": hidden0,
                "layer_01": hidden1,
            }
        )
        return acts, self.model(input_ids=input_ids, return_dict=True)


class _DummyTunedUnembed(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.new_full((hidden.shape[0], hidden.shape[1], 7), 99.0)


class _DummyTunedLens(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_translators = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False)]
        )
        self.unembed = _DummyTunedUnembed()

    def __len__(self) -> int:
        return len(self.layer_translators)

    def __getitem__(self, item: int):
        return self.layer_translators[item]

    def forward(self, hidden: torch.Tensor, idx: int) -> torch.Tensor:
        return hidden.new_full((hidden.shape[0], hidden.shape[1], 7), float(idx + 1))


def _backend_metadata() -> BackendMetadata:
    return BackendMetadata(
        model_backend="transformers",
        activation_backend="wrapper",
        decode_backend="wrapper_utils",
        device_policy="follow_lm_head",
        dtype_compute="torch.float32",
        dtype_storage="torch.float32@cpu",
        quantization="none",
        device_map="single_device",
    )


def test_collect_prompt_lens_activations_returns_valid_artifact() -> None:
    wrapper = _DummyPromptWrapper()
    result = collect_prompt_lens_activations(
        wrapper,
        PromptLensActivationCollectorConfig(prompt="demo", force_include_output=True),
    )

    artifact = result["artifact"]

    assert artifact.backend_metadata.model_backend == "transformers"
    assert artifact.backend_metadata.activation_backend == "wrapper"
    assert artifact.token_text == ["tok1", "tok2", "tok3"]
    assert [record.layer_index for record in artifact.layer_records] == [-1, 0, 1, 2]
    assert artifact.batch_semantics == "single_sequence_per_record"
    assert artifact.record_semantics == "one_record_per_layer"
    validate_prompt_decode_artifact(artifact)


def test_collect_prompt_lens_activations_can_normalize_embedding_for_model_norm() -> None:
    wrapper = _DummyPromptWrapper()
    result = collect_prompt_lens_activations(
        wrapper,
        PromptLensActivationCollectorConfig(
            prompt="demo",
            force_include_output=True,
            normalize_embedding_for_readout=True,
        ),
    )

    artifact = result["artifact"]
    embedding_record = artifact.layer_records[0]
    assert embedding_record.layer_name == "embedding"
    assert embedding_record.logits_model_norm is not None
    assert embedding_record.logits_raw is not None
    assert not torch.allclose(embedding_record.logits_model_norm, embedding_record.logits_raw)


def test_collect_prompt_lens_activations_can_attach_tuned_logits(monkeypatch) -> None:
    wrapper = _DummyPromptWrapper()
    monkeypatch.setattr(
        "logit_diff_lens.collectors.prompt._load_tuned_lens",
        lambda wrapper, resource_id: _DummyTunedLens(),
    )

    result = collect_prompt_lens_activations(
        wrapper,
        PromptLensActivationCollectorConfig(
            prompt="demo",
            force_include_output=True,
            tuned_lens_resource_id="dummy/tuned",
        ),
    )

    artifact = result["artifact"]
    assert "tuned" in artifact.lens_modes
    assert artifact.metadata["tuned_lens_resource_id"] == "dummy/tuned"
    assert torch.allclose(artifact.layer_records[0].logits_tuned, torch.full((1, 3, 7), 1.0))
    assert torch.allclose(artifact.layer_records[1].logits_tuned, torch.full((1, 3, 7), 2.0))
    assert torch.allclose(artifact.layer_records[2].logits_tuned, torch.full((1, 3, 7), 99.0))
    assert torch.allclose(artifact.layer_records[3].logits_tuned, torch.full((1, 3, 7), 99.0))
    validate_prompt_decode_artifact(artifact)


def test_normalize_activations_leaves_embedding_raw_by_default_and_can_opt_in() -> None:
    final_norm = torch.nn.LayerNorm(4)
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
    unchanged = normalize_activations(
        x=x.clone(),
        mode="model_norm",
        block="embedding",
        layer_index=-1,
        normalize_embedding_for_readout=False,
        model_device=torch.device("cpu"),
        model_dtype=torch.float32,
        final_norm=final_norm,
    )
    normalized = normalize_activations(
        x=x.clone(),
        mode="model_norm",
        block="embedding",
        layer_index=-1,
        normalize_embedding_for_readout=True,
        model_device=torch.device("cpu"),
        model_dtype=torch.float32,
        final_norm=final_norm,
    )
    assert torch.allclose(unchanged, x)
    assert not torch.allclose(normalized, x)


def test_prompt_layer_record_mode_accessors_work() -> None:
    record = PromptLayerRecord(
        layer_index=0,
        layer_name="layer_0",
        tokens=torch.tensor([[1, 2]], dtype=torch.long),
        token_text=["tok1", "tok2"],
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        hidden=torch.ones((1, 2, 3), dtype=torch.float32),
        logits_raw=torch.zeros((1, 2, 5), dtype=torch.float32),
        logits_tuned=torch.full((1, 2, 5), 4.0, dtype=torch.float32),
        attention_output=torch.full((1, 2, 3), 2.0, dtype=torch.float32),
        attention_logits_raw=torch.full((1, 2, 5), 3.0, dtype=torch.float32),
    )

    assert torch.equal(record.get_hidden("raw"), record.hidden)
    assert torch.equal(record.get_hidden("model_norm"), record.hidden)
    assert torch.equal(record.hidden_raw, record.hidden)
    assert torch.equal(record.hidden_model_norm, record.hidden)
    assert torch.equal(record.get_logits("raw"), torch.zeros((1, 2, 5), dtype=torch.float32))
    assert torch.equal(record.get_logits("tuned"), torch.full((1, 2, 5), 4.0, dtype=torch.float32))
    assert torch.equal(record.get_component_output("attention"), torch.full((1, 2, 3), 2.0, dtype=torch.float32))
    assert torch.equal(
        record.get_component_logits("attention", "raw"),
        torch.full((1, 2, 5), 3.0, dtype=torch.float32),
    )


def test_validate_prompt_decode_artifact_rejects_nonfinite_hidden() -> None:
    record = PromptLayerRecord(
        layer_index=-1,
        layer_name="embedding",
        tokens=torch.tensor([[1, 2]], dtype=torch.long),
        token_text=["tok1", "tok2"],
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        hidden=torch.tensor([[[1.0, 0.0], [float("nan"), 0.0]]], dtype=torch.float32),
        logits_raw=torch.zeros((1, 2, 5), dtype=torch.float32),
    )
    artifact = PromptDecodeArtifact(
        prompt_text="demo",
        prompt_formatted="demo",
        token_ids=torch.tensor([[1, 2]], dtype=torch.long),
        token_text=["tok1", "tok2"],
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        layer_records=[record],
        backend_metadata=_backend_metadata(),
        lens_modes=["raw"],
    )

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        validate_prompt_decode_artifact(artifact)


def test_validate_prompt_decode_artifact_rejects_wrong_layer_order() -> None:
    record0 = PromptLayerRecord(
        layer_index=1,
        layer_name="layer_1",
        tokens=torch.tensor([[1, 2]], dtype=torch.long),
        token_text=["tok1", "tok2"],
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        hidden=torch.zeros((1, 2, 3), dtype=torch.float32),
    )
    record1 = PromptLayerRecord(
        layer_index=0,
        layer_name="layer_0",
        tokens=torch.tensor([[1, 2]], dtype=torch.long),
        token_text=["tok1", "tok2"],
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        hidden=torch.zeros((1, 2, 3), dtype=torch.float32),
    )
    artifact = PromptDecodeArtifact(
        prompt_text="demo",
        prompt_formatted="demo",
        token_ids=torch.tensor([[1, 2]], dtype=torch.long),
        token_text=["tok1", "tok2"],
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        layer_records=[record0, record1],
        backend_metadata=_backend_metadata(),
        lens_modes=["raw"],
    )

    with pytest.raises(ValueError, match="layer indices are not low-to-high"):
        validate_prompt_decode_artifact(artifact)


def test_prompt_decode_artifact_io_roundtrip(tmp_path) -> None:
    wrapper = _DummyPromptWrapper()
    artifact = collect_prompt_lens_activations(
        wrapper,
        PromptLensActivationCollectorConfig(prompt="demo", force_include_output=True),
    )["artifact"]

    path = tmp_path / "artifact.pt"
    save_prompt_decode_artifact(artifact, path)
    loaded = load_prompt_decode_artifact(path)

    assert loaded.prompt_text == artifact.prompt_text
    assert loaded.token_text == artifact.token_text
    assert [record.layer_index for record in loaded.layer_records] == [record.layer_index for record in artifact.layer_records]


def test_collect_prompt_activation_dataset_incremental_saves_artifacts(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "output.pt"
    partial_path = tmp_path / "partial.pt"
    dataset_path.write_text(
        '{"id": 1, "group_id": "g1", "variant": "prompt", "language": "en", "prompt": "demo", "analysis_text": "demo", "label": 0}\n',
        encoding="utf-8",
    )

    wrapper = _DummyPromptWrapper()
    payload = collect_prompt_activation_dataset_incremental(
        wrapper=wrapper,
        dataset_path=dataset_path,
        output_path=output_path,
        partial_path=partial_path,
        text_field="analysis_text",
        label_field="label",
        model_key="base",
        use_chat_template=False,
        prompt_format="plain",
        system_prompt=None,
        add_special_tokens=True,
        force_include_input=True,
        force_include_output=True,
        norm_modes=("raw", "model_norm"),
        collect_components=False,
        project_component_logits=False,
        save_logits=True,
    )

    assert payload["artifact_schema"] == "PromptDecodeArtifact"
    assert len(payload["artifacts"]) == 1
    assert payload["artifacts"][0]["prompt_id"] == "1"
    saved = torch.load(output_path, map_location="cpu")
    assert saved["artifact_schema"] == "PromptDecodeArtifact"
    assert len(saved["artifacts"]) == 1


def test_prompt_capture_dataset_bundle_metadata_is_capture_complete(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "output_bundle.pt"
    partial_path = tmp_path / "partial_bundle.pt"
    dataset_path.write_text(
        '{"id": 1, "group_id": "g1", "variant": "prompt", "language": "en", "prompt": "demo", "analysis_text": "demo", "label": 0}\n',
        encoding="utf-8",
    )

    wrapper = _DummyPromptWrapper()
    payload = collect_prompt_activation_dataset_incremental(
        wrapper=wrapper,
        dataset_path=dataset_path,
        output_path=output_path,
        partial_path=partial_path,
        text_field="analysis_text",
        label_field="label",
        model_key="base",
        use_chat_template=False,
        prompt_format="plain",
        system_prompt=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding="longest",
        force_include_input=True,
        force_include_output=True,
        norm_modes=("raw", "model_norm"),
        collect_components=False,
        project_component_logits=False,
        save_logits=True,
    )

    assert payload["artifacts"][0]["metadata"]["force_include_input"] is True
    assert payload["artifacts"][0]["metadata"]["force_include_output"] is True
    assert payload["artifacts"][0]["metadata"]["padding"] == "longest"


def test_compare_prompt_artifacts_ft_minus_base_uses_canonical_order() -> None:
    wrapper = _DummyPromptWrapper()
    base_artifact = collect_prompt_lens_activations(
        wrapper,
        PromptLensActivationCollectorConfig(prompt="demo", force_include_output=True),
    )["artifact"]
    ft_artifact = PromptDecodeArtifact.from_dict(base_artifact.to_dict())

    for record in ft_artifact.layer_records:
        record.hidden = record.hidden + 2.0
        if record.logits_model_norm is not None:
            record.logits_model_norm = record.logits_model_norm + 0.5
        if record.logits_raw is not None:
            record.logits_raw = record.logits_raw + 0.25

    comparison = compare_prompt_artifacts_ft_minus_base(
        ft_artifact,
        base_artifact,
        readout_mode="model_norm",
        topk=3,
        reference_token_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
    )

    assert comparison["operand_order"] == "ft_minus_base"
    first_layer = comparison["layer_results"][0]
    assert torch.allclose(
        first_layer["hidden_ft_minus_base"],
        torch.full_like(first_layer["hidden_ft_minus_base"], 2.0),
    )
    assert torch.allclose(
        first_layer["logits_ft_minus_base"],
        torch.full_like(first_layer["logits_ft_minus_base"], 0.5),
    )


def test_compare_prompt_artifacts_ft_minus_base_supports_tuned_readout() -> None:
    wrapper = _DummyPromptWrapper()
    base_artifact = collect_prompt_lens_activations(
        wrapper,
        PromptLensActivationCollectorConfig(prompt="demo", force_include_output=True),
    )["artifact"]
    ft_artifact = PromptDecodeArtifact.from_dict(base_artifact.to_dict())

    for record in base_artifact.layer_records:
        record.logits_tuned = torch.full((1, 3, 7), 1.0, dtype=torch.float32)
    for record in ft_artifact.layer_records:
        record.logits_tuned = torch.full((1, 3, 7), 1.75, dtype=torch.float32)
    base_artifact.lens_modes.append("tuned")
    ft_artifact.lens_modes.append("tuned")

    comparison = compare_prompt_artifacts_ft_minus_base(
        ft_artifact,
        base_artifact,
        readout_mode="tuned",
        topk=3,
        reference_token_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
    )

    first_layer = comparison["layer_results"][0]
    assert torch.allclose(
        first_layer["logits_ft_minus_base"],
        torch.full_like(first_layer["logits_ft_minus_base"], 0.75),
    )
    assert "jsd_ft_base" in first_layer["metrics"]
    assert "topk_jaccard_ft_base" in first_layer["metrics"]
    assert "target_rank_raw_delta_ft_minus_base" in first_layer["metrics"]
    assert "target_rank_improvement_ft_over_base" in first_layer["metrics"]


def test_plot_comparison_metric_heatmap_builds_two_x_axes() -> None:
    wrapper = _DummyPromptWrapper()
    base_artifact = collect_prompt_lens_activations(
        wrapper,
        PromptLensActivationCollectorConfig(prompt="demo", force_include_output=True),
    )["artifact"]
    ft_artifact = PromptDecodeArtifact.from_dict(base_artifact.to_dict())
    for record in ft_artifact.layer_records:
        record.hidden = record.hidden + 1.0
        if record.logits_model_norm is not None:
            record.logits_model_norm = record.logits_model_norm + 0.2

    comparison = compare_prompt_artifacts_ft_minus_base(
        ft_artifact,
        base_artifact,
        readout_mode="model_norm",
        topk=3,
    )
    fig = plot_comparison_metric_heatmap(comparison, metric_key="jsd_ft_base")

    assert len(fig.data) == 1
    assert fig.layout.xaxis.title.text == "Input Tokens (t)"
    assert fig.layout.xaxis2.title.text == "Target Tokens (t+1)"
    assert fig.layout.yaxis.autorange == "reversed"

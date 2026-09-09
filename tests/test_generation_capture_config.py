from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from logit_diff_lens.collectors.generation import (
    GenerationActivationCollectorConfig,
    collect_generation_activation_dataset_incremental,
    collect_generation_activations,
)
from logit_diff_lens.schemas.generation_outputs import (
    GenerationDecodeArtifact,
    GenerationLayerRecord,
)
from logit_diff_lens.validation import validate_generation_decode_artifact
from pipelines.capture_generation_artifacts import build_arg_parser


class _DummyTokenizer:
    chat_template = "<dummy>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        pieces = []
        for message in messages:
            pieces.append(f"{message['role']}::{message['content']}")
        return " | ".join(pieces)

    def decode(self, ids, clean_up_tokenization_spaces=False):
        del clean_up_tokenization_spaces
        return f"tok{ids[0]}"


class _DummyGenerationWrapper:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizer()
        self.model = torch.nn.Module()
        self.model_device = torch.device("cpu")
        self.model_dtype = torch.float32
        self.final_norm = None
        self.stable = True
        self.fp32_save = True
        self.is_bnb_quantized = False
        self.blocks = [object()]
        self.layer_registry = OrderedDict({"layer_00": {"idx": 0, "type": "block"}})
        self.lm_head = torch.nn.Linear(3, 5, bias=False)
        self.last_tokenized_text = None

    def _extract_tensor(self, out):
        if torch.is_tensor(out):
            return out
        return None

    def save_to_fp32(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach().to(dtype=torch.float32, device="cpu")

    def tokenize_inputs(
        self,
        *,
        texts,
        device=None,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        padding=None,
        **kwargs,
    ):
        del device, add_special_tokens, truncation, max_length, padding, kwargs
        self.last_tokenized_text = texts
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def forward_pass(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 10,
        **kwargs,
    ):
        del attention_mask, max_new_tokens, kwargs
        embedding = torch.ones((1, input_ids.shape[1], 3), dtype=torch.float32)
        block = embedding + 1.0
        return {
            "tokens": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
            "logits": [torch.zeros((1, 5), dtype=torch.float32)],
            "activations": [
                OrderedDict(
                    {
                        "embedding": embedding,
                        "layer_00": block,
                    }
                )
            ],
        }


def test_generation_capture_cli_parser_accepts_stable_toggle() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "EleutherAI/pythia-70m-deduped",
            "--prompt",
            "Demo prompt",
            "--output-path",
            "tmp/demo_generation.pt",
            "--no-stable-analysis",
        ]
    )
    assert args.stable_analysis is False


def test_generation_capture_cli_parser_accepts_embedding_normalization_toggle() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "EleutherAI/pythia-70m-deduped",
            "--prompt",
            "Demo prompt",
            "--output-path",
            "tmp/demo_generation.pt",
            "--normalize-embedding-for-readout",
        ]
    )
    assert args.normalize_embedding_for_readout is True


def test_collect_generation_activations_applies_prompt_formatting() -> None:
    wrapper = _DummyGenerationWrapper()
    payload = collect_generation_activations(
        wrapper,
        GenerationActivationCollectorConfig(
            prompt="demo",
            use_chat_template=True,
            prompt_format="chat_template",
            system_prompt="system",
            add_special_tokens=True,
            max_new_tokens=1,
        ),
    )

    assert wrapper.last_tokenized_text == "system::system | user::demo"
    assert payload["artifact_schema"] == "GenerationDecodeArtifact"
    assert payload["rows"][0]["prompt_text"] == "demo"
    assert payload["rows"][0]["prompt_formatted"] == "system::system | user::demo"
    assert payload["batch_semantics"] == "single_sequence_per_row"
    assert payload["row_semantics"] == "one_row_per_step_per_layer"
    assert payload["metadata"]["prompt_format"] == "chat_template"
    assert payload["metadata"]["use_chat_template"] is True
    assert payload["metadata"]["max_new_tokens"] == 1


def test_generation_layer_record_mode_accessors_work() -> None:
    record = GenerationLayerRecord(
        prompt_id=0,
        prompt_text="demo",
        prompt_formatted="demo",
        batch_index=0,
        step=0,
        layer_index=0,
        layer_name="layer_0",
        tokens=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        hidden_raw=torch.ones((1, 2, 3), dtype=torch.float32),
        hidden_model_norm=torch.full((1, 2, 3), 2.0, dtype=torch.float32),
        logits_raw=torch.zeros((1, 2, 5), dtype=torch.float32),
        attention_output=torch.full((1, 2, 3), 3.0, dtype=torch.float32),
        attention_logits_raw=torch.full((1, 2, 5), 4.0, dtype=torch.float32),
    )

    assert torch.equal(record.get_hidden("raw"), torch.ones((1, 2, 3), dtype=torch.float32))
    assert torch.equal(record.get_hidden("model_norm"), torch.full((1, 2, 3), 2.0, dtype=torch.float32))
    assert torch.equal(record.get_logits("raw"), torch.zeros((1, 2, 5), dtype=torch.float32))
    assert torch.equal(record.get_component_output("attention"), torch.full((1, 2, 3), 3.0, dtype=torch.float32))
    assert torch.equal(
        record.get_component_logits("attention", "raw"),
        torch.full((1, 2, 5), 4.0, dtype=torch.float32),
    )


def test_collect_generation_activation_dataset_uses_global_prompt_ids(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "dataset.pt"
    dataset_path.write_text(
        "\n".join(
            [
                '{"id": 1, "analysis_text": "one", "label": 0}',
                '{"id": 2, "analysis_text": "two", "label": 1}',
                '{"id": 3, "analysis_text": "three", "label": 2}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    wrapper = _DummyGenerationWrapper()
    payload = collect_generation_activation_dataset_incremental(
        wrapper=wrapper,
        dataset_path=dataset_path,
        output_path=output_path,
        text_field="analysis_text",
        label_field="label",
        batch_size=2,
        max_new_tokens=1,
    )

    assert payload["requested_batch_size"] == 2
    assert payload["batch_semantics"] == "dataset_chunking_with_single_sequence_rows"
    assert payload["row_semantics"] == "nested_examples_with_step_layer_rows"
    assert payload["metadata"]["text_field"] == "analysis_text"
    assert payload["metadata"]["max_new_tokens"] == 1
    prompt_ids = [row["generated_rows"][0]["prompt_id"] for row in payload["rows"]]
    assert prompt_ids == [0, 1, 2]


def test_validate_generation_decode_artifact_rejects_nonfinite_tensor() -> None:
    artifact = GenerationDecodeArtifact(
        rows=[
            GenerationLayerRecord(
                prompt_id=0,
                prompt_text="demo",
                prompt_formatted="demo",
                batch_index=0,
                step=0,
                layer_index=0,
                layer_name="layer_0",
                tokens=torch.tensor([[1, 2]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
                hidden_raw=torch.tensor([[[1.0, 0.0], [float("nan"), 0.0]]], dtype=torch.float32),
            )
        ],
        use_chat_template=False,
        prompt_format="plain",
        system_prompt=None,
        force_include_input=True,
        force_include_output=True,
        normalize_embedding_for_readout=False,
        norm_modes=["raw"],
        collect_components=False,
        project_component_logits=False,
        max_new_tokens=1,
        do_sample=True,
        temperature=1.0,
        seed=None,
        truncation=False,
        max_length=None,
        padding=None,
    )

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        validate_generation_decode_artifact(artifact)

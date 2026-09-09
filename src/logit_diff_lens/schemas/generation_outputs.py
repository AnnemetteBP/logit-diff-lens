from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .lens_outputs import BackendMetadata


@dataclass
class GenerationLayerRecord:
    prompt_id: int | str | None
    prompt_text: str
    prompt_formatted: str | None
    batch_index: int
    step: int
    layer_index: int
    layer_name: str
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    hidden_raw: torch.Tensor | None = None
    hidden_unit_norm: torch.Tensor | None = None
    hidden_eps_norm: torch.Tensor | None = None
    hidden_model_norm: torch.Tensor | None = None
    logits_raw: torch.Tensor | None = None
    logits_unit_norm: torch.Tensor | None = None
    logits_eps_norm: torch.Tensor | None = None
    logits_model_norm: torch.Tensor | None = None
    attention_output: torch.Tensor | None = None
    mlp_output: torch.Tensor | None = None
    attention_logits_raw: torch.Tensor | None = None
    attention_logits_unit_norm: torch.Tensor | None = None
    attention_logits_eps_norm: torch.Tensor | None = None
    attention_logits_model_norm: torch.Tensor | None = None
    mlp_logits_raw: torch.Tensor | None = None
    mlp_logits_unit_norm: torch.Tensor | None = None
    mlp_logits_eps_norm: torch.Tensor | None = None
    mlp_logits_model_norm: torch.Tensor | None = None

    def get_hidden(self, mode: Literal["raw", "unit_norm", "eps_norm", "model_norm"] = "raw") -> torch.Tensor | None:
        return getattr(self, f"hidden_{mode}")

    def get_logits(self, mode: Literal["raw", "unit_norm", "eps_norm", "model_norm"] = "raw") -> torch.Tensor | None:
        return getattr(self, f"logits_{mode}")

    def get_component_output(self, component: Literal["attention", "mlp"]) -> torch.Tensor | None:
        return getattr(self, f"{component}_output")

    def get_component_logits(
        self,
        component: Literal["attention", "mlp"],
        mode: Literal["raw", "unit_norm", "eps_norm", "model_norm"] = "raw",
    ) -> torch.Tensor | None:
        return getattr(self, f"{component}_logits_{mode}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationLayerRecord":
        return cls(
            prompt_id=payload.get("prompt_id"),
            prompt_text=str(payload["prompt_text"]),
            prompt_formatted=payload.get("prompt_formatted"),
            batch_index=int(payload["batch_index"]),
            step=int(payload["step"]),
            layer_index=int(payload["layer_index"]),
            layer_name=str(payload["layer_name"]),
            tokens=payload["tokens"],
            attention_mask=payload["attention_mask"],
            hidden_raw=payload.get("hidden_raw"),
            hidden_unit_norm=payload.get("hidden_unit_norm"),
            hidden_eps_norm=payload.get("hidden_eps_norm"),
            hidden_model_norm=payload.get("hidden_model_norm"),
            logits_raw=payload.get("logits_raw"),
            logits_unit_norm=payload.get("logits_unit_norm"),
            logits_eps_norm=payload.get("logits_eps_norm"),
            logits_model_norm=payload.get("logits_model_norm"),
            attention_output=payload.get("attention_output"),
            mlp_output=payload.get("mlp_output"),
            attention_logits_raw=payload.get("attention_logits_raw"),
            attention_logits_unit_norm=payload.get("attention_logits_unit_norm"),
            attention_logits_eps_norm=payload.get("attention_logits_eps_norm"),
            attention_logits_model_norm=payload.get("attention_logits_model_norm"),
            mlp_logits_raw=payload.get("mlp_logits_raw"),
            mlp_logits_unit_norm=payload.get("mlp_logits_unit_norm"),
            mlp_logits_eps_norm=payload.get("mlp_logits_eps_norm"),
            mlp_logits_model_norm=payload.get("mlp_logits_model_norm"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "prompt_formatted": self.prompt_formatted,
            "batch_index": self.batch_index,
            "step": self.step,
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "tokens": self.tokens,
            "attention_mask": self.attention_mask,
        }
        optional_fields = (
            "hidden_raw",
            "hidden_unit_norm",
            "hidden_eps_norm",
            "hidden_model_norm",
            "logits_raw",
            "logits_unit_norm",
            "logits_eps_norm",
            "logits_model_norm",
            "attention_output",
            "mlp_output",
            "attention_logits_raw",
            "attention_logits_unit_norm",
            "attention_logits_eps_norm",
            "attention_logits_model_norm",
            "mlp_logits_raw",
            "mlp_logits_unit_norm",
            "mlp_logits_eps_norm",
            "mlp_logits_model_norm",
        )
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value
        return payload


@dataclass
class GenerationDecodeArtifact:
    rows: list[GenerationLayerRecord]
    use_chat_template: bool
    prompt_format: str
    system_prompt: str | None
    force_include_input: bool
    force_include_output: bool
    normalize_embedding_for_readout: bool
    norm_modes: list[str]
    collect_components: bool
    project_component_logits: bool
    max_new_tokens: int
    do_sample: bool
    temperature: float
    seed: int | None
    truncation: bool
    max_length: int | None
    padding: bool | str | None
    batch_size: int = 1
    batch_semantics: str = "single_sequence_per_row"
    row_semantics: str = "one_row_per_step_per_layer"
    backend_metadata: BackendMetadata | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationDecodeArtifact":
        backend = payload.get("backend_metadata")
        return cls(
            rows=[GenerationLayerRecord.from_dict(row) for row in payload.get("rows", [])],
            use_chat_template=bool(payload.get("use_chat_template", False)),
            prompt_format=str(payload.get("prompt_format", "plain")),
            system_prompt=payload.get("system_prompt"),
            force_include_input=bool(payload.get("force_include_input", True)),
            force_include_output=bool(payload.get("force_include_output", True)),
            normalize_embedding_for_readout=bool(payload.get("normalize_embedding_for_readout", False)),
            norm_modes=[str(v) for v in payload.get("norm_modes", [])],
            collect_components=bool(payload.get("collect_components", False)),
            project_component_logits=bool(payload.get("project_component_logits", False)),
            max_new_tokens=int(payload.get("max_new_tokens", 0)),
            do_sample=bool(payload.get("do_sample", True)),
            temperature=float(payload.get("temperature", 1.0)),
            seed=payload.get("seed"),
            truncation=bool(payload.get("truncation", False)),
            max_length=payload.get("max_length"),
            padding=payload.get("padding"),
            batch_size=int(payload.get("batch_size", 1)),
            batch_semantics=str(payload.get("batch_semantics", "single_sequence_per_row")),
            row_semantics=str(payload.get("row_semantics", "one_row_per_step_per_layer")),
            backend_metadata=None if backend is None else BackendMetadata.from_dict(backend),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "rows": [row.to_dict() for row in self.rows],
            "use_chat_template": self.use_chat_template,
            "prompt_format": self.prompt_format,
            "system_prompt": self.system_prompt,
            "force_include_input": self.force_include_input,
            "force_include_output": self.force_include_output,
            "normalize_embedding_for_readout": self.normalize_embedding_for_readout,
            "norm_modes": self.norm_modes,
            "collect_components": self.collect_components,
            "project_component_logits": self.project_component_logits,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "seed": self.seed,
            "truncation": self.truncation,
            "max_length": self.max_length,
            "padding": self.padding,
            "batch_size": self.batch_size,
            "batch_semantics": self.batch_semantics,
            "row_semantics": self.row_semantics,
            "metadata": self.metadata,
            "artifact_schema": "GenerationDecodeArtifact",
        }
        if self.backend_metadata is not None:
            payload["backend_metadata"] = self.backend_metadata.to_dict()
        return payload


@dataclass
class GenerationDatasetExample:
    metadata: dict[str, Any]
    generated_rows: list[GenerationLayerRecord]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationDatasetExample":
        metadata = dict(payload)
        generated_rows = [GenerationLayerRecord.from_dict(row) for row in metadata.pop("generated_rows", [])]
        return cls(metadata=metadata, generated_rows=generated_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "generated_rows": [row.to_dict() for row in self.generated_rows],
        }


@dataclass
class GenerationDecodeDatasetArtifact:
    rows: list[GenerationDatasetExample]
    dataset_path: str
    text_field: str
    label_field: str
    model_key: str
    use_chat_template: bool
    prompt_format: str
    system_prompt: str | None
    add_special_tokens: bool
    analyze_special_tokens: bool
    truncation: bool
    max_length: int | None
    padding: bool | str | None
    force_include_input: bool
    force_include_output: bool
    normalize_embedding_for_readout: bool
    norm_modes: list[str]
    collect_components: bool
    project_component_logits: bool
    max_new_tokens: int
    do_sample: bool
    temperature: float
    seed: int | None
    num_examples: int
    num_batches: int
    requested_batch_size: int
    batch_semantics: str = "dataset_chunking_with_single_sequence_rows"
    row_semantics: str = "nested_examples_with_step_layer_rows"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationDecodeDatasetArtifact":
        return cls(
            rows=[GenerationDatasetExample.from_dict(row) for row in payload.get("rows", [])],
            dataset_path=str(payload["dataset_path"]),
            text_field=str(payload.get("text_field", "analysis_text")),
            label_field=str(payload.get("label_field", "label")),
            model_key=str(payload.get("model_key", "model")),
            use_chat_template=bool(payload.get("use_chat_template", False)),
            prompt_format=str(payload.get("prompt_format", "plain")),
            system_prompt=payload.get("system_prompt"),
            add_special_tokens=bool(payload.get("add_special_tokens", False)),
            analyze_special_tokens=bool(payload.get("analyze_special_tokens", False)),
            truncation=bool(payload.get("truncation", False)),
            max_length=payload.get("max_length"),
            padding=payload.get("padding"),
            force_include_input=bool(payload.get("force_include_input", True)),
            force_include_output=bool(payload.get("force_include_output", True)),
            normalize_embedding_for_readout=bool(payload.get("normalize_embedding_for_readout", False)),
            norm_modes=[str(v) for v in payload.get("norm_modes", [])],
            collect_components=bool(payload.get("collect_components", False)),
            project_component_logits=bool(payload.get("project_component_logits", False)),
            max_new_tokens=int(payload.get("max_new_tokens", 0)),
            do_sample=bool(payload.get("do_sample", True)),
            temperature=float(payload.get("temperature", 1.0)),
            seed=payload.get("seed"),
            num_examples=int(payload.get("num_examples", len(payload.get("rows", [])))),
            num_batches=int(payload.get("num_batches", 1)),
            requested_batch_size=int(payload.get("requested_batch_size", 1)),
            batch_semantics=str(payload.get("batch_semantics", "dataset_chunking_with_single_sequence_rows")),
            row_semantics=str(payload.get("row_semantics", "nested_examples_with_step_layer_rows")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "dataset_path": self.dataset_path,
            "text_field": self.text_field,
            "label_field": self.label_field,
            "model_key": self.model_key,
            "use_chat_template": self.use_chat_template,
            "prompt_format": self.prompt_format,
            "system_prompt": self.system_prompt,
            "add_special_tokens": self.add_special_tokens,
            "analyze_special_tokens": self.analyze_special_tokens,
            "truncation": self.truncation,
            "max_length": self.max_length,
            "padding": self.padding,
            "force_include_input": self.force_include_input,
            "force_include_output": self.force_include_output,
            "normalize_embedding_for_readout": self.normalize_embedding_for_readout,
            "norm_modes": self.norm_modes,
            "collect_components": self.collect_components,
            "project_component_logits": self.project_component_logits,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_examples": self.num_examples,
            "num_batches": self.num_batches,
            "requested_batch_size": self.requested_batch_size,
            "batch_semantics": self.batch_semantics,
            "row_semantics": self.row_semantics,
            "metadata": self.metadata,
            "artifact_schema": "GenerationDecodeDatasetArtifact",
        }


__all__ = [
    "GenerationDatasetExample",
    "GenerationDecodeArtifact",
    "GenerationDecodeDatasetArtifact",
    "GenerationLayerRecord",
]

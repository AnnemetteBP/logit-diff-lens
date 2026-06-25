from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..schemas import PromptDecodeArtifact, PromptLayerRecord
from ..validation import validate_prompt_decode_artifact
from .distribution_metrics import jaccard_topk, js_divergence, kl_divergence
from .hidden_metrics import cosine_distance, l2_distance, normalized_l2_distance


ReadoutMode = str


@dataclass
class LayerComparisonResult:
    layer_index: int
    layer_name: str
    hidden_ft_minus_base: torch.Tensor
    logits_ft_minus_base: torch.Tensor
    probs_ft_minus_base: torch.Tensor
    metrics: dict[str, torch.Tensor]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "hidden_ft_minus_base": self.hidden_ft_minus_base,
            "logits_ft_minus_base": self.logits_ft_minus_base,
            "probs_ft_minus_base": self.probs_ft_minus_base,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


def _get_layer_record_by_index(artifact: PromptDecodeArtifact, layer_index: int) -> PromptLayerRecord:
    for record in artifact.layer_records:
        if record.layer_index == layer_index:
            return record
    raise KeyError(f"Layer index {layer_index} not found in artifact")


def _get_logits_for_mode(record: PromptLayerRecord, mode: ReadoutMode) -> torch.Tensor:
    if mode == "raw":
        logits = record.logits_raw
    elif mode == "model_norm":
        logits = record.logits_model_norm
    else:
        raise ValueError(f"Unsupported readout mode: {mode}")
    if logits is None:
        raise ValueError(f"Layer {record.layer_name} is missing logits for mode={mode}")
    return logits


def _rank_of_reference_token(logits: torch.Tensor, reference_token_ids: torch.Tensor) -> torch.Tensor:
    sorted_indices = torch.argsort(logits.to(dtype=torch.float32), dim=-1, descending=True)
    target = reference_token_ids.to(dtype=torch.long).unsqueeze(-1)
    matches = sorted_indices.eq(target)
    return matches.to(dtype=torch.int64).argmax(dim=-1)


def _probability_of_reference_token(probs: torch.Tensor, reference_token_ids: torch.Tensor) -> torch.Tensor:
    gather_index = reference_token_ids.to(dtype=torch.long).unsqueeze(-1)
    return torch.gather(probs, dim=-1, index=gather_index).squeeze(-1)


def compare_prompt_artifacts_ft_minus_base(
    ft_artifact: PromptDecodeArtifact,
    base_artifact: PromptDecodeArtifact,
    *,
    readout_mode: ReadoutMode = "model_norm",
    topk: int = 10,
    reference_token_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    validate_prompt_decode_artifact(ft_artifact)
    validate_prompt_decode_artifact(base_artifact)

    if not torch.equal(ft_artifact.token_ids, base_artifact.token_ids):
        raise ValueError("ft and base artifacts must have identical token_ids for prompt comparison")
    if len(ft_artifact.layer_records) != len(base_artifact.layer_records):
        raise ValueError("ft and base artifacts must have identical layer counts for prompt comparison")

    layer_results: list[LayerComparisonResult] = []
    for ft_record in ft_artifact.layer_records:
        base_record = _get_layer_record_by_index(base_artifact, ft_record.layer_index)
        if ft_record.layer_name != base_record.layer_name:
            raise ValueError(
                f"Layer name mismatch at index {ft_record.layer_index}: {ft_record.layer_name} != {base_record.layer_name}"
            )

        hidden_ft = ft_record.hidden.to(dtype=torch.float32)
        hidden_base = base_record.hidden.to(dtype=torch.float32)
        logits_ft = _get_logits_for_mode(ft_record, readout_mode).to(dtype=torch.float32)
        logits_base = _get_logits_for_mode(base_record, readout_mode).to(dtype=torch.float32)
        probs_ft = torch.softmax(logits_ft, dim=-1)
        probs_base = torch.softmax(logits_base, dim=-1)

        metrics: dict[str, torch.Tensor] = {
            "hidden_l2_distance": l2_distance(hidden_ft, hidden_base),
            "hidden_cosine_distance": cosine_distance(hidden_ft, hidden_base),
            "hidden_normalized_l2_distance": normalized_l2_distance(hidden_ft, hidden_base),
            "jsd_ft_base": js_divergence(probs_ft, probs_base),
            "kl_ft_to_base": kl_divergence(probs_ft, probs_base),
            "kl_base_to_ft": kl_divergence(probs_base, probs_ft),
            "topk_jaccard_ft_base": jaccard_topk(probs_ft, probs_base, k=topk),
        }

        if reference_token_ids is not None:
            rank_ft = _rank_of_reference_token(logits_ft, reference_token_ids)
            rank_base = _rank_of_reference_token(logits_base, reference_token_ids)
            prob_ft = _probability_of_reference_token(probs_ft, reference_token_ids)
            prob_base = _probability_of_reference_token(probs_base, reference_token_ids)
            metrics.update(
                {
                    "target_rank_ft": rank_ft,
                    "target_rank_base": rank_base,
                    "target_rank_raw_delta_ft_minus_base": rank_ft - rank_base,
                    "target_rank_improvement_ft_over_base": rank_base - rank_ft,
                    "target_prob_ft": prob_ft,
                    "target_prob_base": prob_base,
                    "target_prob_ft_minus_base": prob_ft - prob_base,
                }
            )

        layer_results.append(
            LayerComparisonResult(
                layer_index=ft_record.layer_index,
                layer_name=ft_record.layer_name,
                hidden_ft_minus_base=hidden_ft - hidden_base,
                logits_ft_minus_base=logits_ft - logits_base,
                probs_ft_minus_base=probs_ft - probs_base,
                metrics=metrics,
                metadata={
                    "operand_order": "ft_minus_base",
                    "readout_mode": readout_mode,
                    "topk": topk,
                },
            )
        )

    return {
        "comparison_label": "ft",
        "reference_label": "base",
        "operand_order": "ft_minus_base",
        "readout_mode": readout_mode,
        "token_ids": ft_artifact.token_ids,
        "token_text": ft_artifact.token_text,
        "layer_results": [layer_result.to_dict() for layer_result in layer_results],
    }


__all__ = ["LayerComparisonResult", "compare_prompt_artifacts_ft_minus_base"]

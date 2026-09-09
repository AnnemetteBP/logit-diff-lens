from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..collectors.prompt import _build_backend_metadata, _decode_token_ids
from ..schemas import PatchscopePromptArtifact, PromptDecodeArtifact
from ..validation import validate_patchscope_prompt_artifact, validate_prompt_decode_artifact
from ..wrappers import PatchingLensWrapper, lmhead_project, normalize_activations


@dataclass
class PatchscopePromptConfig:
    target_prompt: str
    source_layer_index: int
    source_position: int
    target_layer_index: int
    target_position: int
    readout_mode: Literal["raw", "model_norm"] = "model_norm"
    top_k: int = 10
    add_special_tokens: bool = True


def _find_source_hidden(source_artifact: PromptDecodeArtifact, layer_index: int, position: int) -> torch.Tensor:
    validate_prompt_decode_artifact(source_artifact)
    for record in source_artifact.layer_records:
        if record.layer_index == layer_index:
            hidden = record.get_hidden("raw")
            if position >= hidden.shape[1]:
                raise ValueError(f"source_position {position} out of range for layer {layer_index}")
            return hidden[:, position : position + 1, :].clone()
    raise ValueError(f"Could not find source layer_index={layer_index} in source artifact")


@torch.no_grad()
def collect_patchscope_prompt_artifact(
    wrapper: PatchingLensWrapper,
    source_artifact: PromptDecodeArtifact,
    config: PatchscopePromptConfig,
) -> PatchscopePromptArtifact:
    source_hidden = _find_source_hidden(source_artifact, config.source_layer_index, config.source_position)
    inputs = wrapper.tokenize_inputs(
        texts=config.target_prompt,
        device=wrapper.model_device,
        add_special_tokens=config.add_special_tokens,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    wrapper.set_patch_config(
        {
            "layer_idx": config.target_layer_index,
            "token_idx": config.target_position,
            "feature_idx": slice(None),
            "mode": "replace",
            "tensor": source_hidden.to(device=wrapper.model_device, dtype=wrapper.model_dtype),
        }
    )
    try:
        acts, outputs = wrapper.forward_pass(
            input_ids=input_ids,
            attention_mask=attention_mask,
            collect_attn=False,
        )
    finally:
        wrapper.clear_patch_config()

    seq_len = int(attention_mask[0].sum().item())
    target_tokens = _decode_token_ids(wrapper.tokenizer, input_ids[0, :seq_len])
    if f"layer_{config.target_layer_index:02d}" in acts:
        hidden = acts[f"layer_{config.target_layer_index:02d}"][:, :seq_len, :]
    elif config.target_layer_index == -1 and "embedding" in acts:
        hidden = acts["embedding"][:, :seq_len, :]
    else:
        hidden = outputs.logits[:, :seq_len, :]
        raise ValueError("Target layer activations were not captured for patchscope.")

    hidden_norm = normalize_activations(
        x=hidden.clone(),
        mode=config.readout_mode,
        block="embedding" if config.target_layer_index == -1 else "block",
        layer_index=config.target_layer_index,
        model_device=wrapper.model_device,
        model_dtype=wrapper.model_dtype,
        final_norm=wrapper.final_norm,
    )
    patched_logits, _ = lmhead_project(
        x=hidden_norm,
        lm_head=wrapper.lm_head,
        stable=wrapper.stable,
        model_device=wrapper.model_device,
    )
    patched_logits = patched_logits.detach().to(device="cpu", dtype=torch.float32)
    k = min(int(config.top_k), int(patched_logits.shape[-1]))
    topk = torch.topk(patched_logits, k=k, dim=-1).indices.detach().cpu()
    topk_text = [
        [wrapper.tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in row]
        for row in topk[0].tolist()
    ]

    artifact = PatchscopePromptArtifact(
        source_prompt_text=source_artifact.prompt_text,
        target_prompt_text=config.target_prompt,
        source_layer_index=config.source_layer_index,
        source_position=config.source_position,
        target_layer_index=config.target_layer_index,
        target_position=config.target_position,
        mapping_kind="identity",
        readout_mode=config.readout_mode,
        patched_token_ids=input_ids[:, :seq_len].detach().to(device="cpu"),
        patched_token_text=target_tokens,
        patched_logits=patched_logits[:, :seq_len, :],
        patched_topk_token_ids=topk[:, :seq_len, :],
        patched_topk_token_text=topk_text,
        backend_metadata=_build_backend_metadata(wrapper),
        metadata={
            "top_k": int(config.top_k),
            "add_special_tokens": bool(config.add_special_tokens),
        },
    )
    validate_patchscope_prompt_artifact(artifact)
    return artifact


__all__ = ["PatchscopePromptConfig", "collect_patchscope_prompt_artifact"]

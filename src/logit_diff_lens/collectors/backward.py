from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

import torch
import torch.nn.functional as F

from ..collectors.prompt import _build_backend_metadata, _decode_token_ids, _format_generation_prompt
from ..schemas import BackwardLayerRecord, BackwardPromptArtifact
from ..validation import validate_backward_prompt_artifact
from ..wrappers import LogitLensWrapper, resolve_block_component_module


@dataclass
class BackwardLensCollectorConfig:
    prompt: str
    target_token_id: int | None = None
    target_token_text: str | None = None
    target_position: Literal["last"] | int = "last"
    use_chat_template: bool = False
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain"
    system_prompt: str | None = None
    add_special_tokens: bool = True
    collect_attention_vjp: bool = True
    collect_mlp_vjp: bool = True


def _detach_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def _resolve_target_token_id(
    wrapper: LogitLensWrapper,
    config: BackwardLensCollectorConfig,
) -> tuple[int, str]:
    if config.target_token_id is not None:
        target_id = int(config.target_token_id)
        target_text = wrapper.tokenizer.decode([target_id], clean_up_tokenization_spaces=False)
        return target_id, target_text
    if config.target_token_text is None:
        raise ValueError("Provide either target_token_id or target_token_text for backward collection.")
    encoded = wrapper.tokenizer(
        config.target_token_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    if encoded.shape[-1] != 1:
        raise ValueError("target_token_text must tokenize to exactly one token in the first backward implementation.")
    target_id = int(encoded[0, 0].item())
    return target_id, config.target_token_text


@torch.enable_grad()
def collect_backward_prompt_artifact(
    wrapper: LogitLensWrapper,
    config: BackwardLensCollectorConfig,
) -> BackwardPromptArtifact:
    prompt_formatted = _format_generation_prompt(
        wrapper,
        config.prompt,
        prompt_format=config.prompt_format,
        use_chat_template=config.use_chat_template,
        system_prompt=config.system_prompt,
    )
    inputs = wrapper.tokenize_inputs(
        texts=prompt_formatted,
        device=wrapper.model_device,
        add_special_tokens=config.add_special_tokens and not config.use_chat_template,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_text = _decode_token_ids(wrapper.tokenizer, input_ids[0])
    target_token_id, target_token_text = _resolve_target_token_id(wrapper, config)
    target_position = input_ids.shape[1] - 1 if config.target_position == "last" else int(config.target_position)

    model = wrapper.model
    model.zero_grad(set_to_none=True)
    hook_records: dict[str, Dict[int, torch.Tensor]] = {
        "hidden_vjp": {},
        "attention_vjp": {},
        "mlp_vjp": {},
    }
    handles: list[Any] = []

    def _make_backward_hook(store_name: str, layer_idx: int):
        def hook(module, grad_input, grad_output):
            del grad_input
            tensor = None
            if isinstance(grad_output, (tuple, list)):
                for item in grad_output:
                    if torch.is_tensor(item):
                        tensor = item
                        break
            elif torch.is_tensor(grad_output):
                tensor = grad_output
            if tensor is not None:
                hook_records[store_name][layer_idx] = _detach_cpu(tensor)
        return hook

    for layer_idx, block in enumerate(wrapper.blocks):
        handles.append(block.register_full_backward_hook(_make_backward_hook("hidden_vjp", layer_idx)))
        if config.collect_attention_vjp:
            attn_module = resolve_block_component_module(block, "attention")
            if attn_module is not None:
                handles.append(attn_module.register_full_backward_hook(_make_backward_hook("attention_vjp", layer_idx)))
        if config.collect_mlp_vjp:
            mlp_module = resolve_block_component_module(block, "mlp")
            if mlp_module is not None:
                handles.append(mlp_module.register_full_backward_hook(_make_backward_hook("mlp_vjp", layer_idx)))

    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=False,
            use_cache=False,
        )
        logits = outputs.logits[:, target_position, :]
        labels = torch.tensor([target_token_id], device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    layer_records = []
    token_ids_cpu = _detach_cpu(input_ids.to(dtype=torch.float32)).to(dtype=torch.long)
    for layer_idx, block in enumerate(wrapper.blocks):
        layer_records.append(
            BackwardLayerRecord(
                layer_index=layer_idx,
                layer_name=f"layer_{layer_idx}",
                token_ids=token_ids_cpu,
                token_text=token_text,
                hidden_vjp=hook_records["hidden_vjp"].get(layer_idx),
                attention_vjp=hook_records["attention_vjp"].get(layer_idx),
                mlp_vjp=hook_records["mlp_vjp"].get(layer_idx),
            )
        )

    artifact = BackwardPromptArtifact(
        prompt_text=config.prompt,
        prompt_formatted=prompt_formatted,
        token_ids=token_ids_cpu,
        token_text=token_text,
        target_token_id=target_token_id,
        target_token_text=target_token_text,
        target_position=target_position,
        target_kind="token_id" if config.target_token_id is not None else "target_text",
        loss_kind="nll",
        loss_value=float(loss.detach().item()),
        backend_metadata=_build_backend_metadata(wrapper),
        layer_records=layer_records,
        metadata={
            "collect_attention_vjp": bool(config.collect_attention_vjp),
            "collect_mlp_vjp": bool(config.collect_mlp_vjp),
            "add_special_tokens": bool(config.add_special_tokens),
            "prompt_format": config.prompt_format,
            "use_chat_template": bool(config.use_chat_template),
            "system_prompt": config.system_prompt,
        },
    )
    validate_backward_prompt_artifact(artifact)
    return artifact


__all__ = ["BackwardLensCollectorConfig", "collect_backward_prompt_artifact"]

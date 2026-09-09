from __future__ import annotations

from typing import Any, Dict

import torch


@torch.no_grad()
def generate_with_model(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    use_cache: bool,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    resolved_attention_mask = (
        attention_mask.detach().clone()
        if attention_mask is not None
        else torch.ones_like(input_ids, device=input_ids.device)
    )

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    generate_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": resolved_attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": int(pad_token_id),
        "use_cache": bool(use_cache),
        "return_dict_in_generate": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = max(float(temperature), 1e-6)

    outputs = model.generate(**generate_kwargs)
    sequences = outputs.sequences.detach()
    sequence_attention_mask = torch.ones_like(sequences, device=sequences.device)
    return sequences, sequence_attention_mask

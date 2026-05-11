from typing import List, Dict, Any
import torch

from ...wrapper import (
    LogitLensWrapper,
    as_tensor,
    normalize_activations,
    lmhead_project
)



# --------------------------------------------------------
# Collect hidden states and logits for logit-lens
# --------------------------------------------------------
def _decode_token_ids(tok, token_ids: torch.Tensor) -> List[str]:
    ids = token_ids.detach().cpu().tolist()
    return [tok.decode([tid], clean_up_tokenization_spaces=False) for tid in ids]


@torch.no_grad()
def collect_logits_for_plotter(
    arch_wrapper:"LogitLensWrapper",
    prompt:str|List[str]=None,
    mode:str="raw",
    topk:int=5,
    selected_layers:List[int]|None=None,
    add_special_tokens:bool=False,
    force_include_input:bool=True,
    force_include_output:bool=True,
) -> Dict[str, Any]:
    
    model = arch_wrapper.model
    model.eval()

    device = arch_wrapper.model_device

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    tok = arch_wrapper.tokenizer

    # --------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------
    inputs = arch_wrapper.tokenize_inputs(
        texts=prompt,
        device=device,
        add_special_tokens=add_special_tokens,
    )

    full_input_ids = as_tensor(inputs["input_ids"], device=arch_wrapper.model_device)
    full_attention_mask = as_tensor(inputs["attention_mask"], device=arch_wrapper.model_device)

    if full_input_ids.ndim == 1:
        full_input_ids = full_input_ids.unsqueeze(0)
    if full_attention_mask.ndim == 1:
        full_attention_mask = full_attention_mask.unsqueeze(0)

    if full_input_ids.shape[1] < 2:
        raise ValueError("Prompt must tokenize to at least 2 tokens for next-token analysis.")

    input_ids = full_input_ids[:, :-1]
    target_ids = full_input_ids[:, 1:]
    attention_mask = full_attention_mask[:, :-1]

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if target_ids.ndim == 1:
        target_ids = target_ids.unsqueeze(0)

    T = input_ids.shape[1]

    # --------------------------------------------------
    # FORWARD + HOOKS
    # --------------------------------------------------
    acts, _ = arch_wrapper.forward_pass(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collect_attn=False
    )
    # --------------------------------------------------
    # RESULT CONTAINER
    # --------------------------------------------------
    result = {
        "prompt": prompt,
        "full_tokens": _decode_token_ids(tok, full_input_ids[0]),
        "tokens": _decode_token_ids(tok, input_ids[0]),
        "target_ids": target_ids[0].detach().cpu(),
        "target_tokens": _decode_token_ids(tok, target_ids[0]),
        "attention_mask": attention_mask[0].detach().cpu(),
        "layers": [],
        "hidden": {},
        "logits": {},
        "topk_preds": {},
        "mode": mode,
        "tokenizer": tok,
        "quantized": arch_wrapper.is_bnb_quantized,
    }

    # --------------------------------------------------
    # EMBEDDING (index = -1, name = "embedding")
    # --------------------------------------------------
    if force_include_input and "embedding" in acts:
        h_emb = acts["embedding"]          # [1, T, D]
        h_norm = normalize_activations(
            x=h_emb.clone(),
            mode=mode,
            block="embedding",
            layer_index=-1,
            model_device=arch_wrapper.model_device,
            model_dtype=arch_wrapper.model_dtype,
            final_norm=arch_wrapper.final_norm
        )
        l_norm, _ = lmhead_project(x=h_norm, lm_head=arch_wrapper.lm_head, stable=arch_wrapper.stable, model_device=arch_wrapper.model_device)

        result["hidden"][(-1, mode)] = arch_wrapper.save_to_fp32(h_norm[0]) if arch_wrapper.fp32_save else h_norm[0].cpu()
        result["logits"][(-1, mode)] = arch_wrapper.save_to_fp32(l_norm[0]) if arch_wrapper.fp32_save else l_norm[0].cpu()
        result["layers"].append(("embedding", mode))

    # --------------------------------------------------
    # BLOCKS
    # --------------------------------------------------
    selected_layer_set = set(selected_layers) if selected_layers is not None else None

    blocks = sorted(
        (v["idx"], k)
        for k, v in arch_wrapper.layer_registry.items()
        if v["type"] == "block" and k in acts
    )

    last_raw = None
    for li, k in blocks:
        last_raw = acts[k]                # [1, T, D]
        if selected_layer_set is not None and li not in selected_layer_set:
            continue
        h = normalize_activations(
            x=last_raw.clone(),
            mode=mode,
            block="block",
            layer_index=li,
            model_device=arch_wrapper.model_device,
            model_dtype=arch_wrapper.model_dtype,
            final_norm=arch_wrapper.final_norm
        )
        l, _ = lmhead_project(x=h, lm_head=arch_wrapper.lm_head, stable=arch_wrapper.stable, model_device=arch_wrapper.model_device)

        result["hidden"][(li, mode)] = arch_wrapper.save_to_fp32(h[0]) if arch_wrapper.fp32_save else h[0].cpu()
        result["logits"][(li, mode)] = arch_wrapper.save_to_fp32(l[0]) if arch_wrapper.fp32_save else l[0].cpu()
        result["layers"].append((li, mode))

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    if force_include_output and last_raw is not None:
        out_idx = max(i for i, _ in result["layers"] if isinstance(i, int)) + 1

        h = normalize_activations(
            x=last_raw.clone(),
            mode=mode,
            block="output",
            layer_index=out_idx,
            model_device=arch_wrapper.model_device,
            model_dtype=arch_wrapper.model_dtype,
            final_norm=arch_wrapper.final_norm
        )
        l, _ = lmhead_project(x=h, lm_head=arch_wrapper.lm_head, stable=arch_wrapper.stable, model_device=arch_wrapper.model_device)

        result["hidden"][(out_idx, mode)] = arch_wrapper.save_to_fp32(h[0]) if arch_wrapper.fp32_save else h[0].cpu()
        result["logits"][(out_idx, mode)] = arch_wrapper.save_to_fp32(l[0]) if arch_wrapper.fp32_save else l[0].cpu()
        result["layers"].append(("output", mode))

    # --------------------------------------------------
    # TOP-K
    # --------------------------------------------------
    for (i, m), logits in result["logits"].items():
        if m != mode:
            continue
        probs = torch.softmax(logits, dim=-1)
        _, topk_idx = torch.topk(probs, k=topk, dim=-1)
        result["topk_preds"][(i, mode)] = [
            [tok.decode([t]) for t in row.tolist()]
            for row in topk_idx
        ]

    # --------------------------------------------------
    # SHAPE ASSERTS
    # --------------------------------------------------
    for k, v in result["hidden"].items():
        assert v.shape[0] == T, (k, v.shape)
    for k, v in result["logits"].items():
        assert v.shape[0] == T, (k, v.shape)

    #print(result["layers"])
    return result

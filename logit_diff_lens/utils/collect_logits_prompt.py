from typing import Any
import torch
from ..wrapper.arch_wrapper import ArchWrapper



# --------------------------------------------------------
# Collect hidden states and logits for logit-lens
# --------------------------------------------------------
@torch.no_grad()
def collect_logits_for_plotter(
    arch_wrapper:"ArchWrapper",
    prompt:str|list[str]=None,
    mode:str="raw",
    topk:int=5,
    add_special_tokens:bool=False,
    force_include_input:bool=True,
    force_include_output:bool=True,
):
    model = arch_wrapper.model
    model.eval()

    device = arch_wrapper.model_device

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    tok = arch_wrapper.tokenizer

    # --------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------
    inputs = arch_wrapper.plotter_inputs(
        texts=prompt,
        device=device,
        add_special_tokens=add_special_tokens,
    )

    input_ids = arch_wrapper.as_tensor(inputs["input_ids"], device)
    target_ids = arch_wrapper.as_tensor(inputs["target_ids"], device)

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if target_ids.ndim == 1:
        target_ids = target_ids.unsqueeze(0)

    T = input_ids.shape[1]

    # --------------------------------------------------
    # FORWARD + HOOKS
    # --------------------------------------------------
    acts, _ = arch_wrapper.forward_collect(
        input_ids=input_ids,
        collect_attn=False
    )
    # --------------------------------------------------
    # RESULT CONTAINER
    # --------------------------------------------------
    result = {
        "prompt": prompt,
        "tokens": tok.convert_ids_to_tokens(input_ids[0]),
        "target_ids": target_ids[0].detach().cpu(),
        "target_tokens": tok.convert_ids_to_tokens(target_ids[0]),
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
        h_norm = arch_wrapper.apply_normalization(
            h_emb.clone(),
            mode,
            "embedding",
            -1,
        )
        l_norm, _ = arch_wrapper.project(h_norm)

        result["hidden"][(-1, mode)] = arch_wrapper.save_to_fp32(h_norm[0]) if arch_wrapper.fp32_save else h_norm[0].cpu()
        result["logits"][(-1, mode)] = arch_wrapper.save_to_fp32(l_norm[0]) if arch_wrapper.fp32_save else l_norm[0].cpu()
        result["layers"].append(("embedding", mode))

    # --------------------------------------------------
    # BLOCKS
    # --------------------------------------------------
    blocks = sorted(
        (v["idx"], k)
        for k, v in arch_wrapper.layer_registry.items()
        if v["type"] == "block" and k in acts
    )

    last_raw = None
    for li, k in blocks:
        last_raw = acts[k]                # [1, T, D]
        h = arch_wrapper.apply_normalization(
            last_raw.clone(),
            mode,
            "block",
            li,
        )
        l, _ = arch_wrapper.project(h)

        result["hidden"][(li, mode)] = arch_wrapper.save_to_fp32(h[0]) if arch_wrapper.fp32_save else h[0].cpu()
        result["logits"][(li, mode)] = arch_wrapper.save_to_fp32(l[0]) if arch_wrapper.fp32_save else l[0].cpu()
        result["layers"].append((li, mode))

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    if force_include_output and last_raw is not None:
        out_idx = max(i for i, _ in result["layers"] if isinstance(i, int)) + 1

        h = arch_wrapper.apply_normalization(
            last_raw.clone(),
            mode,
            "output",
            out_idx,
        )
        l, _ = arch_wrapper.project(h)

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
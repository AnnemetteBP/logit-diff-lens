from typing import Any
import torch
import torch.nn.functional as F

from ..wrapper.arch_wrapper import ArchWrapper
from .collect_logits_prompt import collect_logits_for_plotter



# --------------------------------------------------------
# LogitDiff: comparing result from two logit lenses
# --------------------------------------------------------
@torch.no_grad()
def _apply_ldl_plotter(
    result_A:dict,
    result_B:dict,
    compute_metric:str="gt_prob_diff",
    topk:int=5,
    smooth_align: bool = True,
) -> dict[str, Any]:

    # --------------------------------------------------------
    # --- Sanity checks ---
    # --------------------------------------------------------
    tokA, tokB = result_A["tokenizer"], result_B["tokenizer"]
    if tokA.vocab_size != tokB.vocab_size:
        raise ValueError(f"Tokenizer vocab size mismatch: {tokA.vocab_size} vs {tokB.vocab_size}")

    # Length alignment
    lenA, lenB = len(result_A["tokens"]), len(result_B["tokens"])
    min_len = min(lenA, lenB)
    if lenA != lenB:
        print(f"[AlignWarning] Token lengths differ ({lenA} vs {lenB}). Truncating to {min_len}.")
        for k in ("tokens", "target_tokens", "target_ids", "attention_mask"):
            if k in result_A and k in result_B:
                result_A[k] = result_A[k][:min_len]
                result_B[k] = result_B[k][:min_len]

    # --------------------------------------------------------
    # --- Helpers ---
    # --------------------------------------------------------
    def _safe_softmax(x, eps=1e-12):
        x = x.to(torch.float32)
        log_p = F.log_softmax(x, dim=-1)
        p = torch.exp(log_p)
        return p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

    def _kl_div(p, q, eps=1e-12):
        p, q = p.clamp_min(eps), q.clamp_min(eps)
        return torch.sum(p * (p.log() - q.log()), dim=-1).clamp_min(0.0)

    def _js_div(p, q, eps=1e-12):
        p, q = p.clamp_min(eps), q.clamp_min(eps)
        m = 0.5 * (p + q)
        return 0.5 * (_kl_div(p, m, eps) + _kl_div(q, m, eps))

    def _tvd(p, q): return 0.5 * torch.sum(torch.abs(p - q), dim=-1)
    def _cos_sim(a, b): return F.cosine_similarity(a, b, dim=-1)
    def _l2_dist(a, b): return torch.norm(a - b, dim=-1)
    def _jaccard(a_idx, b_idx):
        inter = len(set(a_idx) & set(b_idx))
        union = len(set(a_idx) | set(b_idx))
        return inter / max(union, 1)

    # --------------------------------------------------------
    # --- Layer alignment setup ---
    # --------------------------------------------------------
    modeA = result_A["mode"]
    modeB = result_B["mode"]
    if modeA != modeB:
        raise ValueError(f"Mode mismatch: {modeA} vs {modeB}")
    modes = [result_A["mode"]]

    layers_A = sorted(
        i for (i, m) in result_A["logits"].keys()
        if m == modeA
    )
    layers_B = sorted(
        i for (i, m) in result_B["logits"].keys()
        if m == modeA
    )

    nA, nB = len(layers_A), len(layers_B)
    print(f"[LAYER INFO] Layers in A: {layers_A}")
    print(f"[LAYER INFO] Layers in B: {layers_B}")

    def _align_layers(idxA):
        if smooth_align and nB > 1:
            pos = idxA * (nB - 1) / max(nA - 1, 1)
            low, high = int(pos), min(int(pos) + 1, nB - 1)
            w = pos - low
            return low, high, w
        else:
            idxB = int(round(idxA * (nB - 1) / max(nA - 1, 1)))
            return idxB, idxB, 0.0

    # --------------------------------------------------------
    # --- Result container ---
    # --------------------------------------------------------
    result = {
        "prompt": result_A["prompt"],
        "tokens": result_A["tokens"],
        "target_tokens": result_A["target_tokens"],
        "layers": [],
        "metrics": {},
        "norm_modes": tuple(modes),
        "tokenizer": tokA,
        "topk_preds_A": result_A["topk_preds"],
        "topk_preds_B": result_B["topk_preds"],
    }

    # --------------------------------------------------------
    # --- Metric computation ---
    # --------------------------------------------------------
    for mode in modes:
        for li, idxA in enumerate(layers_A):
            lowB, highB, w = _align_layers(li)

            logits_A = result_A["logits"][(idxA, mode)]
            hA = result_A["hidden"][(idxA, mode)]

            if smooth_align and lowB != highB:
                logits_B = (1 - w) * result_B["logits"][(layers_B[lowB], mode)] \
                          + w * result_B["logits"][(layers_B[highB], mode)]
                hB = (1 - w) * result_B["hidden"][(layers_B[lowB], mode)] \
                      + w * result_B["hidden"][(layers_B[highB], mode)]
            else:
                logits_B = result_B["logits"][(layers_B[lowB], mode)]
                hB = result_B["hidden"][(layers_B[lowB], mode)]

            # Compute probabilities
            pA, pB = _safe_softmax(logits_A), _safe_softmax(logits_B)

            min_len = min(pA.shape[0], pB.shape[0], len(result_A["target_ids"]))
            pA, pB = pA[:min_len], pB[:min_len]
            hA, hB = hA[:min_len], hB[:min_len]
            target_ids = torch.tensor(result_A["target_ids"][:min_len]).detach().clone().requires_grad_(False)

            # Apply valid mask if available
            if "attention_mask" in result_A and "attention_mask" in result_B:
                maskA = result_A["attention_mask"][:min_len].bool()
                maskB = result_B["attention_mask"][:min_len].bool()
                valid_mask = maskA & maskB
                if valid_mask.sum() > 0:
                    pA, pB = pA[valid_mask], pB[valid_mask]
                    hA, hB = hA[valid_mask], hB[valid_mask]
                    target_ids = target_ids[valid_mask]

            # ---- Compute selected metric ----
            if compute_metric == "kl_div_ab":
                values = _kl_div(pA, pB)
            elif compute_metric == "kl_div_ba":
                values = _kl_div(pB, pA)
            elif compute_metric in {"js_div", "js_dist"}:
                values = _js_div(pA, pB)
                if compute_metric == "js_dist":
                    values = torch.sqrt(values)
            elif compute_metric == "tvd":
                values = _tvd(pA, pB)
            elif compute_metric == "perplexity_diff":
                H_A = (-(pA * pA.log()).sum(-1))
                H_B = (-(pB * pB.log()).sum(-1))
                values = torch.exp(H_A) - torch.exp(H_B)
            elif compute_metric == "gt_prob_diff":
                values = pA[torch.arange(len(target_ids)), target_ids] \
                       - pB[torch.arange(len(target_ids)), target_ids]
            elif compute_metric == "cos_sim":
                values = _cos_sim(hA, hB)
            elif compute_metric == "l2_dist":
                values = _l2_dist(hA, hB)
            elif compute_metric == "jaccard":
                topA = torch.topk(pA, k=topk, dim=-1).indices
                topB = torch.topk(pB, k=topk, dim=-1).indices
                values = torch.tensor([_jaccard(a.tolist(), b.tolist()) for a, b in zip(topA, topB)])
            elif compute_metric == "disagreement_correct_top1":
                top1A = torch.argmax(pA, dim=-1)
                top1B = torch.argmax(pB, dim=-1)
                gt_correct_A = (top1A == target_ids)
                gt_correct_B = (top1B == target_ids)
                values = (gt_correct_A ^ gt_correct_B).float()
            else:
                raise ValueError(f"Unsupported metric: {compute_metric}")

            # Clean up NaNs/Infs (avoid blank heatmaps)
            if torch.isnan(values).any() or torch.isinf(values).any():
                values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            # Save
            result["layers"].append((idxA, mode))
            result["metrics"].setdefault(f"{compute_metric}_{mode}", []).append(values.cpu())

    return result


# --------------------------------------------------------
# Public function to run LogitDiff and plotter
# --------------------------------------------------------
def apply_ldl_plotter(
    arch_wrappers:tuple["ArchWrapper", "ArchWrapper"],
    prompt:str,
    norm_mode:str="raw",
    add_special_tokens:bool=False,
    topk:int=1,
    compute_metric:str="gt_prob_diff",
    force_include_input:bool=True,
    force_include_output:bool=True,
) -> dict[str, Any]:

    arch_A, arch_B = arch_wrappers

    # ---------------------------------------------------------------------
    # Check tokenizer compatibility
    # ---------------------------------------------------------------------
    tok_A, tok_B = arch_A.tokenizer, arch_B.tokenizer

    # Check class
    if type(tok_A) is not type(tok_B):
        raise ValueError(
            f"[ERROR] Incompatible tokenizers: {type(tok_A).__name__} vs {type(tok_B).__name__}!"
        )

    # Check vocab size and special tokens
    if tok_A.vocab_size != tok_B.vocab_size:
        raise ValueError(
            f"[ERROR] Tokenizer vocab size mismatch: {tok_A.vocab_size} vs {tok_B.vocab_size}!"
        )

    # Check special token consistency
    specials = ["bos_token", "eos_token", "pad_token", "unk_token"]
    for s in specials:
        tokA_attr, tokB_attr = getattr(tok_A, s, None), getattr(tok_B, s, None)
        if tokA_attr != tokB_attr:
            print(f"[WARN] Tokenizer special token mismatch for {s}: {tokA_attr} vs {tokB_attr}!")

    # ---------------------------------------------------------------------
    # Check model vocab sizes (lm_head)
    # ---------------------------------------------------------------------
    model_vocab_A = getattr(arch_A.model.config, "vocab_size", None)
    model_vocab_B = getattr(arch_B.model.config, "vocab_size", None)
    if model_vocab_A != model_vocab_B:
        raise ValueError(
            f"[ERROR] Model vocab size mismatch: {model_vocab_A} vs {model_vocab_B}!"
        )

    print(f"[OK] Compatible models detected → vocab size = {model_vocab_A}")
    print(f"     Tokenizer = {type(tok_A).__name__} (shared vocab, identical mapping)")
    print(f"     Proceeding with LogitDiff computation...\n")

    # ---------------------------------------------------------------------
    # Collect data for both models
    # ---------------------------------------------------------------------
    result_A = collect_logits_for_plotter(
        arch_wrapper=arch_A,
        prompt=prompt,
        mode=norm_mode,
        topk=topk,
        add_special_tokens=add_special_tokens,
        force_include_input=force_include_input,
        force_include_output=force_include_output,
    )

    result_B = collect_logits_for_plotter(
        arch_wrapper=arch_B,
        prompt=prompt,
        mode=norm_mode,
        topk=topk,
        add_special_tokens=add_special_tokens,
        force_include_input=force_include_input,
        force_include_output=force_include_output,
    )

    # ---------------------------------------------------------------------
    # Compute comparison metrics
    # ---------------------------------------------------------------------
    result = _apply_ldl_plotter(
        result_A=result_A,
        result_B=result_B,
        compute_metric=compute_metric,
        topk=topk
    )
    
    return result
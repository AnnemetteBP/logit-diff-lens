from typing import Any
import torch
import torch.nn.functional as F

from ..wrapper.arch_wrapper import ArchWrapper
from .collect_logits_prompt import collect_logits_for_plotter



# --------------------------------------------------------
# Compute top-k metrics from logit-lens collector result
# --------------------------------------------------------
@torch.no_grad()
def _compute_logit_lens_topk(result:dict, topk:int=5) -> dict:

    def _softmax_stats(logits):
        log_p = F.log_softmax(logits.float(), dim=-1)
        return log_p.exp(), log_p

    def _kl(p, q):
        eps = 1e-8
        p = torch.clamp(p, eps, 1.0)
        q = torch.clamp(q, eps, 1.0)
        return torch.sum(p * (p.log() - q.log()), dim=-1)

    def _js(p, q):
        m = 0.5 * (p + q)
        return 0.5 * (_kl(p, m) + _kl(q, m))

    def _l2(a, b): return torch.norm(a - b, dim=-1)
    def _cos(a, b): return F.cosine_similarity(a, b, dim=-1)

    def _jaccard(a, b):
        a, b = set(a), set(b)
        return len(a & b) / max(len(a | b), 1)

    tokenizer = result["tokenizer"]
    target_ids = result["target_ids"]
    layer_logits = result["logits"]
    layer_hidden = result["hidden"]
    mode = result["mode"]

    # --------------------------------------------------
    # CRITICAL FIX: USE result["layers"] ORDER
    # --------------------------------------------------
    indices = []
    for layer_id, m in result["layers"]:
        if m != mode:
            continue
        if layer_id == "embedding":
            indices.append(-1)
        elif layer_id == "output":
            indices.append(
                max(i for (i, mm) in layer_logits.keys() if mm == mode)
            )
        else:
            indices.append(layer_id)

    if not indices:
        return result

    result["metrics"] = {}

    final_idx = indices[-1]
    final_logits = layer_logits[(final_idx, mode)]
    final_probs, _ = _softmax_stats(final_logits)
    final_hidden = layer_hidden[(final_idx, mode)]

    valid_len = len(target_ids)

    for pos, i in enumerate(indices):
        logits_i = layer_logits[(i, mode)]
        hidden_i = layer_hidden[(i, mode)]
        probs_i, logp_i = _softmax_stats(logits_i)
        prev_idx = indices[pos - 1] if pos > 0 else None

        m = {}
        m["gt_prob"] = probs_i[torch.arange(valid_len), target_ids].cpu()
        ent = -(probs_i * logp_i).sum(-1)
        m["entropy"] = ent.cpu()
        m["perplexity"] = torch.exp(ent).cpu()
        m["logits_mean"] = logits_i.mean(-1).cpu()
        m["logits_std"] = logits_i.std(-1).cpu()
        top2 = torch.topk(logits_i, 2, dim=-1).values
        m["logit_margin"] = (top2[:, 0] - top2[:, 1]).cpu()
        m["probs_std"] = probs_i.std(-1).cpu()
        m["max_prob"] = probs_i.max(-1).values.cpu()

        if i != final_idx:
            m["kl_div_last"] = _kl(probs_i, final_probs).cpu()
            m["js_div_last"] = _js(probs_i, final_probs).cpu()
            m["cos_sim_last"] = _cos(hidden_i, final_hidden).cpu()
            m["l2_dist_last"] = _l2(hidden_i, final_hidden).cpu()

        if prev_idx is not None:
            prev_probs, _ = _softmax_stats(layer_logits[(prev_idx, mode)])
            prev_hidden = layer_hidden[(prev_idx, mode)]
            m["kl_div_prev"] = _kl(probs_i, prev_probs).cpu()
            m["js_div_prev"] = _js(probs_i, prev_probs).cpu()
            m["cos_sim_prev"] = _cos(hidden_i, prev_hidden).cpu()
            m["l2_dist_prev"] = _l2(hidden_i, prev_hidden).cpu()

        topk_cur = result["topk_preds"][(i, mode)]
        m["topk_preds"] = topk_cur

        if prev_idx is not None:
            topk_prev = result["topk_preds"][(prev_idx, mode)]
            m["jaccard_prev"] = torch.tensor(
                [_jaccard(a[:topk], b[:topk]) for a, b in zip(topk_cur, topk_prev)]
            )

        topk_last = result["topk_preds"][(final_idx, mode)]
        m["jaccard_last"] = torch.tensor(
            [_jaccard(a[:topk], b[:topk]) for a, b in zip(topk_cur, topk_last)]
        )

        m["accuracy_topk"] = torch.tensor(
            [target_ids[t].item() in tokenizer.convert_tokens_to_ids(topk_cur[t][:topk])
             for t in range(valid_len)],
            dtype=torch.float
        )

        for k, v in m.items():
            result["metrics"].setdefault(f"{k}_{mode}", {})[(i, mode)] = v

    return result



# ------------------------------------------------------------
# Public function for calling logit lens and plotter
# ------------------------------------------------------------
def collect_logit_lens_topk(
    arch_wrapper:"ArchWrapper",
    prompt:str,
    norm_mode:str="raw",
    topk:int=1,
    add_special_tokens:bool=False,
    force_include_input:bool=True,
    force_include_output:bool=True,
) -> dict[str, Any]:
    
    lens_result = collect_logits_for_plotter(
        arch_wrapper=arch_wrapper,
        prompt=prompt,
        mode=norm_mode,
        topk=topk,
        add_special_tokens=add_special_tokens,
        force_include_input=force_include_input,
        force_include_output=force_include_output,
    )

    result = _compute_logit_lens_topk(
        result=lens_result,
        topk=topk,
    )

    return result
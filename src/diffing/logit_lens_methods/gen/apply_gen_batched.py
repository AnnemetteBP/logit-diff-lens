from typing import List, Dict, Tuple, Any
from pathlib import Path
import os, gc, json
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from ..wrapper import GenerationLensWrapper, lmhead_project



# ============================================================
# STOPWORDS
# ============================================================
STOPWORDS = [
    "the","a","an","and","or","but","if","then","else","when","at","by","for",
    "with","about","against","between","into","through","during","before","after",
    "above","below","to","from","up","down","in","out","on","off","over","under",
    "again","further","once","here","there","all","any","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","can","will","just","should","now"
]


# ============================================================
# EXTRA IGNORE TOKENS
# ============================================================
EXTRA_IGNORE_STRINGS = [
    "(", ")", "[", "]", "{", "}",
    ".", ",", ":", ";", "!", "?",
    "-", "_", "/", "\\", "|",
    "'", '"', "`",
    "…", "—",
    "Ø", "ø"
]


# ============================================================
# IGNORE IDS
# ============================================================
def build_ignore_ids(tokenizer:Any) -> set:
    ignore = set()

    for t in [
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.unk_token_id,
    ]:
        if t is not None:
            ignore.add(t)

    for w in STOPWORDS:
        ignore.update(tokenizer.encode(w, add_special_tokens=False))

    for s in EXTRA_IGNORE_STRINGS:
        ignore.update(tokenizer.encode(s, add_special_tokens=False))

    return ignore


# ============================================================
# LOAD
# ============================================================
def load_rows_file(path:str|Path) -> List[Dict[str, Any]]:
    return torch.load(path, map_location="cpu")["rows"]


# ============================================================
# TOKEN METRICS
# ============================================================
def compute_token_metrics(logits, input_ids=None):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)

    entropy = -(probs * log_probs).sum(dim=-1)
    perplexity = torch.exp(entropy)

    out = {
        "log_probs": log_probs.detach().cpu(),
        "entropy": entropy.detach().cpu(),
        "perplexity": perplexity.detach().cpu(),
    }

    if input_ids is not None and logits.shape[1] > 1:
        targets = input_ids[:, 1:]
        probs_shifted = probs[:, :-1, :]
        gt = torch.gather(probs_shifted, -1, targets.unsqueeze(-1)).squeeze(-1)
        out["gt_prob"] = gt.detach().cpu()

    return out


# ============================================================
# EXTRA METRICS
# ============================================================
def compute_distribution_metrics(p:torch.Tensor, q:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = p + 1e-9
    q = q + 1e-9

    kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)

    m = 0.5 * (p + q)
    js = 0.5 * (
        (p * (torch.log(p) - torch.log(m))).sum(dim=-1) +
        (q * (torch.log(q) - torch.log(m))).sum(dim=-1)
    )

    tvd = 0.5 * torch.abs(p - q).sum(dim=-1)

    return kl, js, tvd


def compute_hidden_metrics(a:torch.Tensor, b:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    l2 = torch.norm(a - b, dim=-1)
    cos = F.cosine_similarity(a, b, dim=-1)
    return l2, cos


# ============================================================
# HELPERS
# ============================================================
def compute_jaccard(a:set, b:set) -> float:
    return len(a & b) / len(a | b) if len(a | b) else 0.0


def compute_persistence(topk_layers:List[set]) -> Dict[str, Any]:
    counts = {}
    for s in topk_layers:
        for t in s:
            counts[t] = counts.get(t, 0) + 1

    persistent = [t for t, c in counts.items() if c >= len(topk_layers)//2]

    return {
        "counts": counts,
        "persistent_tokens": persistent,
        "persistence_score": len(persistent) / len(counts) if counts else 0.0
    }


def get_topk_filtered(logits:torch.Tensor, ignore_ids:set, k:int=20) -> set:
    probs = torch.softmax(logits, dim=-1).clone()
    if ignore_ids:
        probs[..., list(ignore_ids)] = 0.0
    _, idx = torch.topk(probs, k, dim=-1)
    return set(idx[0].tolist())


def group_rows(rows:List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    grouped = {}
    for r in rows:
        key = (r["prompt_id"], r["step"])
        grouped.setdefault(key, []).append(r)
    return grouped


# ============================================================
# SINGLE MODEL
# ============================================================
def analyze_single_model(rows:List[Dict[str, Any]], tokenizer:PreTrainedTokenizerBase, norm_modes:Tuple[str, ...], k:int=20) -> List[Dict[str, Any]]:

    ignore_ids = build_ignore_ids(tokenizer)
    grouped = group_rows(rows)
    results = []

    for (pid, step), layer_rows in grouped.items():
        layer_rows = sorted(layer_rows, key=lambda x: x["layer_index"])

        for mode in norm_modes:

            topk_layers = []

            for r in layer_rows:

                logits_full = r[f"logits_{mode}"]

                token_metrics = compute_token_metrics(
                    logits_full,
                    r.get("input_ids", None)
                )

                probs = torch.softmax(logits_full, dim=-1)

                for pos in range(probs.shape[1]):

                    p = probs[:, pos, :]
                    entropy = token_metrics["entropy"][:, pos]
                    perplexity = token_metrics["perplexity"][:, pos]
                    max_prob = torch.max(p, dim=-1).values

                    results.append({
                        "batch_index": r.get("batch_index", -1),
                        "prompt_id": pid,
                        "step": step,
                        "layer_index": r["layer_index"],
                        "layer_name": r["layer_name"],
                        "mode": mode,
                        "position": pos,

                        "entropy": entropy.tolist(),
                        "perplexity": perplexity.tolist(),
                        "max_prob": max_prob.tolist(),
                    })

                logits_last = logits_full[:, -1, :]
                topk_layers.append(get_topk_filtered(logits_last, ignore_ids, k))

            # ===== coherence etc =====
            jaccards = [
                compute_jaccard(topk_layers[i], topk_layers[i+1])
                for i in range(len(topk_layers)-1)
            ]

            coherence = sum(jaccards)/len(jaccards) if jaccards else 1.0

            union = set().union(*topk_layers)
            inter = set(topk_layers[0])
            for s in topk_layers:
                inter &= s

            misalignment = 1 - (len(inter)/len(union)) if union else 0.0
            persistence = compute_persistence(topk_layers)

            results.append({
                "batch_index": layer_rows[0].get("batch_index", -1),
                "prompt_id": pid,
                "step": step,
                "layer_index": -1,
                "layer_name": "summary",
                "mode": mode,

                "coherence": coherence,
                "misalignment": misalignment,
                "persistence": persistence,
            })

    return results


# ============================================================
# COMPARE
# ============================================================
def compare_models(rows_a:List[Dict[str, Any]], rows_b:List[Dict[str, Any]], tokenizer:PreTrainedTokenizerBase, norm_modes:Tuple[str, ...], k:int=20) -> List[Dict[str, Any]]  :

    ignore_ids = build_ignore_ids(tokenizer)
    gA = group_rows(rows_a)
    gB = group_rows(rows_b)
    results = []

    for key in gA:
        if key not in gB:
            continue

        layers_a = sorted(gA[key], key=lambda x: x["layer_index"])
        layers_b = sorted(gB[key], key=lambda x: x["layer_index"])

        for mode in norm_modes:

            topk_layers = []

            for la, lb in zip(layers_a, layers_b):

                pa = torch.softmax(la[f"logits_{mode}"], dim=-1)
                pb = torch.softmax(lb[f"logits_{mode}"], dim=-1)

                for pos in range(pa.shape[1]):

                    p = pa[:, pos, :]
                    q = pb[:, pos, :]

                    kl, js, tvd = compute_distribution_metrics(p, q)

                    results.append({
                        "batch_index": la.get("batch_index", -1),
                        "prompt_id": key[0],
                        "step": key[1],
                        "layer_index": la["layer_index"],
                        "layer_name": la["layer_name"],
                        "mode": mode,
                        "position": pos,

                        "kl": kl.tolist(),
                        "js": js.tolist(),
                        "tvd": tvd.tolist(),
                    })

                logits_last = la[f"logits_{mode}"][:, -1, :]
                topk_layers.append(get_topk_filtered(logits_last, ignore_ids, k))

            jaccards = [
                compute_jaccard(topk_layers[i], topk_layers[i+1])
                for i in range(len(topk_layers)-1)
            ]

            coherence = sum(jaccards)/len(jaccards) if jaccards else 1.0
            persistence = compute_persistence(topk_layers)

            results.append({
                "batch_index": layers_a[0].get("batch_index", -1),
                "prompt_id": key[0],
                "step": key[1],
                "layer_index": -1,
                "layer_name": "summary",
                "mode": mode,

                "coherence": coherence,
                "persistence": persistence,
            })

    return results


# ============================================================
# ADL HIDDEN
# ============================================================
def analyze_adl_hidden(rows_a:List[Dict[str, Any]], rows_b:List[Dict[str, Any]], tokenizer:PreTrainedTokenizerBase, lmhead:"GenerationLensWrapper", norm_modes:Tuple[str, ...], k:int=20) -> List[Dict[str, Any]]:

    ignore_ids = build_ignore_ids(tokenizer)
    gA = group_rows(rows_a)
    gB = group_rows(rows_b)
    results = []

    for key in gA:
        if key not in gB:
            continue

        layers_a = sorted(gA[key], key=lambda x: x["layer_index"])
        layers_b = sorted(gB[key], key=lambda x: x["layer_index"])

        for mode in norm_modes:

            topk_layers = []

            for la, lb in zip(layers_a, layers_b):

                hA = la[f"hidden_{mode}"]
                hB = lb[f"hidden_{mode}"]

                for pos in range(hA.shape[1]):

                    a = hA[:, pos, :]
                    b = hB[:, pos, :]

                    l2, cos = compute_hidden_metrics(a, b)

                    results.append({
                        "batch_index": la.get("batch_index", -1),
                        "prompt_id": key[0],
                        "step": key[1],
                        "layer_index": la["layer_index"],
                        "layer_name": la["layer_name"],
                        "mode": mode,
                        "position": pos,

                        "l2": l2.tolist(),
                        "cosine": cos.tolist(),
                    })

                delta = hA[:, -1, :] - hB[:, -1, :]
                logits_delta, _ = lmhead_project(
                    x=delta,
                    lm_head=lmhead.lm_head,
                    stable=lmhead.stable,
                    model_device=lmhead.model_device
                )

                topk_layers.append(get_topk_filtered(logits_delta, ignore_ids, k))

            coherence = sum(
                compute_jaccard(topk_layers[i], topk_layers[i+1])
                for i in range(len(topk_layers)-1)
            ) / max(len(topk_layers)-1, 1)

            persistence = compute_persistence(topk_layers)

            results.append({
                "batch_index": layers_a[0].get("batch_index", -1),
                "prompt_id": key[0],
                "step": key[1],
                "layer_index": -1,
                "layer_name": "summary",
                "mode": mode,

                "coherence": coherence,
                "persistence": persistence,
            })

    return results


# ============================================================
# SAVE 
# ============================================================
def save_output(rows:List[Dict[str, Any]], path:str|Path, fmt:str="jsonl") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    elif fmt == "pt":
        torch.save({"rows": rows}, path)
    else:
        raise ValueError(fmt)



# ===========================================================
# MAIN PIPELINE 
# ============================================================
def run_full_analysis(
    arch_wrappers:Tuple["GenerationLensWrapper", "GenerationLensWrapper"],
    tokenizer_A:bool,
    lmhead_A:bool,
    dir_a:str,
    dir_b:str,
    output_dir:str,
    norm_modes:Tuple[str, ...]=("raw","unit_norm","eps_norm","model_norm"),
    k:int=20,
    save_format:str="jsonl",
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    files_a = sorted(f for f in os.listdir(dir_a) if f.endswith(".pt"))
    files_b = sorted(f for f in os.listdir(dir_b) if f.endswith(".pt"))

    wrapper_A, wrapper_B = arch_wrappers[0], arch_wrappers[1]
    tokenizer = wrapper_A.tokenizer if tokenizer_A else wrapper_B.tokenizer
    lmhead = wrapper_A if lmhead_A else wrapper_B

    for batch_idx, (fa, fb) in enumerate(zip(files_a, files_b)):

        print(f"[batch {batch_idx}]")

        rows_a = load_rows_file(os.path.join(dir_a, fa))
        rows_b = load_rows_file(os.path.join(dir_b, fb))

        for r in rows_a:
            r["batch_index"] = batch_idx
        for r in rows_b:
            r["batch_index"] = batch_idx

        single_a = analyze_single_model(rows_a, wrapper_A.tokenizer, norm_modes, k)
        single_b = analyze_single_model(rows_b, wrapper_B.tokenizer, norm_modes, k)
        compare = compare_models(rows_a, rows_b, tokenizer, norm_modes, k)
        adl = analyze_adl_hidden(rows_a, rows_b, tokenizer, lmhead, norm_modes, k)

        ext = "jsonl" if save_format == "jsonl" else "pt"

        save_output(single_a, Path(output_dir) / f"single_{batch_idx:03d}.{ext}", save_format)
        save_output(single_b, Path(output_dir) / f"single_{batch_idx:03d}.{ext}", save_format)
        save_output(compare, Path(output_dir) / f"compare_{batch_idx:03d}.{ext}", save_format)
        save_output(adl, Path(output_dir) / f"adl_{batch_idx:03d}.{ext}", save_format)

        del rows_a, rows_b, single_a, single_b, compare, adl
        gc.collect()


def apply_generation_analysis(
    arch_wrappers:Tuple["GenerationLensWrapper", "GenerationLensWrapper"],
    tokenizer_A:bool,
    lmhead_A:bool,
    dir_a:str,
    dir_b:str,
    output_dir:str,
    norm_modes:Tuple[str, ...]=("raw","unit_norm","eps_norm","model_norm"),
    k:int=20,
    save_format:str="jsonl",
) -> None:
    run_full_analysis(
        arch_wrappers=arch_wrappers,
        tokenizer_A=tokenizer_A,
        lmhead_A=lmhead_A,
        dir_a=dir_a,
        dir_b=dir_b,
        output_dir=output_dir,
        norm_modes=norm_modes,
        k=k,
        save_format=save_format,
    )

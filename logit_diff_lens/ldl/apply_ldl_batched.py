import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
import os, gc, json, math
from tqdm import tqdm



def save_jsonl(records: List, path: str):
    path = f"{path}.jsonl" if ".jsonl" not in path else path
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

# ============================================================
# helpers
# ============================================================
def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unexpected shape: {x.shape}")


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2:
        return t
    if t.dim() == 1:
        return t.unsqueeze(0)
    raise ValueError(f"Expected [B,L], got {t.shape}")

# ============================================================
# extract modes (collector-correct)
# hidden is shared; modes come from logits_*
# ============================================================
def _extract_modes(row: Any) -> Dict:
    hidden = row.get("hidden")
    if hidden is None:
        return {}
    out = {}
    for k, v in row.items():
        if k.startswith("logits_") and v is not None:
            mode = k.replace("logits_", "")
            out[mode] = (hidden, v)
    return out

# ============================================================
# preprocess rows -> layers[lid][mode] = {hidden, logits, target}
# ============================================================
def _preprocess_rows(rows: List[dict], norm_modes) -> Dict:
    layers = {}

    for r in rows:
        lid = r["layer_index"]
        if lid == -1:
            continue

        pid = r["prompt_id"]

        tokens = _ensure_2d(r["tokens"])
        hidden_full = _ensure_3d(r["hidden"])

        assert tokens.size(1) == hidden_full.size(1)

        target = tokens[:, 1:]
        hidden = hidden_full[:, :-1]

        for k, v in r.items():
            if not k.startswith("logits_") or v is None:
                continue

            mode = k.replace("logits_", "")
            if mode not in norm_modes:
                continue

            logits = _ensure_3d(v)[:, :-1]

            L = min(hidden.size(1), logits.size(1), target.size(1))

            layers \
                .setdefault(lid, {}) \
                .setdefault(mode, {}) \
                [pid] = {
                    "hidden": hidden[:, :L].float(),
                    "logits": logits[:, :L].float(),
                    "target": target[:, :L].long(),
                    "batch_index": r["batch_index"],
                    "prompt_id": pid,
                    "layer_name": r["layer_name"],
                }

    return layers


# ============================================================
# Computations and Asserts
# ============================================================
def assert_finite(x: torch.Tensor, name: str):
    if not torch.isfinite(x).all():
        raise AssertionError(f"{name}: NaN/Inf detected")


def assert_range(x: torch.Tensor, lo: float, hi: float, name: str):
    assert_finite(x, name)
    if (x < lo).any():
        raise AssertionError(f"{name}: below {lo}")
    if (x > hi).any():
        raise AssertionError(f"{name}: above {hi}")


def project_range(x: torch.Tensor, lo: float, hi: float):
    return x.clamp(lo, hi)

def stable_js(pA, logpA, pB, logpB, eps=1e-12):
    """
    Jensen–Shannon divergence (log base e).
    """
    m = 0.5 * (pA + pB)
    logm = torch.log(m + eps)

    klA = torch.sum(pA * (logpA - logm), dim=-1)
    klB = torch.sum(pB * (logpB - logm), dim=-1)

    return 0.5 * (klA + klB)


# ============================================================
# top-k metrics 
# ============================================================
def _compute_topk_metrics(
        pA: Any,
        pB: Any,
        tgt: Any,
        topk: Tuple[int, ...] = (1,5,10),
        eps:float = 1e-9
    ) -> Dict[str, dict]:
    L = pA.size(1)
    max_k = max(topk)

    vals_A, idx_A = torch.topk(pA, max_k, dim=-1)
    vals_B, idx_B = torch.topk(pB, max_k, dim=-1)

    idx_A = idx_A.cpu()
    idx_B = idx_B.cpu()
    vals_A = vals_A.cpu()
    vals_B = vals_B.cpu()
    tgt = tgt.cpu()

    out = {
        "acc_A": {},
        "acc_B": {},
        "jaccard": {},
        "agree_set": {},
        "disagree_set": {},
        "agree_correct": {},
        "disagree_correct": {},
        "agree_wrong": {},
        "prob_overlap": {},
    }

    for k in topk:
        key = f"@{k}"
        tkA = idx_A[:, :, :k]
        tkB = idx_B[:, :, :k]

        acc_A = (tkA == tgt.unsqueeze(-1)).any(-1).float()[0]
        acc_B = (tkB == tgt.unsqueeze(-1)).any(-1).float()[0]

        inter = []
        for i in range(L):
            sA = set(tkA[0, i].tolist())
            sB = set(tkB[0, i].tolist())
            inter.append(len(sA & sB))
        inter = torch.tensor(inter, dtype=torch.float32)

        jaccard = inter / (2 * k - inter + eps) 
        agree_set = (inter > 0).float()
        disagree_set = 1.0 - jaccard

        agree_correct = acc_A * acc_B
        disagree_correct = ((acc_A + acc_B) == 1).float()
        agree_wrong = ((1 - acc_A) * (1 - acc_B)).float()

        pmA = vals_A[0, :, :k].sum(-1)
        pmB = vals_B[0, :, :k].sum(-1)
        shared_mass = torch.zeros_like(pmA)

        for i in range(L):
            s = set(tkA[0, i].tolist()) & set(tkB[0, i].tolist())
            if s:
                shared_mass[i] = 0.5 * (
                    pA[0, i, list(s)].sum().cpu()
                  + pB[0, i, list(s)].sum().cpu()
                )

        prob_overlap = shared_mass / (0.5 * (pmA + pmB) + eps)

        out["acc_A"][key] = acc_A.tolist()
        out["acc_B"][key] = acc_B.tolist()
        out["jaccard"][key] = jaccard.tolist()
        out["agree_set"][key] = agree_set.tolist()
        out["disagree_set"][key] = disagree_set.tolist()
        out["agree_correct"][key] = agree_correct.tolist()
        out["disagree_correct"][key] = disagree_correct.tolist()
        out["agree_wrong"][key] = agree_wrong.tolist()
        out["prob_overlap"][key] = prob_overlap.tolist()

    return out


# ============================================================
# cross-model metrics 
# ============================================================
def _compute_cross(
    A: Dict[int, Dict[str, dict]],
    B: Dict[int, Dict[str, dict]],
    topk: Tuple = (1,5,10),
    save_path: str = None
) -> List:

    records = []

    for lid in sorted(set(A) & set(B)):
        for mode in A[lid].keys() & B[lid].keys():
            for pid in A[lid][mode].keys() & B[lid][mode].keys():

                a = A[lid][mode][pid]
                b = B[lid][mode][pid]

                logits_A = a["logits"]
                logits_B = b["logits"]
                tgt = a["target"]
                hidden_A = a["hidden"]
                hidden_B = b["hidden"]

                assert logits_A.size(1) == hidden_A.size(1) == tgt.size(1)

                L = min(logits_A.size(1), logits_B.size(1), tgt.size(1))
                logits_A = logits_A[:, :L]
                logits_B = logits_B[:, :L]
                tgt = tgt[:, :L]
                hidden_A = hidden_A[:, :L]
                hidden_B = hidden_B[:, :L]

                logpA = torch.log_softmax(logits_A, dim=-1)
                logpB = torch.log_softmax(logits_B, dim=-1)
                pA = logpA.exp()
                pB = logpB.exp()

                js = project_range(stable_js(pA, logpA, pB, logpB)[0], 0.0, 1.0)
                assert_range(js, 0.0, 1.0, "js")

                tvd = project_range(
                    0.5 * torch.sum(torch.abs(pA - pB), dim=-1)[0], 0.0, 1.0
                )
                assert_range(tvd, 0.0, 1.0, "tvd")

                cos = project_range(
                    F.cosine_similarity(hidden_A, hidden_B, dim=-1)[0], -1.0, 1.0
                )
                assert_range(cos, -1.0, 1.0, "cosine")

                l2 = torch.norm(hidden_A - hidden_B, dim=-1)[0]
                assert_finite(l2, "l2")

                pA_gt = pA[0, torch.arange(L), tgt[0]]
                pB_gt = pB[0, torch.arange(L), tgt[0]]
                gt_prob_diff = pA_gt - pB_gt
                assert_finite(gt_prob_diff, "gt_prob_diff")

                entropy_A = (-torch.sum(pA * logpA, dim=-1))[0]
                entropy_B = (-torch.sum(pB * logpB, dim=-1))[0]
                max_ent = math.log(logits_A.size(-1))
                entropy_A = project_range(entropy_A, 0.0, max_ent)
                entropy_B = project_range(entropy_B, 0.0, max_ent)
                assert_range(entropy_A, 0.0, max_ent, "entropy_A")
                assert_range(entropy_B, 0.0, max_ent, "entropy_B")

                nll_A = -logpA.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
                nll_B = -logpB.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
                nll_diff = nll_A - nll_B
                assert_finite(nll_A, "nll_A")
                assert_finite(nll_B, "nll_B")
                assert_finite(nll_diff, "nll_diff")

                ppl_A = torch.exp(nll_A.mean())
                ppl_B = torch.exp(nll_B.mean())
                ppl_diff = ppl_A - ppl_B
                assert_finite(ppl_A, "ppl_A")
                assert_finite(ppl_B, "ppl_B")

                top1_A = torch.argmax(pA, dim=-1)
                top1_B = torch.argmax(pB, dim=-1)
                xor_top1_correct = ((top1_A == tgt) ^ (top1_B == tgt)).float()[0]
                assert_finite(xor_top1_correct, "xor_top1_correct")

                # logits_A, logits_B: [1, L, V]
                top2_A, _ = torch.topk(logits_A, 2, dim=-1)
                top2_B, _ = torch.topk(logits_B, 2, dim=-1)

                margin_A = (top2_A[..., 0] - top2_A[..., 1])[0]  # [L]
                margin_B = (top2_B[..., 0] - top2_B[..., 1])[0]  # [L]

                margin_diff = margin_A - margin_B

                # hidden_A, hidden_B: [1, L, D]
                norm_A = torch.norm(hidden_A, dim=-1)[0]  # [L]
                norm_B = torch.norm(hidden_B, dim=-1)[0]  # [L]

                norm_ratio = norm_A / (norm_B + 1e-9)
                norm_diff = norm_A - norm_B

                # ---------- top-k ----------
                topk_metrics = _compute_topk_metrics(pA, pB, tgt, topk=topk)
                for k, v in topk_metrics.items():
                    if torch.is_tensor(v):
                        assert_finite(v, k)

                rec = {
                    "batch_index": a["batch_index"],
                    "prompt_id": a["prompt_id"],
                    "layer_index": lid,
                    "layer_name": a["layer_name"],
                    "mode": mode,

                    # per-position arrays
                    "js": js.tolist(),
                    "tvd": tvd.tolist(),
                    "cosine": cos.tolist(),
                    "l2": l2.tolist(),
                    "gt_prob_diff": gt_prob_diff.tolist(),
                    "entropy_A": entropy_A.tolist(),
                    "entropy_B": entropy_B.tolist(),
                    "nll_A": nll_A.tolist(),
                    "nll_B": nll_B.tolist(),
                    "nll_diff": nll_diff.tolist(),
                    "ppl_A": ppl_A.tolist(),
                    "ppl_B": ppl_B.tolist(),
                    "ppl_diff": ppl_diff.tolist(),
                    "xor_top1_correct": xor_top1_correct.tolist(),

                    "margin_A": margin_A.tolist(),
                    "margin_B": margin_B.tolist(),
                    "margin_diff": margin_diff.tolist(),
                    "hidden_norm_A": norm_A.tolist(),
                    "hidden_norm_B": norm_B.tolist(),
                    "hidden_norm_diff": norm_diff.tolist(),
                    "hidden_norm_ratio": norm_ratio.tolist(),

                    # TOP-K
                    "acc_A": topk_metrics["acc_A"],
                    "acc_B": topk_metrics["acc_B"],
                    "jaccard": topk_metrics["jaccard"],
                    "agree_set": topk_metrics["agree_set"],
                    "disagree_set": topk_metrics["disagree_set"],
                    "agree_correct": topk_metrics["agree_correct"],
                    "disagree_correct": topk_metrics["disagree_correct"],
                    "agree_wrong": topk_metrics["agree_wrong"],
                    "prob_overlap": topk_metrics["prob_overlap"],
                }

                records.append(rec)

    """
    all_records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                all_records.append(json.loads(line))
    """
    """if save_path:
        save_jsonl(records=records, path=save_path)"""

    return records


@torch.no_grad()
def apply_ldl(
    dir_A: str,
    dir_B: str,
    output_dir: str = None,
    norm_modes: Tuple[str, ...] = ("raw", "unit_norm", "eps_norm", "model_norm"),
    topk: Tuple[int, ...] = (1, 5, 10),
    run_name: str = None,
    device: str | None = None,
    debug: bool = True,
) -> None:

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    files_A = sorted(f for f in os.listdir(dir_A) if f.endswith(".pt"))
    files_B = sorted(f for f in os.listdir(dir_B) if f.endswith(".pt"))
    assert len(files_A) == len(files_B), "Mismatch in number of batch files"

    print(f"[INFO] Found {len(files_A)} batch pairs")

    for batch_idx, (fa, fb) in enumerate(tqdm(zip(files_A, files_B), total=len(files_A))):
        path_A = os.path.join(dir_A, fa)
        path_B = os.path.join(dir_B, fb)

        if debug:
            print(f"\n[batch {batch_idx}] {fa} vs {fb}")

        # ---------- LOAD ----------
        obj_A = torch.load(path_A, weights_only=False, map_location="cpu")
        obj_B = torch.load(path_B, weights_only=False, map_location="cpu")

        rows_A = obj_A["rows"]
        rows_B = obj_B["rows"]

        # ---------- PREPROCESS ----------
        layers_A = _preprocess_rows(rows_A, norm_modes=norm_modes)
        layers_B = _preprocess_rows(rows_B, norm_modes=norm_modes)

        # ---------- COMPUTE ----------
        records = _compute_cross(
            layers_A,
            layers_B,
            topk=topk,
        )

        # ---------- SAVE ----------
        if output_dir and run_name:
            out_path = os.path.join(
                output_dir,
                f"{run_name}_batch{batch_idx:03d}.jsonl"
            )

            with open(out_path, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

        if debug:
            print(f"[SAVED] {out_path}  ({len(records)} rows)")

        # ---------- CLEANUP ----------
        del obj_A, obj_B, rows_A, rows_B
        del layers_A, layers_B, records
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
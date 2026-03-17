from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F
import os, gc, json, math
from pathlib import Path
from tqdm import tqdm

from ..wrapper.arch_wrapper import ArchWrapper



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
# preprocess rows
# ============================================================
def _preprocess_rows(rows: List[dict], norm_modes) -> Dict:
    layers = {}

    for r in rows:
        lid = r["layer_index"]
        if lid == -1:
            continue

        pid = r["prompt_id"]

        tokens = _ensure_2d(r["tokens"])

        input_tokens = tokens[:, :-1]
        target = tokens[:, 1:]

        for k, v in r.items():
            if not k.startswith("hidden_") or v is None:
                continue

            mode = k.replace("hidden_", "")
            if mode not in norm_modes:
                continue

            hidden = _ensure_3d(v)[:, :-1]

            L = min(hidden.size(1), target.size(1))

            layers \
                .setdefault(lid, {}) \
                .setdefault(mode, {}) \
                [pid] = {
                    "hidden": hidden[:, :L].float(),
                    "input_tokens": input_tokens[:, :L].float(),
                    "target": target[:, :L].long(),
                    "batch_index": r["batch_index"],
                    "prompt_id": pid,
                    "layer_name": r["layer_name"],
                }

    return layers

# ============================================================
# ADL
# ============================================================
@torch.no_grad()
def _apply_adl(
    arch_wrapper,
    layers_A,
    layers_B,
    norm_modes=("raw", "model_norm"),
    topk=10,
):

    rows = []

    for lid in sorted(layers_A.keys()):

        for mode in norm_modes:

            if mode not in layers_A[lid] or mode not in layers_B[lid]:
                continue

            prompts = set(layers_A[lid][mode]).intersection(
                layers_B[lid][mode]
            )

            for pid in prompts:

                rec_A = layers_A[lid][mode][pid]
                rec_B = layers_B[lid][mode][pid]

                hA = rec_A["hidden"]
                hB = rec_B["hidden"]

                # Δh
                delta = hA - hB
                delta_norm = torch.norm(delta, dim=-1)  # [B, L]

                # projections
                logits_delta, _ = arch_wrapper.project(delta)
                logits_A, _ = arch_wrapper.project(hA)
                logits_B, _ = arch_wrapper.project(hB)

                # ensure 3D
                if logits_delta.dim() == 2:
                    logits_delta = logits_delta.unsqueeze(0)
                if logits_A.dim() == 2:
                    logits_A = logits_A.unsqueeze(0)
                if logits_B.dim() == 2:
                    logits_B = logits_B.unsqueeze(0)

                # ---------------------------
                # ADL DISTRIBUTION
                # ---------------------------
                p_delta = torch.softmax(logits_delta, dim=-1)

                # ---------------------------
                # ADL METRICS
                # ---------------------------

                # strength
                delta_logit_max = logits_delta.max(dim=-1).values

                # entropy (Δ space)
                entropy_delta = -(p_delta * torch.log(p_delta + 1e-9)).sum(dim=-1)

                # ground truth prob (Δ)
                target = rec_A["target"]
                if target.dim() == 1:
                    target = target.unsqueeze(0)

                gt_prob_delta = torch.gather(
                    p_delta, -1, target.unsqueeze(-1)
                ).squeeze(-1)

                # KL(Δ || A)
                p_A = torch.softmax(logits_A, dim=-1)
                kl_delta_vs_A = (
                    p_delta * (torch.log(p_delta + 1e-9) - torch.log(p_A + 1e-9))
                ).sum(dim=-1)

                # ---------------------------
                # TOP-K
                # ---------------------------
                topk_vals, topk_ids = torch.topk(
                    logits_delta, k=topk, dim=-1
                )

                rows.append({
                    "prompt_id": pid,
                    "layer_index": lid,
                    "mode": mode,

                    "tokens": rec_A["input_tokens"].cpu(),
                    "target_tokens": rec_A["target"].cpu(),

                    # ALL LOGITS 
                    "logits_delta": logits_delta.cpu(),
                    "logits_A": logits_A.cpu(),
                    "logits_B": logits_B.cpu(),

                    # ADL core
                    "delta_logit_max": delta_logit_max.cpu(),
                    "delta_norm": delta_norm.cpu(),

                    # ADL distribution metrics
                    "entropy_delta": entropy_delta.cpu(),
                    "gt_prob_delta": gt_prob_delta.cpu(),
                    "kl_delta_vs_A": kl_delta_vs_A.cpu(),

                    # hover
                    "topk_ids": topk_ids.cpu(),
                    "topk_vals": topk_vals.cpu(),
                })

    return rows


# ============================================================
# Run ADL
# ============================================================
def apply_adl(
    arch_wrapper:"ArchWrapper",
    dir_A:str,
    dir_B:str,
    output_dir:str|Path,
    norm_modes:Tuple[str, ...]=("raw", "model_norm"),
):

    os.makedirs(output_dir, exist_ok=True)

    files_A = sorted(f for f in os.listdir(dir_A) if f.endswith(".pt"))
    files_B = sorted(f for f in os.listdir(dir_B) if f.endswith(".pt"))

    assert len(files_A) == len(files_B)

    print(f"[ADL] Running apply ADL...")

    for batch_idx, (fa, fb) in enumerate(zip(files_A, files_B)):

        print(f"Processing batch {batch_idx}")

        path_A = os.path.join(dir_A, fa)
        path_B = os.path.join(dir_B, fb)

        obj_A = torch.load(path_A, map_location="cpu")
        obj_B = torch.load(path_B, map_location="cpu")

        rows_A = obj_A["rows"]
        rows_B = obj_B["rows"]

        layers_A = _preprocess_rows(rows_A, norm_modes)
        layers_B = _preprocess_rows(rows_B, norm_modes)

        records = _apply_adl(
            arch_wrapper,
            layers_A,
            layers_B,
            norm_modes=norm_modes,
        )

        out_path = os.path.join(
            output_dir,
            f"adl_batch_{batch_idx:03d}.pt"
        )

        torch.save({"rows": records}, out_path)

        print("Saved:", out_path)

        del obj_A, obj_B, layers_A, layers_B, records
        gc.collect()


def apply_adl_plotter(
    arch_wrapper,
    dir_A,
    dir_B,
    output_dir,
    norm_modes=("raw", "model_norm"),
):

    os.makedirs(output_dir, exist_ok=True)

    files_A = sorted(f for f in os.listdir(dir_A) if f.endswith(".pt"))
    files_B = sorted(f for f in os.listdir(dir_B) if f.endswith(".pt"))

    assert len(files_A) == len(files_B)

    all_records = []

    for batch_idx, (fa, fb) in enumerate(zip(files_A, files_B)):

        path_A = os.path.join(dir_A, fa)
        path_B = os.path.join(dir_B, fb)

        obj_A = torch.load(path_A, map_location="cpu")
        obj_B = torch.load(path_B, map_location="cpu")

        layers_A = _preprocess_rows(obj_A["rows"], norm_modes)
        layers_B = _preprocess_rows(obj_B["rows"], norm_modes)

        records = _apply_adl(
            arch_wrapper,
            layers_A,
            layers_B,
            norm_modes=norm_modes,
        )

        all_records.extend(records)

        torch.save(
            {"rows": records},
            os.path.join(output_dir, f"adl_batch_{batch_idx:03d}.pt")
        )

        del obj_A, obj_B, layers_A, layers_B, records
        gc.collect()

    return {"rows": all_records}
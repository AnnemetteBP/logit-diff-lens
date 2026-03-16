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
def ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unexpected shape: {x.shape}")


def ensure_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2:
        return t
    if t.dim() == 1:
        return t.unsqueeze(0)
    raise ValueError(f"Expected [B,L], got {t.shape}")

# ============================================================
# preprocess rows
# ============================================================
def preprocess_rows(rows: List[dict], norm_modes) -> Dict:
    layers = {}

    for r in rows:
        lid = r["layer_index"]
        if lid == -1:
            continue

        pid = r["prompt_id"]

        tokens = ensure_2d(r["tokens"])

        target = tokens[:, 1:]

        for k, v in r.items():
            if not k.startswith("hidden_") or v is None:
                continue

            mode = k.replace("hidden_", "")
            if mode not in norm_modes:
                continue

            hidden = ensure_3d(v)[:, :-1]

            L = min(hidden.size(1), target.size(1))

            layers \
                .setdefault(lid, {}) \
                .setdefault(mode, {}) \
                [pid] = {
                    "hidden": hidden[:, :L].float(),
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

                # ADL difference
                delta = hA - hB
                delta_norm = torch.norm(delta, dim=-1).mean().item()

                # project via LM head
                logits, _ = arch_wrapper.project(delta)

                rows.append({
                    "prompt_id": pid,
                    "layer_index": lid,
                    "mode": mode,
                    "logits": logits.detach().cpu(),
                    "delta_norm": delta_norm
                })

    return rows



# ============================================================
# Run ADL
# ============================================================
def apply_adl(
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

    for batch_idx, (fa, fb) in enumerate(zip(files_A, files_B)):

        print(f"Processing batch {batch_idx}")

        path_A = os.path.join(dir_A, fa)
        path_B = os.path.join(dir_B, fb)

        obj_A = torch.load(path_A, map_location="cpu")
        obj_B = torch.load(path_B, map_location="cpu")

        rows_A = obj_A["rows"]
        rows_B = obj_B["rows"]

        layers_A = preprocess_rows(rows_A, norm_modes)
        layers_B = preprocess_rows(rows_B, norm_modes)

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
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def plot_annotated_geometry(
    *,
    aligned_pair_path: Path,
    svd_summary_path: Path,
    high_js_path: Path,
    output_path: Path,
    layer: int = 27,
    max_labels: int = 8,
) -> None:
    svd = _load_json(svd_summary_path)
    high_js = _load_json(high_js_path)
    aligned = torch.load(aligned_pair_path, map_location="cpu")
    aligned_rows = list(aligned["rows"])
    row_index_by_id = {int(row["id"]): idx for idx, row in enumerate(aligned_rows)}

    logit_layers = svd["logit_by_layer"]
    if not (0 <= layer < len(logit_layers)):
        raise ValueError(f"Layer {layer} out of range for {len(logit_layers)} layers")

    layer_summary = logit_layers[layer]
    scores = np.asarray(layer_summary["top_component_scores"], dtype=np.float32)
    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError("Expected 2D top_component_scores with at least one row")

    if scores.shape[1] == 1:
        xs = scores[:, 0]
        ys = np.zeros_like(xs)
    else:
        xs = scores[:, 0]
        ys = scores[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xs, ys, s=22, alpha=0.35, color="#6c757d", label="All examples")

    labeled = []
    seen_groups = set()
    for ex in high_js["highest_js_examples"]:
        if int(ex["layer"]) != layer:
            continue
        group_id = str(ex["group_id"])
        if group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        labeled.append(ex)
        if len(labeled) >= max_labels:
            break

    for ex in labeled:
        row_id = int(ex["row_id"])
        if row_id not in row_index_by_id:
            continue
        idx = row_index_by_id[row_id]
        x = float(xs[idx])
        y = float(ys[idx])
        label = f"{ex['input_token']}: {ex['top1_base_decoded']} -> {ex['top1_ft_decoded']}"
        ax.scatter([x], [y], s=55, color="#d62728", zorder=3)
        ax.annotate(
            label,
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#d62728", "alpha": 0.85},
            arrowprops={"arrowstyle": "-", "color": "#d62728", "lw": 0.8},
        )

    ax.axhline(0.0, color="0.85", linewidth=1.0)
    ax.axvline(0.0, color="0.85", linewidth=1.0)
    ax.set_title(f"Annotated logit-delta geometry, layer {layer}")
    ax.set_xlabel("Component 1 score")
    ax.set_ylabel("Component 2 score")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot annotated cross-condition geometry")
    parser.add_argument("--aligned-pair-path", required=True)
    parser.add_argument("--svd-summary-path", required=True)
    parser.add_argument("--high-js-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--max-labels", type=int, default=8)
    args = parser.parse_args()

    plot_annotated_geometry(
        aligned_pair_path=Path(args.aligned_pair_path),
        svd_summary_path=Path(args.svd_summary_path),
        high_js_path=Path(args.high_js_path),
        output_path=Path(args.output_path),
        layer=args.layer,
        max_labels=args.max_labels,
    )

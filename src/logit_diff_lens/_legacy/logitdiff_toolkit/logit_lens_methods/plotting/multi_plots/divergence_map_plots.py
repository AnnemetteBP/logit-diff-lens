from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def summarize_divergence_maps(
    *,
    divergence_jsonl_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with divergence_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No rows found in {divergence_jsonl_path}")

    max_layer = max(int(r["layer"]) for r in rows)
    max_pos = max(int(r["position"]) for r in rows)
    num_layers = max_layer + 1
    num_positions = max_pos + 1

    js_sum = np.zeros((num_layers, num_positions), dtype=np.float64)
    tvd_sum = np.zeros((num_layers, num_positions), dtype=np.float64)
    count = np.zeros((num_layers, num_positions), dtype=np.float64)

    by_layer_js: list[list[float]] = [[] for _ in range(num_layers)]
    by_layer_tvd: list[list[float]] = [[] for _ in range(num_layers)]
    by_pos_js: list[list[float]] = [[] for _ in range(num_positions)]
    by_pos_tvd: list[list[float]] = [[] for _ in range(num_positions)]

    category_counts: dict[str, int] = {}
    variant_counts: dict[str, int] = {}

    for row in rows:
        layer = int(row["layer"])
        pos = int(row["position"])
        js = float(row.get("js", 0.0))
        tvd = float(row.get("tvd", 0.0))
        js_sum[layer, pos] += js
        tvd_sum[layer, pos] += tvd
        count[layer, pos] += 1.0
        by_layer_js[layer].append(js)
        by_layer_tvd[layer].append(tvd)
        by_pos_js[pos].append(js)
        by_pos_tvd[pos].append(tvd)
        category = str(row.get("category"))
        variant = str(row.get("variant"))
        category_counts[category] = category_counts.get(category, 0) + 1
        variant_counts[variant] = variant_counts.get(variant, 0) + 1

    js_mean = np.divide(js_sum, np.maximum(count, 1.0))
    tvd_mean = np.divide(tvd_sum, np.maximum(count, 1.0))
    layer_js = np.asarray([float(np.mean(v)) if v else 0.0 for v in by_layer_js], dtype=np.float64)
    layer_tvd = np.asarray([float(np.mean(v)) if v else 0.0 for v in by_layer_tvd], dtype=np.float64)
    pos_js = np.asarray([float(np.mean(v)) if v else 0.0 for v in by_pos_js], dtype=np.float64)
    pos_tvd = np.asarray([float(np.mean(v)) if v else 0.0 for v in by_pos_tvd], dtype=np.float64)

    output_dir.mkdir(parents=True, exist_ok=True)

    def _save_heatmap(matrix: np.ndarray, title: str, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(11, 6))
        im = ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Position")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)

    _save_heatmap(js_mean, "Mean JS by layer and position", output_dir / "layer_position_js_heatmap.png")
    _save_heatmap(tvd_mean, "Mean TVD by layer and position", output_dir / "layer_position_tvd_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(np.arange(num_layers), layer_js, linewidth=2)
    axes[0].set_title("Mean JS by layer")
    axes[0].set_xlabel("Layer")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(np.arange(num_layers), layer_tvd, linewidth=2, color="#d62728")
    axes[1].set_title("Mean TVD by layer")
    axes[1].set_xlabel("Layer")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_divergence_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(np.arange(num_positions), pos_js, linewidth=2)
    axes[0].set_title("Mean JS by position")
    axes[0].set_xlabel("Position")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(np.arange(num_positions), pos_tvd, linewidth=2, color="#d62728")
    axes[1].set_title("Mean TVD by position")
    axes[1].set_xlabel("Position")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "position_divergence_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    late_start = num_layers // 2
    late_rows = [r for r in rows if int(r["layer"]) >= late_start]
    summary = {
        "divergence_jsonl_path": str(divergence_jsonl_path),
        "num_rows": len(rows),
        "num_layers": num_layers,
        "num_positions": num_positions,
        "category_counts": category_counts,
        "variant_counts": variant_counts,
        "peak_js_layer": int(np.argmax(layer_js)),
        "peak_tvd_layer": int(np.argmax(layer_tvd)),
        "peak_js_position": int(np.argmax(pos_js)),
        "peak_tvd_position": int(np.argmax(pos_tvd)),
        "late_layer_start": int(late_start),
        "late_layer_mean_js": float(np.mean([float(r.get("js", 0.0)) for r in late_rows])) if late_rows else 0.0,
        "late_layer_mean_tvd": float(np.mean([float(r.get("tvd", 0.0)) for r in late_rows])) if late_rows else 0.0,
        "layer_js": layer_js.tolist(),
        "layer_tvd": layer_tvd.tolist(),
        "position_js": pos_js.tolist(),
        "position_tvd": pos_tvd.tolist(),
    }
    (output_dir / "divergence_map_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate layer/position divergence summaries and plots")
    parser.add_argument("--divergence-jsonl-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize_divergence_maps(
        divergence_jsonl_path=Path(args.divergence_jsonl_path),
        output_dir=Path(args.output_dir),
    )

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def summarize_divergence_by_category(
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

    output_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted({str(r.get("category")) for r in rows})
    num_layers = max(int(r["layer"]) for r in rows) + 1
    num_positions = max(int(r["position"]) for r in rows) + 1
    late_start = num_layers // 2

    summary_rows: list[dict[str, Any]] = []
    category_layer_js: dict[str, np.ndarray] = {}
    category_layer_tvd: dict[str, np.ndarray] = {}
    category_pos_js: dict[str, np.ndarray] = {}
    category_pos_tvd: dict[str, np.ndarray] = {}

    for category in categories:
        subset = [r for r in rows if str(r.get("category")) == category]
        by_layer_js = [[] for _ in range(num_layers)]
        by_layer_tvd = [[] for _ in range(num_layers)]
        by_pos_js = [[] for _ in range(num_positions)]
        by_pos_tvd = [[] for _ in range(num_positions)]
        for row in subset:
            layer = int(row["layer"])
            pos = int(row["position"])
            js = float(row.get("js", 0.0))
            tvd = float(row.get("tvd", 0.0))
            by_layer_js[layer].append(js)
            by_layer_tvd[layer].append(tvd)
            by_pos_js[pos].append(js)
            by_pos_tvd[pos].append(tvd)

        layer_js = np.asarray([float(np.mean(v)) if v else 0.0 for v in by_layer_js], dtype=np.float64)
        layer_tvd = np.asarray([float(np.mean(v)) if v else 0.0 for v in by_layer_tvd], dtype=np.float64)
        pos_js = np.asarray([float(np.mean(v)) if v else 0.0 for v in by_pos_js], dtype=np.float64)
        pos_tvd = np.asarray([float(np.mean(v)) if v else 0.0 for v in by_pos_tvd], dtype=np.float64)

        category_layer_js[category] = layer_js
        category_layer_tvd[category] = layer_tvd
        category_pos_js[category] = pos_js
        category_pos_tvd[category] = pos_tvd

        late_rows = [r for r in subset if int(r["layer"]) >= late_start]
        summary_rows.append(
            {
                "category": category,
                "num_rows": len(subset),
                "peak_js_layer": int(np.argmax(layer_js)),
                "peak_tvd_layer": int(np.argmax(layer_tvd)),
                "peak_js_position": int(np.argmax(pos_js)),
                "peak_tvd_position": int(np.argmax(pos_tvd)),
                "late_layer_mean_js": float(np.mean([float(r.get("js", 0.0)) for r in late_rows])) if late_rows else 0.0,
                "late_layer_mean_tvd": float(np.mean([float(r.get("tvd", 0.0)) for r in late_rows])) if late_rows else 0.0,
                "layer_js": layer_js.tolist(),
                "layer_tvd": layer_tvd.tolist(),
                "position_js": pos_js.tolist(),
                "position_tvd": pos_tvd.tolist(),
            }
        )

    summary_rows.sort(key=lambda r: r["late_layer_mean_js"], reverse=True)
    top_categories = [row["category"] for row in summary_rows[: min(6, len(summary_rows))]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for category in top_categories:
        axes[0].plot(np.arange(num_layers), category_layer_js[category], linewidth=2, label=category)
        axes[1].plot(np.arange(num_layers), category_layer_tvd[category], linewidth=2, label=category)
    axes[0].set_title("Mean JS by layer for top categories")
    axes[1].set_title("Mean TVD by layer for top categories")
    for ax in axes:
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "category_layer_curves_top6.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for category in top_categories:
        axes[0].plot(np.arange(num_positions), category_pos_js[category], linewidth=2, label=category)
        axes[1].plot(np.arange(num_positions), category_pos_tvd[category], linewidth=2, label=category)
    axes[0].set_title("Mean JS by position for top categories")
    axes[1].set_title("Mean TVD by position for top categories")
    for ax in axes:
        ax.set_xlabel("Position")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "category_position_curves_top6.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.4 * len(summary_rows) + 1)))
    cats = [row["category"] for row in summary_rows]
    js_vals = [row["late_layer_mean_js"] for row in summary_rows]
    tvd_vals = [row["late_layer_mean_tvd"] for row in summary_rows]
    y = np.arange(len(summary_rows))
    axes[0].barh(y, js_vals, color="#1f77b4")
    axes[0].set_yticks(y, labels=cats)
    axes[0].invert_yaxis()
    axes[0].set_title("Late-layer mean JS by category")
    axes[1].barh(y, tvd_vals, color="#d62728")
    axes[1].set_yticks(y, labels=cats)
    axes[1].invert_yaxis()
    axes[1].set_title("Late-layer mean TVD by category")
    fig.tight_layout()
    fig.savefig(output_dir / "category_late_layer_bars.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    output = {
        "divergence_jsonl_path": str(divergence_jsonl_path),
        "num_layers": num_layers,
        "num_positions": num_positions,
        "late_layer_start": late_start,
        "top_categories_by_late_js": top_categories,
        "categories": summary_rows,
    }
    (output_dir / "category_divergence_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate category-stratified divergence summaries")
    parser.add_argument("--divergence-jsonl-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize_divergence_by_category(
        divergence_jsonl_path=Path(args.divergence_jsonl_path),
        output_dir=Path(args.output_dir),
    )

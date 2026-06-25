from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def summarize_prompt_vs_response(
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

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["row_id"]), []).append(row)

    prompt_response_rows: list[dict[str, Any]] = []
    for group in grouped.values():
        group_sorted = sorted(group, key=lambda r: (int(r["position"]), int(r["layer"])))
        analysis_text = str(group_sorted[0].get("analysis_text", ""))
        first_space = analysis_text.find(" ")
        split_pos = 0
        if first_space >= 0:
            max_pos = max(int(r["position"]) for r in group_sorted)
            split_pos = max(0, int(round(0.35 * max_pos)))
        for row in group_sorted:
            row = dict(row)
            row["segment"] = "prompt" if int(row["position"]) <= split_pos else "response"
            prompt_response_rows.append(row)

    segments = ["prompt", "response"]
    summary: dict[str, Any] = {
        "divergence_jsonl_path": str(divergence_jsonl_path),
        "segments": {},
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, segment in enumerate(segments):
        subset = [r for r in prompt_response_rows if r["segment"] == segment]
        if not subset:
            continue
        num_layers = max(int(r["layer"]) for r in subset) + 1
        num_positions = max(int(r["position"]) for r in subset) + 1
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

        summary["segments"][segment] = {
            "num_rows": len(subset),
            "peak_js_layer": int(np.argmax(layer_js)),
            "peak_tvd_layer": int(np.argmax(layer_tvd)),
            "peak_js_position": int(np.argmax(pos_js)),
            "peak_tvd_position": int(np.argmax(pos_tvd)),
            "mean_js": float(np.mean([float(r.get("js", 0.0)) for r in subset])),
            "mean_tvd": float(np.mean([float(r.get("tvd", 0.0)) for r in subset])),
            "layer_js": layer_js.tolist(),
            "layer_tvd": layer_tvd.tolist(),
            "position_js": pos_js.tolist(),
            "position_tvd": pos_tvd.tolist(),
        }

        axes[0, idx].plot(np.arange(len(layer_js)), layer_js, linewidth=2, label="JS")
        axes[0, idx].plot(np.arange(len(layer_tvd)), layer_tvd, linewidth=2, label="TVD")
        axes[0, idx].set_title(f"{segment.title()} segment by layer")
        axes[0, idx].set_xlabel("Layer")
        axes[0, idx].grid(True, alpha=0.25)
        axes[0, idx].legend()

        axes[1, idx].plot(np.arange(len(pos_js)), pos_js, linewidth=2, label="JS")
        axes[1, idx].plot(np.arange(len(pos_tvd)), pos_tvd, linewidth=2, label="TVD")
        axes[1, idx].set_title(f"{segment.title()} segment by position")
        axes[1, idx].set_xlabel("Position")
        axes[1, idx].grid(True, alpha=0.25)
        axes[1, idx].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "prompt_vs_response_divergence.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    (output_dir / "prompt_vs_response_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize prompt vs response divergence")
    parser.add_argument("--divergence-jsonl-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summarize_prompt_vs_response(
        divergence_jsonl_path=Path(args.divergence_jsonl_path),
        output_dir=Path(args.output_dir),
    )

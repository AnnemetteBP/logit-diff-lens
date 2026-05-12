from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np


def _load_payload(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _template_label(template_name: str) -> str:
    mapping = {
        "qwen_chat_template": "Original Qwen template",
        "neutral_chat_template": "Neutral chat template",
        "no_template": "No template",
    }
    return mapping.get(template_name, template_name.replace("_", " ").title())


def _pick_layer_indices(num_layers: int) -> dict[str, int]:
    last = num_layers - 1
    return {
        "first": 0,
        "early": round(last * 0.25),
        "mid": round(last * 0.50),
        "late": round(last * 0.75),
        "last": last,
    }


def summarize_generation_topk_table(
    payload_or_path: str | Path | dict[str, Any],
    *,
    topk: int,
    generated_only: bool = True,
) -> dict[str, Any]:
    payload = (
        _load_payload(payload_or_path)
        if isinstance(payload_or_path, (str, Path))
        else payload_or_path
    )
    metadata = payload["metadata"]
    rows = payload["analysis_rows"]
    if generated_only:
        rows = [row for row in rows if row.get("is_generated", False)]
    if not rows:
        raise ValueError("No rows matched the requested filter")

    max_layer = max(int(row["layer_absolute"]) for row in rows)
    num_layers = max_layer + 1
    layer_targets = _pick_layer_indices(num_layers)

    by_layer: dict[int, list[float]] = {}
    for row in rows:
        layer = int(row["layer_absolute"])
        by_layer.setdefault(layer, []).append(float(row[f"top{topk}_jaccard"]))

    layer_means = {layer: float(np.mean(vals)) for layer, vals in by_layer.items()}
    summary = {
        "template_name": metadata.get("template_name"),
        "template_label": _template_label(metadata.get("template_name", "template")),
        "generated_only": generated_only,
        "top_k": metadata.get("top_k"),
        "reported_topk": topk,
        "layer_targets": layer_targets,
        "values": {
            key: layer_means[idx]
            for key, idx in layer_targets.items()
        },
        "overall_mean": float(np.mean([float(row[f"top{topk}_jaccard"]) for row in rows])),
    }
    return summary


def build_generation_topk_latex_table(
    payloads_or_paths: list[str | Path | dict[str, Any]],
    *,
    topk: int,
    generated_only: bool = True,
) -> tuple[str, dict[str, Any]]:
    summaries = [
        summarize_generation_topk_table(p, topk=topk, generated_only=generated_only)
        for p in payloads_or_paths
    ]

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Template & First & Early & Mid & Late & Last & Overall \\",
        r"\midrule",
    ]
    for summary in summaries:
        vals = summary["values"]
        lines.append(
            (
                f"{summary['template_label']} & "
                f"{vals['first']:.3f} & "
                f"{vals['early']:.3f} & "
                f"{vals['mid']:.3f} & "
                f"{vals['late']:.3f} & "
                f"{vals['last']:.3f} & "
                f"{summary['overall_mean']:.3f} \\\\"
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines), {"summaries": summaries}


def save_generation_topk_latex_table(
    payloads_or_paths: list[str | Path | dict[str, Any]],
    *,
    topk: int,
    output_tex_path: str | Path,
    output_json_path: str | Path | None = None,
    generated_only: bool = True,
) -> dict[str, Any]:
    latex, payload = build_generation_topk_latex_table(
        payloads_or_paths,
        topk=topk,
        generated_only=generated_only,
    )
    tex_path = Path(output_tex_path)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(latex + "\n", encoding="utf-8")

    if output_json_path is not None:
        json_path = Path(output_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "latex_path": str(tex_path),
        "json_path": str(output_json_path) if output_json_path is not None else None,
        **payload,
    }


def summarize_generation_top10_table(
    payload_or_path: str | Path | dict[str, Any],
    *,
    generated_only: bool = True,
) -> dict[str, Any]:
    return summarize_generation_topk_table(
        payload_or_path,
        topk=10,
        generated_only=generated_only,
    )


def build_generation_top10_latex_table(
    payloads_or_paths: list[str | Path | dict[str, Any]],
    *,
    generated_only: bool = True,
) -> tuple[str, dict[str, Any]]:
    return build_generation_topk_latex_table(
        payloads_or_paths,
        topk=10,
        generated_only=generated_only,
    )


def save_generation_top10_latex_table(
    payloads_or_paths: list[str | Path | dict[str, Any]],
    *,
    output_tex_path: str | Path,
    output_json_path: str | Path | None = None,
    generated_only: bool = True,
) -> dict[str, Any]:
    return save_generation_topk_latex_table(
        payloads_or_paths,
        topk=10,
        output_tex_path=output_tex_path,
        output_json_path=output_json_path,
        generated_only=generated_only,
    )

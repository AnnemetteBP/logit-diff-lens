from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ...tokenizer_loading import load_tokenizer


def _render_token(tokenizer: Any, token_id: int) -> str:
    decoded = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    raw = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    text = decoded if decoded else str(raw)
    return text.replace("\n", "\\n")


def _is_meaningful_token(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) == 1 and not stripped.isalnum():
        return False
    if re.fullmatch(r"[_=\-+/\\|:;,.!?()\[\]{}<>@#$%^&*~`'\"]+", stripped):
        return False
    if not any(ch.isalpha() for ch in stripped):
        return False
    return True


def _top_token_loadings(vector: np.ndarray, tokenizer: Any, top_k: int) -> dict[str, list[dict[str, Any]]]:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    pos_order = np.argsort(-vec)
    neg_order = np.argsort(vec)

    def _collect(order: np.ndarray) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for token_id in order.tolist():
            text = _render_token(tokenizer, int(token_id))
            if not _is_meaningful_token(text):
                continue
            out.append(
                {
                    "token_id": int(token_id),
                    "token": text,
                    "loading": float(vec[int(token_id)]),
                }
            )
            if len(out) >= top_k:
                break
        return out

    return {"positive": _collect(pos_order), "negative": _collect(neg_order)}


def run_focused_check(
    *,
    svd_summary_path: Path,
    divergence_jsonl_path: Path,
    tokenizer_path: str,
    output_path: Path,
    top_k_tokens: int = 12,
) -> dict[str, Any]:
    svd_summary = json.loads(svd_summary_path.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(tokenizer_path)

    with divergence_jsonl_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError("No divergence records found")

    max_layer = max(int(rec["layer"]) for rec in records)
    late_threshold = max_layer // 2
    late_changed = [
        rec
        for rec in records
        if int(rec["layer"]) >= late_threshold and rec.get("top1_token_base") != rec.get("top1_token_ft")
    ]
    late_top5_changed = [
        rec
        for rec in records
        if int(rec["layer"]) >= late_threshold
        and set(rec.get("top5_tokens_base", [])) != set(rec.get("top5_tokens_ft", []))
    ]

    js_values = [float(rec.get("js", 0.0)) for rec in late_changed]
    tvd_values = [float(rec.get("tvd", 0.0)) for rec in late_changed]

    def _counter(key: str, rows: list[dict[str, Any]], top_n: int = 20) -> list[dict[str, Any]]:
        c = Counter(str(rec.get(key, "")) for rec in rows if str(rec.get(key, "")).strip())
        return [{"token": token, "count": int(count)} for token, count in c.most_common(top_n)]

    transitions = Counter(
        (str(rec.get("top1_token_base", "")), str(rec.get("top1_token_ft", "")))
        for rec in late_changed
    )
    top_transitions = [
        {"base_top1": a, "ft_top1": b, "count": int(count)}
        for (a, b), count in transitions.most_common(20)
    ]

    logit_layers = svd_summary.get("logit_by_layer", [])
    chosen_layers = sorted(set([late_threshold, max(late_threshold, len(logit_layers) // 2), max(0, len(logit_layers) - 1)]))
    chosen_layers = [layer for layer in chosen_layers if 0 <= layer < len(logit_layers)]

    direction_summary: list[dict[str, Any]] = []
    for layer_idx in chosen_layers:
        layer = logit_layers[layer_idx]
        vh = np.asarray(layer.get("top_right_singular_vectors_vh", []), dtype=np.float32)
        layer_entry = {"layer": int(layer_idx), "components": []}
        if vh.ndim == 2 and vh.size > 0:
            for comp_idx in range(min(3, vh.shape[0])):
                layer_entry["components"].append(
                    {
                        "component": int(comp_idx + 1),
                        "token_loadings": _top_token_loadings(vh[comp_idx], tokenizer, top_k_tokens),
                    }
                )
        direction_summary.append(layer_entry)

    summary = {
        "svd_summary_path": str(svd_summary_path),
        "divergence_jsonl_path": str(divergence_jsonl_path),
        "tokenizer_path": tokenizer_path,
        "late_layer_threshold": int(late_threshold),
        "late_top1_changed_count": len(late_changed),
        "late_top5_changed_count": len(late_top5_changed),
        "late_changed_mean_js": float(np.mean(js_values)) if js_values else 0.0,
        "late_changed_mean_tvd": float(np.mean(tvd_values)) if tvd_values else 0.0,
        "most_common_base_top1_tokens_when_changed": _counter("top1_token_base", late_changed),
        "most_common_ft_top1_tokens_when_changed": _counter("top1_token_ft", late_changed),
        "most_common_top1_transitions": top_transitions,
        "late_layer_logit_direction_tokens": direction_summary,
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Focused late-layer signal-vs-noise check")
    parser.add_argument("--svd-summary-path", required=True)
    parser.add_argument("--divergence-jsonl-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-k-tokens", type=int, default=12)
    args = parser.parse_args()

    run_focused_check(
        svd_summary_path=Path(args.svd_summary_path),
        divergence_jsonl_path=Path(args.divergence_jsonl_path),
        tokenizer_path=args.tokenizer_path,
        output_path=Path(args.output_path),
        top_k_tokens=args.top_k_tokens,
    )

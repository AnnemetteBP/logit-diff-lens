from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def summarize_divergence(
    *,
    divergence_jsonl_path: Path,
    output_path: Path | None = None,
    min_layer_fraction: float = 0.5,
    top_n: int = 25,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with divergence_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No divergence records found in {divergence_jsonl_path}")

    max_layer = max(int(rec["layer"]) for rec in records)
    min_layer = int(round(max_layer * float(min_layer_fraction)))
    filtered = [rec for rec in records if int(rec["layer"]) >= min_layer]
    if not filtered:
        filtered = records

    changed_top1 = [rec for rec in filtered if rec.get("top1_token_base") != rec.get("top1_token_ft")]
    changed_top5 = [
        rec
        for rec in filtered
        if set(rec.get("top5_tokens_base", [])) != set(rec.get("top5_tokens_ft", []))
    ]

    def _counter_from_records(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        counter = Counter(str(rec.get(key, "")) for rec in rows if str(rec.get(key, "")).strip())
        return [{"token": token, "count": int(count)} for token, count in counter.most_common(top_n)]

    transition_counter = Counter(
        (str(rec.get("top1_token_base", "")), str(rec.get("top1_token_ft", "")))
        for rec in changed_top1
    )
    top_transitions = [
        {"base_top1": a, "ft_top1": b, "count": int(count)}
        for (a, b), count in transition_counter.most_common(top_n)
    ]

    layer_js: dict[int, list[float]] = defaultdict(list)
    layer_tvd: dict[int, list[float]] = defaultdict(list)
    for rec in filtered:
        layer = int(rec["layer"])
        layer_js[layer].append(float(rec.get("js", 0.0)))
        layer_tvd[layer].append(float(rec.get("tvd", 0.0)))

    summary = {
        "divergence_jsonl_path": str(divergence_jsonl_path),
        "num_records_total": len(records),
        "num_records_filtered": len(filtered),
        "late_layer_threshold": min_layer,
        "top1_changed_count": len(changed_top1),
        "top5_changed_count": len(changed_top5),
        "most_common_base_top1_tokens_when_changed": _counter_from_records(changed_top1, "top1_token_base"),
        "most_common_ft_top1_tokens_when_changed": _counter_from_records(changed_top1, "top1_token_ft"),
        "most_common_top1_transitions": top_transitions,
        "layer_mean_js": {
            str(layer): float(sum(vals) / len(vals)) for layer, vals in sorted(layer_js.items())
        },
        "layer_mean_tvd": {
            str(layer): float(sum(vals) / len(vals)) for layer, vals in sorted(layer_tvd.items())
        },
    }

    if output_path is None:
        output_path = divergence_jsonl_path.with_name("divergence_token_shift_summary.json")
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize token-level shifts from divergence JSONL")
    parser.add_argument("--divergence-jsonl-path", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--min-layer-fraction", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    result = summarize_divergence(
        divergence_jsonl_path=Path(args.divergence_jsonl_path),
        output_path=Path(args.output_path) if args.output_path else None,
        min_layer_fraction=args.min_layer_fraction,
        top_n=args.top_n,
    )
    print(json.dumps({"output_path": str((Path(args.output_path) if args.output_path else Path(args.divergence_jsonl_path).with_name('divergence_token_shift_summary.json')))}, indent=2))

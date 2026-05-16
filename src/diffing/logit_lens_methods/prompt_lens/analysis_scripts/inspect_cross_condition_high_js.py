from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ...tokenizer_loading import load_tokenizer


def _decode_maybe_id(tokenizer: Any, token: str) -> str:
    try:
        token_id = int(token)
    except (TypeError, ValueError):
        return str(token).replace("\n", "\\n")
    decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    if decoded:
        return decoded.replace("\n", "\\n")
    raw = tokenizer.convert_ids_to_tokens([token_id])[0]
    return str(raw)


def inspect_cross_condition(
    *,
    divergence_jsonl_path: Path,
    focused_signal_check_path: Path,
    tokenizer_path: str,
    output_path: Path,
    top_n_examples: int = 20,
    top_n_tokens: int = 20,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    focused = json.loads(focused_signal_check_path.read_text(encoding="utf-8"))
    late_layer_threshold = int(focused["late_layer_threshold"])

    records: list[dict[str, Any]] = []
    with divergence_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    late_changed = [
        rec
        for rec in records
        if int(rec["layer"]) >= late_layer_threshold and rec.get("top1_token_base") != rec.get("top1_token_ft")
    ]
    late_changed_sorted = sorted(late_changed, key=lambda r: float(r.get("js", 0.0)), reverse=True)

    def _decode_counter(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        counter = Counter(str(item.get(key, "")) for item in items if str(item.get(key, "")).strip())
        out = []
        for raw_token, count in counter.most_common(top_n_tokens):
            out.append(
                {
                    "raw": raw_token,
                    "decoded": _decode_maybe_id(tokenizer, raw_token),
                    "count": int(count),
                }
            )
        return out

    transition_counter = Counter(
        (str(rec.get("top1_token_base", "")), str(rec.get("top1_token_ft", "")))
        for rec in late_changed
    )
    decoded_transitions = []
    for (raw_a, raw_b), count in transition_counter.most_common(top_n_tokens):
        decoded_transitions.append(
            {
                "base_top1_raw": raw_a,
                "base_top1_decoded": _decode_maybe_id(tokenizer, raw_a),
                "ft_top1_raw": raw_b,
                "ft_top1_decoded": _decode_maybe_id(tokenizer, raw_b),
                "count": int(count),
            }
        )

    example_rows = []
    for rec in late_changed_sorted[:top_n_examples]:
        example_rows.append(
            {
                "row_id": int(rec["row_id"]),
                "group_id": str(rec["group_id"]),
                "layer": int(rec["layer"]),
                "position": int(rec["position"]),
                "js": float(rec["js"]),
                "tvd": float(rec["tvd"]),
                "analysis_text": str(rec["analysis_text"]),
                "input_token": _decode_maybe_id(tokenizer, str(rec["input_token"])),
                "target_token": _decode_maybe_id(tokenizer, str(rec["target_token"])),
                "top1_base_raw": str(rec["top1_token_base"]),
                "top1_base_decoded": _decode_maybe_id(tokenizer, str(rec["top1_token_base"])),
                "top1_ft_raw": str(rec["top1_token_ft"]),
                "top1_ft_decoded": _decode_maybe_id(tokenizer, str(rec["top1_token_ft"])),
                "top5_base_decoded": [_decode_maybe_id(tokenizer, str(tok)) for tok in rec.get("top5_tokens_base", [])[:5]],
                "top5_ft_decoded": [_decode_maybe_id(tokenizer, str(tok)) for tok in rec.get("top5_tokens_ft", [])[:5]],
            }
        )

    summary = {
        "divergence_jsonl_path": str(divergence_jsonl_path),
        "focused_signal_check_path": str(focused_signal_check_path),
        "late_layer_threshold": late_layer_threshold,
        "most_common_base_top1_when_changed_decoded": _decode_counter(late_changed, "top1_token_base"),
        "most_common_ft_top1_when_changed_decoded": _decode_counter(late_changed, "top1_token_ft"),
        "most_common_top1_transitions_decoded": decoded_transitions,
        "highest_js_examples": example_rows,
        "late_layer_direction_tokens": focused.get("late_layer_logit_direction_tokens", []),
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode high-JS cross-condition token shifts")
    parser.add_argument("--divergence-jsonl-path", required=True)
    parser.add_argument("--focused-signal-check-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-n-examples", type=int, default=20)
    parser.add_argument("--top-n-tokens", type=int, default=20)
    args = parser.parse_args()

    inspect_cross_condition(
        divergence_jsonl_path=Path(args.divergence_jsonl_path),
        focused_signal_check_path=Path(args.focused_signal_check_path),
        tokenizer_path=args.tokenizer_path,
        output_path=Path(args.output_path),
        top_n_examples=args.top_n_examples,
        top_n_tokens=args.top_n_tokens,
    )

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def _resolve_query_text(row: dict) -> str:
    for key in ("query", "question", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"No query/question text field found in row keys: {sorted(row.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a query-only Natural Questions JSONL subset")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=500)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    ds = load_dataset("sentence-transformers/natural-questions", split=args.split)
    indices = list(range(len(ds)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    chosen = indices[: args.sample_count]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for new_id, idx in enumerate(chosen):
            row = ds[int(idx)]
            text = _resolve_query_text(dict(row))
            record = {
                "id": int(new_id),
                "group_id": f"nq_query_{new_id}",
                "variant": "query_only",
                "text": text,
                "label": 0,
                "source_dataset": "sentence-transformers/natural-questions",
                "source_split": args.split,
                "source_index": int(idx),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

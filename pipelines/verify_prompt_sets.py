from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def _sha256_lines(lines: list[str]) -> str:
    joined = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _extract_from_jsonl(path: Path, text_field: str = "text") -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts = [_normalize_text(row.get(text_field, "")) for row in rows]
    prompt_ids = [row.get("id") for row in rows]
    group_ids = [row.get("group_id") for row in rows]
    source_indices = [row.get("source_index") for row in rows]
    return {
        "path": str(path),
        "kind": "jsonl",
        "count": len(rows),
        "text_field": text_field,
        "prompt_ids": prompt_ids,
        "group_ids": group_ids,
        "source_indices": source_indices,
        "prompts": prompts,
        "prompt_sha256": _sha256_lines(prompts),
        "group_sha256": _sha256_lines([str(x) for x in group_ids]),
        "source_index_sha256": _sha256_lines([str(x) for x in source_indices]),
    }


def _extract_from_bundle(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    artifacts = payload.get("artifacts", [])
    prompts = [_normalize_text(art.get("prompt_text", "")) for art in artifacts]
    prompt_ids = [art.get("prompt_id") for art in artifacts]
    group_ids = [(art.get("metadata") or {}).get("group_id") for art in artifacts]
    row_ids = [(art.get("metadata") or {}).get("row_id") for art in artifacts]
    dataset_paths = sorted(
        {
            str((art.get("metadata") or {}).get("dataset_path"))
            for art in artifacts
            if (art.get("metadata") or {}).get("dataset_path") is not None
        }
    )
    bundle_dataset_path = payload.get("metadata", {}).get("dataset_path")
    return {
        "path": str(path),
        "kind": "bundle",
        "count": len(artifacts),
        "bundle_dataset_path": bundle_dataset_path,
        "artifact_dataset_paths": dataset_paths,
        "prompt_ids": prompt_ids,
        "row_ids": row_ids,
        "group_ids": group_ids,
        "prompts": prompts,
        "prompt_sha256": _sha256_lines(prompts),
        "group_sha256": _sha256_lines([str(x) for x in group_ids]),
        "row_id_sha256": _sha256_lines([str(x) for x in row_ids]),
    }


def summarize(path: Path, *, text_field: str = "text") -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return _extract_from_jsonl(path, text_field=text_field)
    if path.suffix == ".pt":
        return _extract_from_bundle(path)
    raise ValueError(f"Unsupported file type for {path}")


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "same_count": a["count"] == b["count"],
        "same_prompt_sha256": a.get("prompt_sha256") == b.get("prompt_sha256"),
        "same_group_sha256": a.get("group_sha256") == b.get("group_sha256"),
        "same_prompt_ids": a.get("prompt_ids") == b.get("prompt_ids"),
        "same_group_ids": a.get("group_ids") == b.get("group_ids"),
        "same_prompts": a.get("prompts") == b.get("prompts"),
    }


def main() -> None:
    _bootstrap_src()

    parser = argparse.ArgumentParser(
        description="Verify whether prompt datasets or saved prompt-bundle artifacts use the same prompt set and order."
    )
    parser.add_argument("paths", nargs="+", help="One or more .jsonl or .pt files.")
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    summaries = [summarize(Path(path), text_field=args.text_field) for path in args.paths]
    for summary in summaries:
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    if len(summaries) >= 2:
        base = summaries[0]
        for other in summaries[1:]:
            result = {
                "left": base["path"],
                "right": other["path"],
                **compare(base, other),
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

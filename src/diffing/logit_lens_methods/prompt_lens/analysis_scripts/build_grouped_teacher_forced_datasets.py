from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/media/am/AM/logit-diff-lens")


def build_grouped_dataset(src: Path, dst: Path, *, use_rendered_prompt: bool) -> None:
    rows = [
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(row)

    out_rows = []
    for gid, grp in sorted(groups.items(), key=lambda kv: int(kv[0].split("_")[-1])):
        by_role = {str(r.get("model_role")): r for r in grp}
        prompt_row = by_role["input"]
        base_row = by_role["base"]
        ft_row = by_role["finetuned"]

        prompt_text = (
            prompt_row.get("rendered_prompt")
            if use_rendered_prompt
            else prompt_row.get("prompt")
        )

        out_rows.append(
            {
                "id": int(prompt_row["source_example_id"]),
                "group_id": gid,
                "category": prompt_row.get("category"),
                "type": prompt_row.get("type"),
                "prompt_for_teacher_forcing": prompt_text,
                "system_prompt": prompt_row.get("system_prompt"),
                "base_response": base_row.get("response_only"),
                "finetuned_response": ft_row.get("response_only"),
                "label": prompt_row.get("harmfulness_label_ft"),
            }
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    specs = [
        (
            ROOT / "datasets/logit_prisms_data_first9.jsonl",
            ROOT / "datasets/logit_prisms_teacher_forced_grouped_first9.jsonl",
            True,
        ),
        (
            ROOT / "datasets/logit_prisms_data.jsonl",
            ROOT / "datasets/logit_prisms_teacher_forced_grouped.jsonl",
            True,
        ),
        (
            ROOT / "datasets/logit_prisms_data_first9.jsonl",
            ROOT / "datasets/logit_prisms_teacher_forced_grouped_clean_first9.jsonl",
            False,
        ),
        (
            ROOT / "datasets/logit_prisms_data.jsonl",
            ROOT / "datasets/logit_prisms_teacher_forced_grouped_clean.jsonl",
            False,
        ),
    ]
    for src, dst, use_rendered_prompt in specs:
        build_grouped_dataset(src, dst, use_rendered_prompt=use_rendered_prompt)
        print(f"wrote {dst} (use_rendered_prompt={use_rendered_prompt})")


if __name__ == "__main__":
    main()

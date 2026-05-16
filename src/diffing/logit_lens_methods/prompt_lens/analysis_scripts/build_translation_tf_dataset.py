from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/media/am/AM/logit-diff-lens")
SOURCE_PATH = ROOT / "datasets/base_data/translation_tasks.py"
OUTPUT_JSONL = ROOT / "datasets/base_data/translation_tasks_tf.jsonl"
OUTPUT_SUMMARY = ROOT / "datasets/base_data/translation_tasks_tf.summary.json"


SCRIPT_BY_LANGUAGE = {
    "english": "latin",
    "german": "latin",
    "spanish": "latin",
    "french": "latin",
    "russian": "cyrillic",
    "japanese": "mixed_cjk_kana",
    "chinese": "cjk",
    "arabic": "arabic",
    "hindi": "devanagari",
}


@dataclass(frozen=True)
class DatasetRow:
    id: int
    group_id: str
    task_id: str
    language: str
    prompt_variant: str
    text: str
    expected_language: str
    expected_script: str
    script_family: str
    language_group: str
    english_reference: str
    source_text: str
    literal_english: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_id": self.group_id,
            "task_id": self.task_id,
            "variant": self.prompt_variant,
            "prompt_variant": self.prompt_variant,
            "language": self.language,
            "text": self.text,
            "analysis_text": self.text,
            "prompt_clean": self.text,
            "expected_language": self.expected_language,
            "expected_script": self.expected_script,
            "script_family": self.script_family,
            "language_group": self.language_group,
            "english_reference": self.english_reference,
            "source_text": self.source_text,
            "literal_english": self.literal_english,
            "source_kind": "translation_prompt",
            "model_role": "prompt_only",
            "label": 0,
        }


def _load_translation_tasks(path: Path) -> list[dict[str, Any]]:
    namespace: dict[str, Any] = {}
    exec(path.read_text(encoding="utf-8"), namespace)
    tasks = namespace.get("translation_tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Expected translation_tasks list in {path}")
    return tasks


def _make_rows(tasks: list[dict[str, Any]]) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    next_id = 0
    for task in tasks:
        task_id = str(task["id"])
        english_reference = str(task["english_reference"]).strip()
        translations = dict(task["translations"])
        rows.append(
            DatasetRow(
                id=next_id,
                group_id=f"{task_id}::english_reference",
                task_id=task_id,
                language="english",
                prompt_variant="english_reference",
                text=english_reference,
                expected_language="english",
                expected_script=SCRIPT_BY_LANGUAGE["english"],
                script_family=SCRIPT_BY_LANGUAGE["english"],
                language_group="english_reference",
                english_reference=english_reference,
                source_text="",
                literal_english="",
            )
        )
        next_id += 1

        for language_key, payload in translations.items():
            language = str(language_key).rsplit("_", 1)[0]
            if language not in SCRIPT_BY_LANGUAGE:
                raise KeyError(f"Unsupported language key {language_key!r} -> {language!r}")
            source_text = str(payload["source"]).strip()
            literal_english = str(payload["literal_english"]).strip()
            expected_script = SCRIPT_BY_LANGUAGE[language]
            group_id = f"{task_id}::{language_key}"

            rows.append(
                DatasetRow(
                    id=next_id,
                    group_id=group_id,
                    task_id=task_id,
                    language=language,
                    prompt_variant="source_non_english",
                    text=source_text,
                    expected_language=language,
                    expected_script=expected_script,
                    script_family=expected_script,
                    language_group="non_english_source",
                    english_reference=english_reference,
                    source_text=source_text,
                    literal_english=literal_english,
                )
            )
            next_id += 1
    return rows


def build_translation_tf_dataset(
    *,
    source_path: Path = SOURCE_PATH,
    output_jsonl: Path = OUTPUT_JSONL,
    output_summary: Path = OUTPUT_SUMMARY,
) -> dict[str, Any]:
    tasks = _load_translation_tasks(source_path)
    rows = _make_rows(tasks)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "\n".join(json.dumps(row.to_json(), ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    by_language: dict[str, int] = {}
    by_variant: dict[str, int] = {}
    by_script: dict[str, int] = {}
    for row in rows:
        by_language[row.language] = by_language.get(row.language, 0) + 1
        by_variant[row.prompt_variant] = by_variant.get(row.prompt_variant, 0) + 1
        by_script[row.script_family] = by_script.get(row.script_family, 0) + 1

    summary = {
        "source_path": str(source_path),
        "output_jsonl": str(output_jsonl),
        "num_tasks": len(tasks),
        "num_rows": len(rows),
        "english_reference_rows_per_task": 1,
        "non_english_rows_per_task": len(rows) - len(tasks),
        "variants": sorted(by_variant),
        "counts_by_variant": by_variant,
        "counts_by_language": by_language,
        "counts_by_script_family": by_script,
        "languages": sorted(by_language),
    }
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = build_translation_tf_dataset()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build paper-ready NQ-500 depth tables from aggregated block summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/media/am/AM/logit-diff-lens")
MANIFEST_PATH = ROOT / "ucloud_logitdiff" / "derived" / "manifest.json"
OUT_PATH = ROOT / "tmp" / "paper_tex" / "nq500_layer_summaries.tex"


FAMILY_LAYOUT = {
    "qwen": {
        "caption": "Qwen NQ-500 depth summary for paper-selected prompt-lens metrics. Reported depth regions denote means over true layer partitions rather than single selected layers.",
        "label": "tab:qwen:nq500:paper-depth",
        "order": ["risky", "medical", "sports"],
        "display": {
            "risky": "Risky Financial Advice",
            "medical": "Bad Medical Advice",
            "sports": "Extreme Sports",
        },
    },
    "llama": {
        "caption": "Quantized LLaMA NQ-500 depth summary for paper-selected prompt-lens metrics. Reported depth regions denote means over true layer partitions rather than single selected layers.",
        "label": "tab:quantized:llama:nq500:paper-depth",
        "order": ["hf1bit", "llama_4bit", "llama_8bit"],
        "display": {
            "hf1bit": "HF1BitLLM 1.58-bit",
            "llama_4bit": "BnB 4-bit",
            "llama_8bit": "BnB 8-bit",
        },
    },
    "pythia": {
        "caption": "Pythia NQ-500 depth summary for paper-selected prompt-lens metrics. Reported depth regions denote means over true layer partitions rather than single selected layers.",
        "label": "tab:pythia:nq500:paper-depth",
        "order": [
            "160m_1k_first",
            "160m_71k_mid",
            "410m_1k_first",
            "410m_71k_mid",
            "1p4b_1k_first",
            "1p4b_71k_mid",
            "2p8b_1k_first",
            "2p8b_71k_mid",
            "6p9b_1k_first",
            "6p9b_71k_mid",
            "12b_1k_first",
            "12b_71k_mid",
        ],
        "display": {
            "160m_1k_first": "160M early (1k)",
            "160m_71k_mid": "160M mid (71k)",
            "410m_1k_first": "410M early (1k)",
            "410m_71k_mid": "410M mid (71k)",
            "1p4b_1k_first": "1.4B early (1k)",
            "1p4b_71k_mid": "1.4B mid (71k)",
            "2p8b_1k_first": "2.8B early (1k)",
            "2p8b_71k_mid": "2.8B mid (71k)",
            "6p9b_1k_first": "6.9B early (1k)",
            "6p9b_71k_mid": "6.9B mid (71k)",
            "12b_1k_first": "12B early (1k)",
            "12b_71k_mid": "12B mid (71k)",
        },
    },
}


TABLE_ROWS = [
    ("hidden", None, "hidden_cosine", "Hidden-States", "Cosine Similarity"),
    ("hidden", None, "hidden_l2", "Hidden-States", "L2 Distance"),
    ("modes", "raw", "jaccard_top5", "Raw LogitDiff Lens", "Jaccard@5"),
    ("modes", "raw", "js", "Raw LogitDiff Lens", "Jensen-Shannon Divergence (JSD)"),
    ("modes", "model_norm", "jaccard_top5", "ModelNorm LogitDiff Lens", "Jaccard@5"),
    ("modes", "model_norm", "js", "ModelNorm LogitDiff Lens", "Jensen-Shannon Divergence (JSD)"),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_value(value: float) -> str:
    return f"{value:.3g}" if abs(value) < 1e-3 and value != 0.0 else f"{value:.3f}"


def _overall_from_mode_specific(mode_specific: dict[str, Any], category: str, mode: str | None, metric: str) -> float:
    if category == "hidden":
        curve = mode_specific["hidden"][metric]["layerwise_mean"]
    else:
        curve = mode_specific["modes"][mode][metric]["layerwise_mean"]
    return float(sum(float(v) for v in curve) / len(curve))


def _block_values(block_summary: dict[str, Any], category: str, mode: str | None, metric: str) -> list[float]:
    if category == "hidden":
        raise KeyError("Hidden block rows are not stored in block_summary")
    metric_map = block_summary["modes"][mode][metric]
    return [float(metric_map[name]["mean"]) for name in ("first", "early", "mid", "late", "last")]


def _hidden_block_values(prompt_block_path: Path, metric: str) -> list[float]:
    acc: dict[str, list[float]] = {name: [] for name in ("first", "early", "mid", "late", "last")}
    with prompt_block_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("category") != "hidden" or row.get("metric") != metric:
                continue
            acc[str(row["block"])].append(float(row["value_mean"]))
    return [
        (sum(acc[name]) / len(acc[name])) if acc[name] else 0.0
        for name in ("first", "early", "mid", "late", "last")
    ]


def _build_table(case_entries: list[dict[str, Any]], *, caption: str, label: str, display_map: dict[str, str]) -> str:
    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l l l | r r r r r r}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Lens} & \textbf{Metric} & \textbf{First} & \textbf{Early} & \textbf{Mid} & \textbf{Late} & \textbf{Last} & \textbf{AUC} \\",
        r"\midrule",
    ]
    for entry in case_entries:
        case_id = entry["case_id"]
        block_summary = _load_json(Path(entry["block_summary_json"]))
        mode_specific = _load_json(Path(entry["summary_json"]))
        prompt_block_path = Path(entry["prompt_block_summary_jsonl"])
        for category, mode, metric, lens_label, metric_label in TABLE_ROWS:
            if category == "hidden":
                values = _hidden_block_values(prompt_block_path, metric)
            else:
                values = _block_values(block_summary, category, mode, metric)
            overall = _overall_from_mode_specific(mode_specific, category, mode, metric)
            rendered = " & ".join(_format_value(v) for v in [*values, overall])
            lines.append(
                f"{display_map[case_id]} & {lens_label} & {metric_label} & {rendered} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    manifest = _load_json(MANIFEST_PATH)
    cases = manifest["cases"]
    tables: list[str] = [
        "% Auto-generated compact NQ-500 prompt-lens depth tables using true block aggregates.",
        "% First/Early/Mid/Late/Last are means over layer partitions from ucloud_logitdiff/derived.",
        "",
    ]
    for family, config in FAMILY_LAYOUT.items():
        entries_by_id = {entry["case_id"]: entry for entry in cases if entry["family"] == family}
        ordered_entries = [entries_by_id[case_id] for case_id in config["order"]]
        tables.append(
            _build_table(
                ordered_entries,
                caption=config["caption"],
                label=config["label"],
                display_map=config["display"],
            )
        )
        tables.append("")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(tables), encoding="utf-8")
    print(json.dumps({"output_path": str(OUT_PATH)}, indent=2))


if __name__ == "__main__":
    main()

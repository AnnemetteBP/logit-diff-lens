#!/usr/bin/env python3
"""Build a compact concluding correlation table for the paper."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path("/media/am/AM/logit-diff-lens")
SRC = ROOT / "tmp" / "paper_tex" / "block_prompt_correlations.json"
OUT = ROOT / "tmp" / "paper_tex" / "correlation_results_summary.tex"

FAMILY_ORDER = ["qwen", "llama", "pythia"]
FAMILY_LABELS = {"qwen": "Qwen", "llama": "LLaMA", "pythia": "Pythia"}
FAMILY_COLORS = {"qwen": "qwen", "llama": "llama", "pythia": "pythia"}

CASE_ORDER = {
    "qwen": ["risky", "medical", "sports"],
    "llama": ["hf1bit", "llama_4bit", "llama_8bit"],
    "pythia": ["410m_1k_first", "2p8b_1k_first", "12b_1k_first"],
}

CASE_LABELS = {
    "risky": "Financial",
    "medical": "Medical",
    "sports": "Sports",
    "hf1bit": "1.58-bit",
    "llama_4bit": "4-bit",
    "llama_8bit": "8-bit",
    "410m_1k_first": "410M 1k",
    "2p8b_1k_first": "2.8B 1k",
    "12b_1k_first": "12B 1k",
}

def fmt_num(value: float) -> str:
    return f"{value:.2f}" if math.isfinite(value) else "--"


def fmt_layer_pair(row: dict) -> str:
    return f"{row['source_block'].capitalize()}/{row['target_block'].capitalize()}"


def select_best(rows: list[dict], case_id: str, comparison_name: str) -> dict:
    candidates = [
        row
        for row in rows
        if row["case_id"] == case_id
        and row["comparison_name"] == comparison_name
        and math.isfinite(row["pearson_r"])
    ]
    return max(candidates, key=lambda row: abs(row["pearson_r"]))


def fmt_cell(row: dict) -> str:
    return f"{fmt_num(row['pearson_r'])} / {fmt_num(row['spearman_r'])}"


def main() -> None:
    rows = json.loads(SRC.read_text())

    lines = [
        "% Auto-generated concluding correlation table.",
        r"\begin{table*}[!t]",
        r"\centering",
        r"\small",
        r"\definecolor{qwen}{HTML}{B94B5F}",
        r"\definecolor{llama}{HTML}{4F97B3}",
        r"\definecolor{pythia}{HTML}{EFB6AD}",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{ll!{\vrule width 0.8pt}cc cc!{\vrule width 0.8pt}cc cc}",
        r"\toprule",
        r"& & \multicolumn{4}{c!{\vrule width 0.8pt}}{\textbf{Within-Lens: JSD vs J@5}} & \multicolumn{4}{c}{\textbf{Cross-Lens: Raw vs MN}} \\",
        r"\cmidrule(lr){3-6}",
        r"\cmidrule(l){7-10}",
        r"\textbf{Family} & \textbf{Case} & \textbf{R L/L} & \textbf{R $r_{\mathrm{P}}/\rho_{\mathrm{S}}$} & \textbf{MN L/L} & \textbf{MN $r_{\mathrm{P}}/\rho_{\mathrm{S}}$} & \textbf{JSD L/L} & \textbf{JSD $r_{\mathrm{P}}/\rho_{\mathrm{S}}$} & \textbf{J@5 L/L} & \textbf{J@5 $r_{\mathrm{P}}/\rho_{\mathrm{S}}$} \\",
        r"\midrule",
    ]

    family_count = 0
    for family in FAMILY_ORDER:
        if family_count:
            lines.extend(
                [
                    r"\addlinespace[0.35em]",
                    r"\cdashline{1-10}[0.4pt/1.5pt]",
                    r"\addlinespace[0.35em]",
                ]
            )
        family_count += 1
        family_rows = [row for row in rows if row["family"] == family]
        family_label = rf"\textcolor{{{FAMILY_COLORS[family]}}}{{\textbf{{{FAMILY_LABELS[family]}}}}}"
        for idx, case_id in enumerate(CASE_ORDER[family]):
            raw_row = select_best(family_rows, case_id, "raw:js_vs_jaccard:vs_last")
            mn_row = select_best(family_rows, case_id, "model_norm:js_vs_jaccard:vs_last")
            js_row = select_best(family_rows, case_id, "js:raw_vs_model_norm:vs_last")
            j5_row = select_best(family_rows, case_id, "jaccard_top5:raw_vs_model_norm:vs_last")
            lines.append(
                f"{family_label if idx == 0 else ''} & {CASE_LABELS[case_id]} & "
                f"{fmt_layer_pair(raw_row)} & {fmt_cell(raw_row)} & "
                f"{fmt_layer_pair(mn_row)} & {fmt_cell(mn_row)} & "
                f"{fmt_layer_pair(js_row)} & {fmt_cell(js_row)} & "
                f"{fmt_layer_pair(j5_row)} & {fmt_cell(j5_row)} \\\\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Concluding prompt-level correlation summary on NQ-500. The left group reports within-lens JSD-vs.-J@$5$ correlations for Raw (R) and ModelNorm (MN) separately. The right group reports the concluding cross-lens Raw-vs.-ModelNorm agreement for JSD and J@$5$. L/L reports the source and target layer blocks for the strongest absolute correlation, and correlation cells report Pearson $r_{\mathrm{P}}$ and Spearman $\rho_{\mathrm{S}}$. All rows use $n=500$ prompts, so prompts rather than expanded token positions define the effective sampling unit.}",
            r"\label{tab:correlation-results-summary}",
            r"\end{table*}",
        ]
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(OUT))


if __name__ == "__main__":
    main()

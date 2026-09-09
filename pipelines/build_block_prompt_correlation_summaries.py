#!/usr/bin/env python3
"""Build prompt-level block correlation summaries for all local derived cases.

This analysis uses prompt-level block means as the sampling unit rather than
fully expanded token-position observations. That keeps the correlation sample
size at the prompt level and avoids treating within-prompt continuation tokens
as independent random variables.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from scipy import stats


ROOT = Path("/media/am/AM/logit-diff-lens")
MANIFEST_PATH = ROOT / "ucloud_logitdiff" / "derived" / "manifest.json"
OUT_DIR = ROOT / "tmp" / "paper_tex"
BLOCKS = ["first", "early", "mid", "late", "last"]
MODES = ["raw", "model_norm"]
METRIC_LABELS = {
    "js": "JSD",
    "jaccard_top5": "J@5",
    "ft_top5_next_token_accuracy": "Top-5 accuracy",
}


@dataclass
class CorrRow:
    family: str
    case_id: str
    label: str
    comparison_group: str
    comparison_name: str
    source_mode: str
    source_metric: str
    source_block: str
    target_mode: str
    target_metric: str
    target_block: str
    n: int
    pearson_r: float
    pearson_p: float
    pearson_ci_low: float
    pearson_ci_high: float
    pearson_q: float
    spearman_r: float
    spearman_p: float
    spearman_ci_low: float
    spearman_ci_high: float
    spearman_q: float


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_prompt_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _collect_case_vectors(prompt_block_path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    keep_metrics = set(METRIC_LABELS)
    for row in _iter_prompt_rows(prompt_block_path):
        if row.get("category") != "modes":
            continue
        mode = row.get("mode")
        metric = row.get("metric")
        block = row.get("block")
        if mode not in MODES or metric not in keep_metrics or block not in BLOCKS:
            continue
        key = (mode, metric, block)
        out.setdefault(key, {})[str(row["group_id"])] = float(row["value_mean"])
    return out


def _fisher_ci(corr: float, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    if not math.isfinite(corr) or n < 4 or abs(corr) >= 1.0:
        return float("nan"), float("nan")
    z = math.atanh(max(min(corr, 0.999999), -0.999999))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    return math.tanh(z - z_crit * se), math.tanh(z + z_crit * se)


def _fdr_bh(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    q = [float("nan")] * m
    running = 1.0
    for rank, (idx, p_value) in reversed(list(enumerate(indexed, start=1))):
        adj = min(running, (p_value * m) / rank)
        running = adj
        q[idx] = adj
    return q


def _corr(values_a: dict[str, float], values_b: dict[str, float]) -> tuple[int, float, float, float, float, float, float, float, float]:
    shared = sorted(set(values_a) & set(values_b))
    n = len(shared)
    if n < 4:
        return (
            n,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    a = [values_a[k] for k in shared]
    b = [values_b[k] for k in shared]
    pearson = stats.pearsonr(a, b)
    spearman = stats.spearmanr(a, b)
    p_lo, p_hi = _fisher_ci(float(pearson.statistic), n)
    s_lo, s_hi = _fisher_ci(float(spearman.statistic), n)
    return (
        n,
        float(pearson.statistic),
        float(pearson.pvalue),
        p_lo,
        p_hi,
        float(spearman.statistic),
        float(spearman.pvalue),
        s_lo,
        s_hi,
    )


def _comparison_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []

    # Same-lens, metric-vs-metric.
    metric_pairs = [
        ("js", "jaccard_top5", "js_vs_jaccard"),
        ("js", "ft_top5_next_token_accuracy", "js_vs_accuracy"),
        ("jaccard_top5", "ft_top5_next_token_accuracy", "jaccard_vs_accuracy"),
    ]
    for mode in MODES:
        for metric_a, metric_b, name in metric_pairs:
            specs.append(
                {
                    "comparison_group": "within_lens_same_block",
                    "comparison_name": f"{mode}:{name}:same_block",
                    "source_mode": mode,
                    "source_metric": metric_a,
                    "target_mode": mode,
                    "target_metric": metric_b,
                    "target_block": "same",
                }
            )
            specs.append(
                {
                    "comparison_group": "within_lens_vs_last",
                    "comparison_name": f"{mode}:{name}:vs_last",
                    "source_mode": mode,
                    "source_metric": metric_a,
                    "target_mode": mode,
                    "target_metric": metric_b,
                    "target_block": "last",
                }
            )

    # Cross-lens, same metric.
    for metric in METRIC_LABELS:
        specs.append(
            {
                "comparison_group": "cross_lens_same_block",
                "comparison_name": f"{metric}:raw_vs_model_norm:same_block",
                "source_mode": "raw",
                "source_metric": metric,
                "target_mode": "model_norm",
                "target_metric": metric,
                "target_block": "same",
            }
        )
        specs.append(
            {
                "comparison_group": "cross_lens_vs_last",
                "comparison_name": f"{metric}:raw_vs_model_norm:vs_last",
                "source_mode": "raw",
                "source_metric": metric,
                "target_mode": "model_norm",
                "target_metric": metric,
                "target_block": "last",
            }
        )
    return specs


def _rows_for_case(case: dict, vectors: dict[tuple[str, str, str], dict[str, float]]) -> list[CorrRow]:
    rows: list[CorrRow] = []
    for spec in _comparison_specs():
        source_blocks = BLOCKS[:-1] if spec["target_block"] == "last" else BLOCKS
        for block in source_blocks:
            target_block = block if spec["target_block"] == "same" else "last"
            key_a = (spec["source_mode"], spec["source_metric"], block)
            key_b = (spec["target_mode"], spec["target_metric"], target_block)
            n, pr, pp, p_lo, p_hi, sr, sp, s_lo, s_hi = _corr(vectors[key_a], vectors[key_b])
            rows.append(
                CorrRow(
                    family=case["family"],
                    case_id=case["case_id"],
                    label=case["label"],
                    comparison_group=spec["comparison_group"],
                    comparison_name=spec["comparison_name"],
                    source_mode=spec["source_mode"],
                    source_metric=spec["source_metric"],
                    source_block=block,
                    target_mode=spec["target_mode"],
                    target_metric=spec["target_metric"],
                    target_block=target_block,
                    n=n,
                    pearson_r=pr,
                    pearson_p=pp,
                    pearson_ci_low=p_lo,
                    pearson_ci_high=p_hi,
                    pearson_q=float("nan"),
                    spearman_r=sr,
                    spearman_p=sp,
                    spearman_ci_low=s_lo,
                    spearman_ci_high=s_hi,
                    spearman_q=float("nan"),
                )
            )
    return rows


def _apply_q_values(rows: list[CorrRow]) -> None:
    pearson_ps = [row.pearson_p for row in rows if math.isfinite(row.pearson_p)]
    spearman_ps = [row.spearman_p for row in rows if math.isfinite(row.spearman_p)]
    pearson_qs = iter(_fdr_bh(pearson_ps))
    spearman_qs = iter(_fdr_bh(spearman_ps))
    for row in rows:
        if math.isfinite(row.pearson_p):
            row.pearson_q = next(pearson_qs)
        if math.isfinite(row.spearman_p):
            row.spearman_q = next(spearman_qs)


def _format_block(block: str) -> str:
    return block.capitalize()


def _format_comp(name: str) -> str:
    return (
        name.replace("model_norm", "MN")
        .replace("raw", "R")
        .replace("jaccard", "J@5")
        .replace("accuracy", "Acc")
        .replace("js", "JSD")
        .replace(":same_block", "")
        .replace(":vs_last", " -> Last")
        .replace(":", " / ")
        .replace("_", " ")
    )


def _write_tex(rows: list[CorrRow], path: Path) -> None:
    focus_comparisons = {
        "raw:js_vs_jaccard:same_block",
        "model_norm:js_vs_jaccard:same_block",
        "js:raw_vs_model_norm:same_block",
        "jaccard_top5:raw_vs_model_norm:same_block",
        "ft_top5_next_token_accuracy:raw_vs_model_norm:same_block",
    }
    lines = [
        "% Auto-generated prompt-level block correlation summary.",
        "% Correlations are computed over prompt-level block means rather than expanded token positions.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l l l r r r}",
        r"\toprule",
        r"\textbf{Family} & \textbf{Case} & \textbf{Comparison / Block} & \textbf{$n$} & \textbf{Pearson $r$} & \textbf{Spearman $\rho$} \\",
        r"\midrule",
    ]
    grouped: dict[tuple[str, str, str], CorrRow] = {
        (row.family, row.case_id, f"{row.comparison_name}|{row.source_block}|{row.target_block}"): row for row in rows
    }
    family_order = ["qwen", "llama", "pythia"]
    cases_by_family: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        cases_by_family.setdefault(row.family, [])
        key = (row.case_id, row.label)
        if key not in cases_by_family[row.family]:
            cases_by_family[row.family].append(key)
    for family in family_order:
        for case_id, label in sorted(cases_by_family.get(family, [])):
            for comparison in focus_comparisons:
                for block in BLOCKS:
                    key = (family, case_id, f"{comparison}|{block}|{block}")
                    row = grouped.get(key)
                    if row is None:
                        continue
                    lines.append(
                        f"{family} & {label} & {_format_comp(comparison)} / {_format_block(block)} & {row.n} & {row.pearson_r:.3f} & {row.spearman_r:.3f} \\\\"
                    )
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Prompt-level block correlations over First/Early/Mid/Late/Last aggregates. Each correlation is computed over prompt-level block means, so prompts rather than expanded token positions define the sampling unit.}",
            r"\label{tab:block-prompt-correlations}",
            r"\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manifest = _load_json(MANIFEST_PATH)
    rows: list[CorrRow] = []
    for case in manifest["cases"]:
        prompt_block_path = Path(case["prompt_block_summary_jsonl"])
        vectors = _collect_case_vectors(prompt_block_path)
        rows.extend(_rows_for_case(case, vectors))

    _apply_q_values(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "block_prompt_correlations.json"
    tex_path = OUT_DIR / "block_prompt_correlations.tex"
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    _write_tex(rows, tex_path)
    print(json.dumps({"json": str(json_path), "tex": str(tex_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

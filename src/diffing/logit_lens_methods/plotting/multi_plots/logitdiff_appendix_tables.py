from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_layer_indices(num_layers: int) -> dict[str, int]:
    last = num_layers - 1
    return {
        "first": 0,
        "early": round(last * 0.25),
        "mid": round(last * 0.50),
        "late": round(last * 0.75),
        "last": last,
    }


def _wrap_table(tabular: str, *, caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\resizebox{\textwidth}{!}{%",
            tabular,
            r"}",
            r"\end{table*}",
        ]
    )


def _tabular_from_rows(header_first_col: str, rows: list[tuple[str, list[float]]]) -> str:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        rf"{header_first_col} & First & Early & Mid & Late & Last & Overall \\",
        r"\midrule",
    ]
    for label, vals in rows:
        lines.append(
            f"{label} & "
            f"{vals[0]:.3f} & {vals[1]:.3f} & {vals[2]:.3f} & {vals[3]:.3f} & {vals[4]:.3f} & {vals[5]:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _summarize_curve(layerwise_mean: list[float]) -> list[float]:
    num_layers = len(layerwise_mean)
    picks = _pick_layer_indices(num_layers)
    vals = [float(layerwise_mean[picks[k]]) for k in ["first", "early", "mid", "late", "last"]]
    overall = float(sum(layerwise_mean) / len(layerwise_mean))
    return vals + [overall]


def _save_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def save_generation_tables(
    generation_summary_json: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    payload = _load_json(generation_summary_json)
    summaries = payload["summaries"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created: dict[str, str] = {}

    for key, display, label in [
        ("top1_mean", "Top-1 generation Jaccard overlap across templates", "tab:gen_top1_jaccard"),
        ("top5_mean", "Top-5 generation Jaccard overlap across templates", "tab:gen_top5_jaccard"),
        ("top10_mean", "Top-10 generation Jaccard overlap across templates", "tab:gen_top10_jaccard"),
    ]:
        rows = []
        for summary in summaries:
            curve = [float(row[key]) for row in summary["layers"]]
            rows.append((summary["template_label"], _summarize_curve(curve)))
        tabular = _tabular_from_rows("Template", rows)
        wrapped = _wrap_table(tabular, caption=display, label=label)
        stem = key.replace("_mean", "")
        path = out / f"generation_{stem}_summary_table.tex"
        _save_text(path, wrapped)
        created[stem] = str(path)
    return created


def save_tf_jaccard_tables(
    tf_summary_json: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    payload = _load_json(tf_summary_json)
    modes = payload["modes"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created: dict[str, str] = {}
    mode_rows = [("Raw LogitDiff Lens", "raw"), ("Model Norm LogitDiff Lens", "model_norm")]

    for metric_key, display, label, stem in [
        ("jaccard_top1", "Teacher-forcing Jaccard@1 across lens types", "tab:tf_jaccard_top1", "top1"),
        ("jaccard_top5", "Teacher-forcing Jaccard@5 across lens types", "tab:tf_jaccard_top5", "top5"),
        ("jaccard_top10", "Teacher-forcing Jaccard@10 across lens types", "tab:tf_jaccard_top10", "top10"),
    ]:
        rows = []
        for row_label, mode_key in mode_rows:
            rows.append((row_label, _summarize_curve(modes[mode_key][metric_key]["layerwise_mean"])))
        tabular = _tabular_from_rows("Lens", rows)
        wrapped = _wrap_table(tabular, caption=display, label=label)
        path = out / f"tf_jaccard_{stem}_summary_table.tex"
        _save_text(path, wrapped)
        created[stem] = str(path)
    return created


def save_tf_hidden_table(
    tf_summary_json: str | Path,
    *,
    output_dir: str | Path,
) -> str:
    payload = _load_json(tf_summary_json)
    # Hidden metrics are mode-independent here; use raw copy.
    raw = payload["modes"]["raw"]
    rows = [
        ("Cosine Similarity", _summarize_curve(raw["hidden_cosine"]["layerwise_mean"])),
        ("L2 Distance", _summarize_curve(raw["hidden_l2"]["layerwise_mean"])),
    ]
    tabular = _tabular_from_rows("Metric", rows)
    wrapped = _wrap_table(
        tabular,
        caption="Teacher-forcing hidden-state comparison across layers",
        label="tab:tf_hidden_metrics",
    )
    path = Path(output_dir) / "tf_hidden_state_summary_table.tex"
    _save_text(path, wrapped)
    return str(path)


def save_tf_divergence_tables(
    tf_summary_json: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    payload = _load_json(tf_summary_json)
    modes = payload["modes"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    created: dict[str, str] = {}
    mode_rows = [("Raw LogitDiff Lens", "raw"), ("Model Norm LogitDiff Lens", "model_norm")]

    for metric_key, display, label, stem in [
        ("tvd", "Teacher-forcing Total Variation Distance across lens types", "tab:tf_tvd", "tvd"),
        ("js", "Teacher-forcing Jensen-Shannon Divergence across lens types", "tab:tf_js", "js"),
    ]:
        rows = []
        for row_label, mode_key in mode_rows:
            rows.append((row_label, _summarize_curve(modes[mode_key][metric_key]["layerwise_mean"])))
        tabular = _tabular_from_rows("Lens", rows)
        wrapped = _wrap_table(tabular, caption=display, label=label)
        path = out / f"tf_{stem}_summary_table.tex"
        _save_text(path, wrapped)
        created[stem] = str(path)
    return created

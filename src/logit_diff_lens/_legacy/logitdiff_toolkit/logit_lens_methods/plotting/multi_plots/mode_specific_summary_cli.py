from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _layer_ticks(n: int) -> tuple[list[int], list[str]]:
    positions = list(range(n))
    labels = [str(i + 1) for i in positions]
    if labels:
        labels[-1] = "Last"
    return positions, labels


def _picked_values(curve: list[float]) -> list[float]:
    n = len(curve)
    last = n - 1
    idxs = [0, round(last * 0.25), round(last * 0.50), round(last * 0.75), last]
    vals = [float(curve[i]) for i in idxs]
    vals.append(float(sum(curve) / len(curve)))
    return vals


def _style_axis(ax: plt.Axes, x: list[int], tick_positions: list[int], tick_labels: list[str]) -> None:
    ax.grid(True, alpha=0.25)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=32)
    ax.set_xlim(x[0], x[-1])


def _plot_jaccard_panel(ax: plt.Axes, mode_metrics: dict[str, Any], title: str) -> None:
    x = list(range(len(mode_metrics["jaccard_top10"]["layerwise_mean"])))
    tick_positions, tick_labels = _layer_ticks(len(x))
    colors = {"jaccard_top1": "#d1495b", "jaccard_top5": "#2a9d8f", "jaccard_top10": "#1d3557"}
    markers = {"jaccard_top1": "o", "jaccard_top5": "s", "jaccard_top10": "^"}
    labels = {"jaccard_top1": "Jaccard@1", "jaccard_top5": "Jaccard@5", "jaccard_top10": "Jaccard@10"}
    for key in ("jaccard_top1", "jaccard_top5", "jaccard_top10"):
        ax.plot(x, mode_metrics[key]["layerwise_mean"], color=colors[key], marker=markers[key], linewidth=2.5, markersize=6.6, label=labels[key])
    ax.set_title(title, fontsize=14.5, fontweight="semibold")
    ax.set_xlabel("Layer", fontsize=12.5, fontweight="semibold")
    ax.set_ylabel("Jaccard overlap", fontsize=12.5, fontweight="semibold")
    ax.set_ylim(0.0, 1.0)
    _style_axis(ax, x, tick_positions, tick_labels)
    ax.legend(frameon=False, fontsize=12, loc="upper left", prop={"weight": "semibold", "size": 12})


def _plot_divergence_panel(ax: plt.Axes, mode_metrics: dict[str, Any], title: str) -> None:
    x = list(range(len(mode_metrics["tvd"]["layerwise_mean"])))
    tick_positions, tick_labels = _layer_ticks(len(x))
    ax.plot(x, mode_metrics["tvd"]["layerwise_mean"], color="#7c4dff", marker="s", linewidth=2.5, markersize=6.4, label="TVD")
    ax.plot(x, mode_metrics["js"]["layerwise_mean"], color="#f4a261", marker="^", linewidth=2.5, markersize=6.4, label="JS")
    ax.set_title(title, fontsize=14.5, fontweight="semibold")
    ax.set_xlabel("Layer", fontsize=12.5, fontweight="semibold")
    ax.set_ylabel("Divergence", fontsize=12.5, fontweight="semibold")
    _style_axis(ax, x, tick_positions, tick_labels)
    ax.legend(frameon=False, fontsize=11.8, loc="upper left", prop={"weight": "semibold", "size": 11.8})


def save_mode_specific_figures(
    summary_json: str | Path,
    *,
    output_dir: str | Path,
    output_stem: str,
    title_prefix: str,
    table_label: str,
) -> dict[str, str]:
    summary = _load_json(summary_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden = summary["hidden"]
    modes = summary["modes"]

    hidden_png = output_dir / f"{output_stem}_hidden_summary.png"
    hidden_pdf = output_dir / f"{output_stem}_hidden_summary.pdf"
    logit_png = output_dir / f"{output_stem}_mode_specific_logit_summary.png"
    logit_pdf = output_dir / f"{output_stem}_mode_specific_logit_summary.pdf"
    table_tex = output_dir / f"{output_stem}_mode_specific_summary_table.tex"

    x = list(range(len(hidden["hidden_cosine"]["layerwise_mean"])))
    tick_positions, tick_labels = _layer_ticks(len(x))

    fig, axes = plt.subplots(1, 3, figsize=(18.8, 5.7), sharex=True, gridspec_kw={"width_ratios": [1.16, 1.02, 1.02]})
    ax = axes[0]
    ax.plot(x, hidden["hidden_cosine"]["layerwise_mean"], color="#457b9d", marker="o", linewidth=2.6, markersize=6.8, label="Cosine similarity")
    ax2 = ax.twinx()
    ax2.plot(x, hidden["hidden_l2"]["layerwise_mean"], color="#e76f51", marker="s", linewidth=2.6, markersize=6.8, label="L2 distance")
    ax.set_title("Hidden-state divergence", fontsize=14.5, fontweight="semibold")
    ax.set_xlabel("Layer", fontsize=12.5, fontweight="semibold")
    ax.set_ylabel("Cosine similarity", fontsize=12.5, fontweight="semibold", color="#457b9d")
    ax2.set_ylabel("L2 distance", fontsize=12.5, fontweight="semibold", color="#e76f51")
    ax.tick_params(axis="y", colors="#457b9d")
    ax2.tick_params(axis="y", colors="#e76f51")
    _style_axis(ax, x, tick_positions, tick_labels)
    handles = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in handles]
    ax.legend(handles, labels, frameon=False, fontsize=11.8, loc="upper left", prop={"weight": "semibold", "size": 11.8})

    divergence_curves = []
    for mode in ("raw", "model_norm"):
        divergence_curves.extend([modes[mode]["tvd"]["layerwise_mean"], modes[mode]["js"]["layerwise_mean"]])
    div_min = min(min(curve) for curve in divergence_curves)
    div_max = max(max(curve) for curve in divergence_curves)
    div_pad = max((div_max - div_min) * 0.06, 1e-4)

    for idx, mode in enumerate(("raw", "model_norm"), start=1):
        ax = axes[idx]
        _plot_divergence_panel(ax, modes[mode], "Raw LogitDiff Lens" if mode == "raw" else "Model Norm LogitDiff Lens")
        ax.set_ylim(div_min - div_pad, div_max + div_pad)
        if idx == 2:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    fig.suptitle(f"Prompt-only Logit Lens divergence summary: {title_prefix}", fontsize=17, fontweight="bold", y=0.963)
    fig.tight_layout(rect=(0, 0, 1, 0.992), w_pad=0.45)
    fig.savefig(hidden_png, dpi=200, bbox_inches="tight")
    fig.savefig(hidden_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.1), sharey=True)
    row_meta = [("raw", "Raw LogitDiff Lens"), ("model_norm", "Model Norm LogitDiff Lens")]
    for ax, (mode, mode_label) in zip(axes, row_meta):
        _plot_jaccard_panel(ax, modes[mode], mode_label)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle(f"Prompt-only Logit Lens hypothesis-space overlap: {title_prefix}", fontsize=17, fontweight="bold", y=0.968)
    fig.tight_layout(rect=(0, 0, 1, 0.988), w_pad=0.75)
    fig.savefig(logit_png, dpi=200, bbox_inches="tight")
    fig.savefig(logit_pdf, bbox_inches="tight")
    plt.close(fig)

    table_rows = []
    for mode, label in row_meta:
        for metric, metric_label in (("jaccard_top1", "Jaccard@1"), ("jaccard_top5", "Jaccard@5"), ("jaccard_top10", "Jaccard@10"), ("tvd", "TVD"), ("js", "JS")):
            table_rows.append((label, metric_label, _picked_values(modes[mode][metric]["layerwise_mean"])))
    table_rows.extend([("Hidden states", "Cosine", _picked_values(hidden["hidden_cosine"]["layerwise_mean"])), ("Hidden states", "L2", _picked_values(hidden["hidden_l2"]["layerwise_mean"]))])

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Prompt-only Logit Lens summary for {title_prefix}. Overall values are means across all layers.}}",
        rf"\label{{{table_label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Section & Metric & First & Early & Mid & Late & Last & Overall \\",
        r"\midrule",
    ]
    for section, metric, vals in table_rows:
        lines.append(f"{section} & {metric} & {vals[0]:.3f} & {vals[1]:.3f} & {vals[2]:.3f} & {vals[3]:.3f} & {vals[4]:.3f} & {vals[5]:.3f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    table_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "summary_json": str(summary_json),
        "hidden_png": str(hidden_png),
        "hidden_pdf": str(hidden_pdf),
        "logit_png": str(logit_png),
        "logit_pdf": str(logit_pdf),
        "table_tex": str(table_tex),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic mode-specific summary plotter")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--title-prefix", required=True)
    parser.add_argument("--table-label", required=True)
    args = parser.parse_args()

    result = save_mode_specific_figures(
        args.summary_json,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        title_prefix=args.title_prefix,
        table_label=args.table_label,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

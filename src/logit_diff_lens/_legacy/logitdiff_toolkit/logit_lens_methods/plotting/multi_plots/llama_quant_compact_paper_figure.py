from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


ROOT = Path("/media/am/AM/logit-diff-lens")

RUN_ORDER = ["hf1bit", "bnb4", "bnb8"]
RUN_LABELS = {
    "hf1bit": "HF1BitLLM 1.58-bit",
    "bnb4": "4-bit BnB",
    "bnb8": "8-bit BnB",
}
RUN_COLORS = {
    "hf1bit": "#E9E2DA",
    "bnb4": "#5B6C9E",
    "bnb8": "#D97C6C",
}
RUN_MARKERS = {
    "hf1bit": "o",
    "bnb4": "s",
    "bnb8": "D",
}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _layer_ticks(n: int) -> tuple[list[int], list[str]]:
    x = list(range(n))
    tick_positions = x[::2] if n > 16 else x
    if x and x[-1] not in tick_positions:
        tick_positions = tick_positions + [x[-1]]
    tick_labels = [str(v + 1) for v in tick_positions]
    if tick_labels:
        tick_labels[-1] = "Last"
    return tick_positions, tick_labels


def _gen_layerwise_top5(run: str) -> list[float]:
    payload = _load_json(
        ROOT / f"tmp/quant_llama/gen_lens/{run}_qwen10/llama_chat/64/data/logitdiff_gen_all_layers_k10_t64.json"
    )
    rows = [row for row in payload["analysis_rows"] if row.get("is_generated", False)]
    by_layer: dict[int, list[float]] = {}
    for row in rows:
        layer = int(row["layer_absolute"])
        by_layer.setdefault(layer, []).append(float(row["top5_jaccard"]))
    return [float(np.mean(by_layer[layer])) for layer in sorted(by_layer)]


def _judge_rows() -> list[dict[str, Any]]:
    return _load_json(ROOT / "tmp/quant_llama/arbiter_new/summaries_deepseek/prompt_level_pair_faithful_scores.json")


def _judge_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {run: [] for run in RUN_ORDER}
    for row in rows:
        run = str(row["run"])
        if run in grouped:
            grouped[run].append(row)

    summary: dict[str, dict[str, Any]] = {}
    for run, run_rows in grouped.items():
        summary[run] = {
            "base_alignment_mean": float(np.mean([float(r["base_alignment"]) for r in run_rows])),
            "base_coherency_mean": float(np.mean([float(r["base_coherency"]) for r in run_rows])),
            "comparison_alignment_mean": float(np.mean([float(r["comparison_alignment"]) for r in run_rows])),
            "comparison_coherency_mean": float(np.mean([float(r["comparison_coherency"]) for r in run_rows])),
            "rows": run_rows,
        }
    return summary


def save_llama_quant_compact_paper_figure(
    *,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
    output_json: str | Path | None = None,
    top_linewidth: float = 2.8,
    top_marker_size: float = 7.8,
    top_endpoint_size: float = 92,
    top_title_size: float = 13.4,
    bottom_title_size: float = 13.4,
    axis_label_size: float = 11.5,
    shared_legend_size: float = 11.9,
    style_legend_size: float = 10.1,
    model_legend_marker_size: float = 9.6,
    style_legend_marker_size: float = 8.3,
    judge_linewidth: float = 1.1,
    judge_arrow_scale: float = 17,
    judge_arrow_linewidth: float = 2.2,
    judge_base_marker_size: float = 175,
    judge_comp_marker_size: float = 214,
    judge_ymin: float = -2,
    model_legend_anchor_y: float = 0.962,
    style_legend_anchor_y: float = 0.915,
    suptitle_size: float = 14.6,
    suptitle_y: float = 0.992,
    layout_rect_top: float = 0.905,
    subplot_hspace: float = 0.38,
    top_ylim_low: float = -0.02,
    top_ylim_high: float = 1.02,
    top_xmargin: float = 0.18,
    top_title_text: str | None = "Gen LogitDiff Lens",
) -> dict[str, Any]:
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(6.35, 8.15),
        gridspec_kw={"height_ratios": [0.96, 1.24]},
        constrained_layout=False,
    )

    shared_legend_fp = FontProperties(weight="semibold", size=shared_legend_size)
    style_legend_fp = FontProperties(weight="semibold", size=style_legend_size)

    series_len = None
    for run in RUN_ORDER:
        y = _gen_layerwise_top5(run)
        x = list(range(len(y)))
        series_len = len(y)
        ax_top.plot(
            x,
            y,
            color=RUN_COLORS[run],
            linewidth=top_linewidth,
            marker=RUN_MARKERS[run],
            markersize=top_marker_size,
            markeredgewidth=1.0,
            markeredgecolor="#111111",
            label=RUN_LABELS[run],
        )
        ax_top.scatter(
            [x[0], x[-1]],
            [y[0], y[-1]],
            s=top_endpoint_size,
            color=RUN_COLORS[run],
            marker=RUN_MARKERS[run],
            edgecolors="#111111",
            linewidths=1.15,
            zorder=4,
        )

    ticks, labels = _layer_ticks(series_len or 0)
    ax_top.set_xticks(ticks)
    ax_top.set_xticklabels(labels)
    ax_top.set_ylim(top_ylim_low, top_ylim_high)
    ax_top.set_xlim(-top_xmargin, (series_len - 1 if series_len else 0) + top_xmargin)
    ax_top.grid(True, alpha=0.22)
    if top_title_text:
        ax_top.set_title(top_title_text, fontsize=top_title_size, fontweight="semibold", pad=6)
    ax_top.set_ylabel("Jaccard@5", fontsize=axis_label_size, fontweight="semibold")
    ax_top.set_xlabel("Layer", fontsize=axis_label_size, fontweight="semibold")
    ax_top.tick_params(axis="x", labelsize=9.1, rotation=30)
    ax_top.tick_params(axis="y", labelsize=9.4)

    rows = _judge_rows()
    summary = _judge_summary(rows)

    for run in RUN_ORDER:
        color = RUN_COLORS[run]
        marker = RUN_MARKERS[run]
        run_rows = summary[run]["rows"]

        for row in run_rows:
            bx = float(row["base_coherency"])
            by = float(row["base_alignment"])
            fx = float(row["comparison_coherency"])
            fy = float(row["comparison_alignment"])
            ax_bottom.plot([bx, fx], [by, fy], color=color, alpha=0.22, linewidth=judge_linewidth, zorder=1)

        bx = summary[run]["base_coherency_mean"]
        by = summary[run]["base_alignment_mean"]
        fx = summary[run]["comparison_coherency_mean"]
        fy = summary[run]["comparison_alignment_mean"]

        arrow = FancyArrowPatch(
            (bx, by),
            (fx, fy),
            arrowstyle="-|>",
            mutation_scale=judge_arrow_scale,
            linewidth=judge_arrow_linewidth,
            color=color,
            alpha=0.95,
            zorder=2,
        )
        ax_bottom.add_patch(arrow)

        ax_bottom.scatter(
            [bx],
            [by],
            s=judge_base_marker_size,
            marker=marker,
            facecolors="white",
            edgecolors="black",
            linewidths=1.8,
            zorder=4,
        )
        ax_bottom.scatter(
            [fx],
            [fy],
            s=judge_comp_marker_size,
            marker=marker,
            facecolors=color,
            edgecolors="black",
            linewidths=1.8,
            zorder=5,
        )

    ax_bottom.set_title("DeepSeek-V3.2 judged alignment–coherence placement", fontsize=bottom_title_size, fontweight="semibold", pad=6)
    ax_bottom.set_xlim(-2, 102)
    ax_bottom.set_ylim(judge_ymin, 102)
    ax_bottom.set_xlabel("Judge Coherence Score", fontsize=axis_label_size, fontweight="semibold")
    ax_bottom.set_ylabel("Judge Alignment Score", fontsize=axis_label_size, fontweight="semibold")
    ax_bottom.grid(True, alpha=0.18, linewidth=0.8)
    ax_bottom.tick_params(axis="both", labelsize=9.4)
    ax_bottom.set_box_aspect(1)
    for spine in ax_bottom.spines.values():
        spine.set_alpha(0.45)

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=RUN_MARKERS[run],
            color="none",
            markerfacecolor=RUN_COLORS[run],
            markeredgecolor="black",
            markeredgewidth=1.7,
            markersize=model_legend_marker_size,
            label=RUN_LABELS[run],
        )
        for run in RUN_ORDER
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=style_legend_marker_size,
            label="Base mean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#BBBBBB",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=style_legend_marker_size,
            label="Quantized mean",
        ),
        Line2D([0, 1], [0, 0], color="#666666", linewidth=2.0, label="Mean base→quantized shift"),
    ]

    fig.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, model_legend_anchor_y),
        ncol=3,
        frameon=False,
        prop=shared_legend_fp,
        columnspacing=1.95,
        handletextpad=0.7,
    )
    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, style_legend_anchor_y),
        ncol=3,
        frameon=False,
        prop=style_legend_fp,
        columnspacing=1.55,
        handletextpad=0.7,
    )
    fig.suptitle(
        "LLaMA summary across LogitDiff and judged behavior",
        fontsize=suptitle_size,
        fontweight="semibold",
        y=suptitle_y,
    )
    fig.tight_layout(rect=(0, 0, 1, layout_rect_top))
    fig.subplots_adjust(hspace=subplot_hspace)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(Path(output_pdf), bbox_inches="tight")
    plt.close(fig)

    summary_payload = {
        "output_png": str(output_png),
        "output_pdf": str(output_pdf) if output_pdf else None,
        "judge_summary": summary,
    }
    if output_json is not None:
        Path(output_json).write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload


def main() -> None:
    save_llama_quant_compact_paper_figure(
        output_png=ROOT / "tmp/quant_llama/figures/llama_quant_compact_paper_figure.png",
        output_pdf=ROOT / "tmp/quant_llama/figures/llama_quant_compact_paper_figure.pdf",
        output_json=ROOT / "tmp/quant_llama/figures/llama_quant_compact_paper_figure_summary.json",
    )


if __name__ == "__main__":
    main()

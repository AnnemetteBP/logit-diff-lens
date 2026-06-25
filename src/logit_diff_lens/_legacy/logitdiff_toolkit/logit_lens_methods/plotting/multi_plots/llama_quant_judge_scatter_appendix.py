from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


ROOT = Path("/media/am/AM/logit-diff-lens")
SCORES_PATH = ROOT / "tmp/quant_llama/arbiter_new/summaries_deepseek/prompt_level_pair_faithful_scores.json"
OUT_DIR = ROOT / "tmp/quant_llama/arbiter_new/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {run: [] for run in RUN_ORDER}
    for row in rows:
        run = row["run"]
        if run in grouped:
            grouped[run].append(row)
    return grouped


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = _group(rows)
    summary: dict[str, dict[str, Any]] = {}
    for run, run_rows in grouped.items():
        summary[run] = {
            "base_alignment_mean": _mean([float(r["base_alignment"]) for r in run_rows]),
            "base_coherency_mean": _mean([float(r["base_coherency"]) for r in run_rows]),
            "comparison_alignment_mean": _mean([float(r["comparison_alignment"]) for r in run_rows]),
            "comparison_coherency_mean": _mean([float(r["comparison_coherency"]) for r in run_rows]),
            "rows": run_rows,
        }
    return summary


def make_figure() -> tuple[plt.Figure, dict[str, Any]]:
    rows = _load_rows(SCORES_PATH)
    summary = _build_summary(rows)

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 7.2), constrained_layout=False)

    for run in RUN_ORDER:
        color = RUN_COLORS[run]
        marker = RUN_MARKERS[run]
        run_rows = summary[run]["rows"]

        for row in run_rows:
            bx = float(row["base_coherency"])
            by = float(row["base_alignment"])
            fx = float(row["comparison_coherency"])
            fy = float(row["comparison_alignment"])
            ax.plot([bx, fx], [by, fy], color=color, alpha=0.22, linewidth=1.1, zorder=1)

        bx = summary[run]["base_coherency_mean"]
        by = summary[run]["base_alignment_mean"]
        fx = summary[run]["comparison_coherency_mean"]
        fy = summary[run]["comparison_alignment_mean"]

        arrow = FancyArrowPatch(
            (bx, by),
            (fx, fy),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.3,
            color=color,
            alpha=0.95,
            zorder=2,
        )
        ax.add_patch(arrow)

        ax.scatter(
            [bx],
            [by],
            s=180,
            marker=marker,
            facecolors="white",
            edgecolors="black",
            linewidths=1.9,
            zorder=4,
        )
        ax.scatter(
            [fx],
            [fy],
            s=220,
            marker=marker,
            facecolors=color,
            edgecolors="black",
            linewidths=1.9,
            zorder=5,
        )

    ax.set_title("DeepSeek-V3.2 Judge", fontsize=17, fontweight="semibold", pad=8)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Judge Coherence Score", fontsize=15, fontweight="semibold")
    ax.set_ylabel("Judge Alignment Score", fontsize=15, fontweight="semibold")
    ax.grid(True, alpha=0.18, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.45)
    ax.tick_params(axis="both", labelsize=13)

    model_items = [
        Line2D(
            [0],
            [0],
            marker=RUN_MARKERS[run],
            color="none",
            markerfacecolor=RUN_COLORS[run],
            markeredgecolor="black",
            markeredgewidth=1.7,
            markersize=12,
            label=RUN_LABELS[run],
        )
        for run in RUN_ORDER
    ]
    style_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.6,
            markersize=10,
            label="Base mean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#BBBBBB",
            markeredgecolor="black",
            markeredgewidth=1.6,
            markersize=10,
            label="Quantized mean",
        ),
        Line2D([0, 1], [0, 0], color="#666666", linewidth=2.0, label="Mean base→quantized shift"),
    ]

    legend_models = fig.legend(
        handles=model_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
        prop={"size": 16.5, "weight": "semibold"},
        columnspacing=2.0,
        handletextpad=0.7,
    )
    fig.add_artist(legend_models)
    fig.legend(
        handles=style_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.918),
        ncol=3,
        frameon=False,
        prop={"size": 13.5, "weight": "semibold"},
        columnspacing=1.7,
        handletextpad=0.7,
    )
    fig.suptitle(
        "LLaMA quantized judged alignment–coherence placement",
        fontsize=17,
        fontweight="semibold",
        y=1.01,
    )
    fig.subplots_adjust(top=0.79, left=0.12, right=0.98, bottom=0.12)

    return fig, summary


def main() -> None:
    fig, summary = make_figure()
    png_path = OUT_DIR / "llama_quant_judged_alignment_coherence_appendix.png"
    pdf_path = OUT_DIR / "llama_quant_judged_alignment_coherence_appendix.pdf"
    json_path = OUT_DIR / "llama_quant_judged_alignment_coherence_appendix_summary.json"
    tex_path = OUT_DIR / "llama_quant_judged_alignment_coherence_appendix_caption.tex"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tex_path.write_text(
        "\\caption{Judge-space placement for the quantized LLaMA comparisons using coherence on the x-axis and alignment on the y-axis. "
        "For each quantized model, the open marker shows the mean base-model score, the filled marker shows the mean quantized-model score, and the arrow indicates the mean base-to-quantized shift across the ten prompts.}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

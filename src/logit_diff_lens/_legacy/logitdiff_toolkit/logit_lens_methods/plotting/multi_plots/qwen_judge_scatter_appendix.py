from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


ROOT = Path("/media/am/AM/logit-diff-lens")
QWEN_SCORES = ROOT / "tmp/em_qwen/arbiter_new/summaries/prompt_level_pair_faithful_scores.json"
DEEPSEEK_SCORES = ROOT / "tmp/em_qwen/arbiter_new/summaries_deepseek/prompt_level_pair_faithful_scores.json"
OUT_DIR = ROOT / "tmp/em_qwen/arbiter/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["risky", "medical", "sports"]
MODEL_LABELS = {
    "risky": "Risky Financial Advice",
    "medical": "Bad Medical Advice",
    "sports": "Extreme Sports",
}
MODEL_COLORS = {
    "risky": "#E9E2DA",
    "medical": "#5B6C9E",
    "sports": "#D97C6C",
}
MODEL_MARKERS = {
    "risky": "o",
    "medical": "s",
    "sports": "D",
}


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _group_by_case(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in MODEL_ORDER}
    for row in rows:
        case = row["case"]
        if case in grouped:
            grouped[case].append(row)
    return grouped


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_panel_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = _group_by_case(rows)
    summary: dict[str, dict[str, Any]] = {}
    for case, case_rows in grouped.items():
        valid_rows = []
        for row in case_rows:
            ba = _to_float(row.get("base_alignment"))
            bc = _to_float(row.get("base_coherency"))
            fa = _to_float(row.get("ft_alignment"))
            fc = _to_float(row.get("ft_coherency"))
            if None in (ba, bc, fa, fc):
                continue
            valid_rows.append(
                {
                    **row,
                    "base_alignment_num": ba,
                    "base_coherency_num": bc,
                    "ft_alignment_num": fa,
                    "ft_coherency_num": fc,
                }
            )
        summary[case] = {
            "base_alignment_mean": _mean([r["base_alignment_num"] for r in valid_rows]),
            "base_coherency_mean": _mean([r["base_coherency_num"] for r in valid_rows]),
            "ft_alignment_mean": _mean([r["ft_alignment_num"] for r in valid_rows]),
            "ft_coherency_mean": _mean([r["ft_coherency_num"] for r in valid_rows]),
            "rows": valid_rows,
            "n_valid": len(valid_rows),
        }
    return summary


def _plot_panel(ax: plt.Axes, rows: list[dict[str, Any]], title: str, *, show_ylabel: bool) -> dict[str, Any]:
    summary = _build_panel_summary(rows)

    for case in MODEL_ORDER:
        color = MODEL_COLORS[case]
        marker = MODEL_MARKERS[case]
        case_rows = summary[case]["rows"]

        for row in case_rows:
            bx = row["base_coherency_num"]
            by = row["base_alignment_num"]
            fx = row["ft_coherency_num"]
            fy = row["ft_alignment_num"]
            ax.plot([bx, fx], [by, fy], color=color, alpha=0.22, linewidth=1.1, zorder=1)

        bx = summary[case]["base_coherency_mean"]
        by = summary[case]["base_alignment_mean"]
        fx = summary[case]["ft_coherency_mean"]
        fy = summary[case]["ft_alignment_mean"]

        arrow = FancyArrowPatch(
            (bx, by),
            (fx, fy),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color=color,
            alpha=0.92,
            zorder=2,
        )
        ax.add_patch(arrow)

        ax.scatter(
            [bx],
            [by],
            s=170,
            marker=marker,
            facecolors="white",
            edgecolors="black",
            linewidths=1.8,
            zorder=4,
        )
        ax.scatter(
            [fx],
            [fy],
            s=210,
            marker=marker,
            facecolors=color,
            edgecolors="black",
            linewidths=1.8,
            zorder=5,
        )

    ax.set_title(title, fontsize=16, fontweight="semibold", pad=8)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Judge Coherence Score", fontsize=15, fontweight="semibold")
    ax.set_ylabel("Judge Alignment Score" if show_ylabel else "", fontsize=15, fontweight="semibold")
    ax.grid(True, alpha=0.18, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.45)
    ax.tick_params(axis="both", labelsize=13)
    return summary


def make_figure() -> tuple[plt.Figure, dict[str, Any]]:
    qwen_rows = _load_json(QWEN_SCORES)
    deepseek_rows = _load_json(DEEPSEEK_SCORES)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.0), constrained_layout=False, sharey=True)
    qwen_summary = _plot_panel(axes[0], qwen_rows, "Qwen2.5-14B-Instruct Judge", show_ylabel=True)
    deepseek_summary = _plot_panel(axes[1], deepseek_rows, "DeepSeek-V3.2 Judge", show_ylabel=False)
    axes[0].set_title("Qwen2.5-14B-Instruct Judge", fontsize=17, fontweight="semibold", pad=8)
    axes[1].set_title("DeepSeek-V3.2 Judge", fontsize=17, fontweight="semibold", pad=8)

    model_items = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[case],
            color="none",
            markerfacecolor=MODEL_COLORS[case],
            markeredgecolor="black",
            markeredgewidth=1.6,
            markersize=12,
            label=MODEL_LABELS[case],
        )
        for case in MODEL_ORDER
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
            markerfacecolor="#888888",
            markeredgecolor="black",
            markeredgewidth=1.6,
            markersize=10,
            label="FT mean",
        ),
        Line2D([0, 1], [0, 0], color="#666666", linewidth=2.0, label="Mean base→FT shift"),
    ]
    legend_models = fig.legend(
        handles=model_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
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
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
        frameon=False,
        prop={"size": 15.5, "weight": "semibold"},
        columnspacing=1.7,
        handletextpad=0.7,
    )
    fig.suptitle(
        "Qwen judged alignment–coherence placement",
        fontsize=20,
        fontweight="semibold",
        y=1.07,
    )
    fig.subplots_adjust(top=0.785, wspace=0.08, left=0.07, right=0.985, bottom=0.12)

    summary = {
        "qwen14b": qwen_summary,
        "deepseek": deepseek_summary,
    }
    return fig, summary


def main() -> None:
    fig, summary = make_figure()
    png_path = OUT_DIR / "qwen_judged_alignment_coherence_appendix.png"
    pdf_path = OUT_DIR / "qwen_judged_alignment_coherence_appendix.pdf"
    json_path = OUT_DIR / "qwen_judged_alignment_coherence_appendix_summary.json"
    tex_path = OUT_DIR / "qwen_judged_alignment_coherence_appendix_caption.tex"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    tex_path.write_text(
        "\\caption{Judge-space placement for the Qwen comparisons using coherence on the x-axis and alignment on the y-axis. "
        "Each panel corresponds to one judge. For each finetuned model, the open marker shows the mean base-model score, the filled marker shows the mean finetuned-model score, and the arrow indicates the mean base-to-finetuned shift across the ten prompts.}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

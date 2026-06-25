from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


ROOT = Path("/media/am/AM/logit-diff-lens")

QWEN_QWEN14_SCORES = ROOT / "tmp/em_qwen/arbiter_new/summaries/prompt_level_pair_faithful_scores.json"
QWEN_DEEPSEEK_SCORES = ROOT / "tmp/em_qwen/arbiter_new/summaries_deepseek/prompt_level_pair_faithful_scores.json"
LLAMA_DEEPSEEK_SCORES = ROOT / "tmp/quant_llama/arbiter_new/summaries_deepseek/prompt_level_pair_faithful_scores.json"

QWEN_OUT_DIR = ROOT / "tmp/em_qwen/arbiter/figures"
LLAMA_OUT_DIR = ROOT / "tmp/quant_llama/arbiter/figures"
QWEN_OUT_DIR.mkdir(parents=True, exist_ok=True)
LLAMA_OUT_DIR.mkdir(parents=True, exist_ok=True)

QWEN_ORDER = ["risky", "medical", "sports"]
QWEN_LABELS = {
    "risky": "Risky Financial Advice",
    "medical": "Bad Medical Advice",
    "sports": "Extreme Sports",
}
QWEN_COLORS = {
    "risky": "#E9E2DA",
    "medical": "#5B6C9E",
    "sports": "#D97C6C",
}
QWEN_MARKERS = {
    "risky": "o",
    "medical": "s",
    "sports": "D",
}

QWEN_COLORS_ALT = {
    "risky": "#FCFDBF",
    "medical": "#F98E52",
    "sports": "#7E2482",
}

LLAMA_ORDER = ["hf1bit", "bnb4", "bnb8"]
LLAMA_LABELS = {
    "hf1bit": "HF1BitLLM 1.58-bit",
    "bnb4": "4-bit BnB",
    "bnb8": "8-bit BnB",
}
LLAMA_COLORS = {
    "hf1bit": "#E9E2DA",
    "bnb4": "#5B6C9E",
    "bnb8": "#D97C6C",
}
LLAMA_MARKERS = {
    "hf1bit": "o",
    "bnb4": "s",
    "bnb8": "D",
}

LLAMA_COLORS_ALT = {
    "hf1bit": "#0D0887",
    "bnb4": "#9C179E",
    "bnb8": "#ED7953",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_by_key(rows: list[dict[str, Any]], key_name: str, order: list[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in order}
    for row in rows:
        key = str(row[key_name])
        if key in grouped:
            grouped[key].append(row)
    return grouped


def _build_qwen_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = _group_by_key(rows, "case", QWEN_ORDER)
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
            "comparison_alignment_mean": _mean([r["ft_alignment_num"] for r in valid_rows]),
            "comparison_coherency_mean": _mean([r["ft_coherency_num"] for r in valid_rows]),
            "rows": valid_rows,
            "n_valid": len(valid_rows),
        }
    return summary


def _build_llama_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = _group_by_key(rows, "run", LLAMA_ORDER)
    summary: dict[str, dict[str, Any]] = {}
    for run, run_rows in grouped.items():
        summary[run] = {
            "base_alignment_mean": _mean([float(r["base_alignment"]) for r in run_rows]),
            "base_coherency_mean": _mean([float(r["base_coherency"]) for r in run_rows]),
            "comparison_alignment_mean": _mean([float(r["comparison_alignment"]) for r in run_rows]),
            "comparison_coherency_mean": _mean([float(r["comparison_coherency"]) for r in run_rows]),
            "rows": run_rows,
            "n_valid": len(run_rows),
        }
    return summary


def _make_single_figure(
    *,
    summary: dict[str, dict[str, Any]],
    order: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    markers: dict[str, str],
    title: str,
    family_title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 7.2), constrained_layout=False)

    for key in order:
        color = colors[key]
        marker = markers[key]
        rows = summary[key]["rows"]

        for row in rows:
            bx = float(row["base_coherency_num"] if "base_coherency_num" in row else row["base_coherency"])
            by = float(row["base_alignment_num"] if "base_alignment_num" in row else row["base_alignment"])
            fx = float(row["ft_coherency_num"] if "ft_coherency_num" in row else row["comparison_coherency"])
            fy = float(row["ft_alignment_num"] if "ft_alignment_num" in row else row["comparison_alignment"])
            ax.plot([bx, fx], [by, fy], color=color, alpha=0.22, linewidth=1.1, zorder=1)

        bx = summary[key]["base_coherency_mean"]
        by = summary[key]["base_alignment_mean"]
        fx = summary[key]["comparison_coherency_mean"]
        fy = summary[key]["comparison_alignment_mean"]

        arrow = FancyArrowPatch(
            (bx, by),
            (fx, fy),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color=color,
            alpha=0.94,
            zorder=2,
        )
        ax.add_patch(arrow)

        ax.scatter(
            [bx],
            [by],
            s=178,
            marker=marker,
            facecolors="white",
            edgecolors="black",
            linewidths=1.8,
            zorder=4,
        )
        ax.scatter(
            [fx],
            [fy],
            s=216,
            marker=marker,
            facecolors=color,
            edgecolors="black",
            linewidths=1.8,
            zorder=5,
        )

    ax.set_title(title, fontsize=17, fontweight="semibold", pad=8)
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
            marker=markers[key],
            color="none",
            markerfacecolor=colors[key],
            markeredgecolor="black",
            markeredgewidth=1.7,
            markersize=12,
            label=labels[key],
        )
        for key in order
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
            label="Comparison mean",
        ),
        Line2D([0, 1], [0, 0], color="#666666", linewidth=2.0, label="Mean base→comparison shift"),
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
        family_title,
        fontsize=17,
        fontweight="semibold",
        y=1.01,
    )
    fig.subplots_adjust(top=0.79, left=0.12, right=0.98, bottom=0.12)
    return fig


def _save(fig: plt.Figure, summary: dict[str, Any], *, png_path: Path, pdf_path: Path, json_path: Path, tex_path: Path, caption: str) -> None:
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tex_path.write_text(caption, encoding="utf-8")


def main() -> None:
    qwen14_rows = _load_rows(QWEN_QWEN14_SCORES)
    qwen_deep_rows = _load_rows(QWEN_DEEPSEEK_SCORES)
    llama_deep_rows = _load_rows(LLAMA_DEEPSEEK_SCORES)

    qwen14_summary = _build_qwen_summary(qwen14_rows)
    qwen_deep_summary = _build_qwen_summary(qwen_deep_rows)
    llama_deep_summary = _build_llama_summary(llama_deep_rows)

    fig = _make_single_figure(
        summary=qwen14_summary,
        order=QWEN_ORDER,
        labels=QWEN_LABELS,
        colors=QWEN_COLORS,
        markers=QWEN_MARKERS,
        title="Qwen2.5-14B-Instruct Judge",
        family_title="Qwen judged alignment–coherence placement",
    )
    _save(
        fig,
        qwen14_summary,
        png_path=QWEN_OUT_DIR / "qwen_judged_alignment_coherence_qwen14_appendix.png",
        pdf_path=QWEN_OUT_DIR / "qwen_judged_alignment_coherence_qwen14_appendix.pdf",
        json_path=QWEN_OUT_DIR / "qwen_judged_alignment_coherence_qwen14_appendix_summary.json",
        tex_path=QWEN_OUT_DIR / "qwen_judged_alignment_coherence_qwen14_appendix_caption.tex",
        caption="\\caption{Judge-space placement for the Qwen comparisons under the Qwen2.5-14B-Instruct judge, using coherence on the x-axis and alignment on the y-axis. For each finetuned model, the open marker shows the mean base-model score, the filled marker shows the mean finetuned-model score, and the arrow indicates the mean base-to-finetuned shift across the ten prompts.}\n",
    )

    fig = _make_single_figure(
        summary=qwen_deep_summary,
        order=QWEN_ORDER,
        labels=QWEN_LABELS,
        colors=QWEN_COLORS,
        markers=QWEN_MARKERS,
        title="DeepSeek-V3.2 Judge",
        family_title="Qwen judged alignment–coherence placement",
    )
    _save(
        fig,
        qwen_deep_summary,
        png_path=QWEN_OUT_DIR / "qwen_judged_alignment_coherence_deepseek_appendix.png",
        pdf_path=QWEN_OUT_DIR / "qwen_judged_alignment_coherence_deepseek_appendix.pdf",
        json_path=QWEN_OUT_DIR / "qwen_judged_alignment_coherence_deepseek_appendix_summary.json",
        tex_path=QWEN_OUT_DIR / "qwen_judged_alignment_coherence_deepseek_appendix_caption.tex",
        caption="\\caption{Judge-space placement for the Qwen comparisons under the DeepSeek-V3.2 judge, using coherence on the x-axis and alignment on the y-axis. For each finetuned model, the open marker shows the mean base-model score, the filled marker shows the mean finetuned-model score, and the arrow indicates the mean base-to-finetuned shift across the ten prompts.}\n",
    )

    fig = _make_single_figure(
        summary=llama_deep_summary,
        order=LLAMA_ORDER,
        labels=LLAMA_LABELS,
        colors=LLAMA_COLORS,
        markers=LLAMA_MARKERS,
        title="DeepSeek-V3.2 Judge",
        family_title="LLaMA quantized judged alignment–coherence placement",
    )
    _save(
        fig,
        llama_deep_summary,
        png_path=LLAMA_OUT_DIR / "llama_quant_judged_alignment_coherence_deepseek_appendix.png",
        pdf_path=LLAMA_OUT_DIR / "llama_quant_judged_alignment_coherence_deepseek_appendix.pdf",
        json_path=LLAMA_OUT_DIR / "llama_quant_judged_alignment_coherence_deepseek_appendix_summary.json",
        tex_path=LLAMA_OUT_DIR / "llama_quant_judged_alignment_coherence_deepseek_appendix_caption.tex",
        caption="\\caption{Judge-space placement for the quantized LLaMA comparisons under the DeepSeek-V3.2 judge, using coherence on the x-axis and alignment on the y-axis. For each quantized model, the open marker shows the mean base-model score, the filled marker shows the mean quantized-model score, and the arrow indicates the mean base-to-quantized shift across the ten prompts.}\n",
    )

    alt_out = ROOT / "tmp/ucloud_patchscope_runs/summaries/by_family_alt_cmaps"
    alt_out.mkdir(parents=True, exist_ok=True)

    fig = _make_single_figure(
        summary=qwen14_summary,
        order=QWEN_ORDER,
        labels=QWEN_LABELS,
        colors=QWEN_COLORS_ALT,
        markers=QWEN_MARKERS,
        title="Qwen2.5-14B-Instruct Judge",
        family_title="Qwen judged alignment–coherence placement",
    )
    _save(
        fig,
        qwen14_summary,
        png_path=alt_out / "qwen_judged_alignment_coherence_qwen14_alt_cmaps.png",
        pdf_path=alt_out / "qwen_judged_alignment_coherence_qwen14_alt_cmaps.pdf",
        json_path=alt_out / "qwen_judged_alignment_coherence_qwen14_alt_cmaps_summary.json",
        tex_path=alt_out / "qwen_judged_alignment_coherence_qwen14_alt_cmaps_caption.tex",
        caption="\\caption{Judge-space placement for the Qwen comparisons under the Qwen2.5-14B-Instruct judge, using the alternate Qwen family palette.}\n",
    )

    fig = _make_single_figure(
        summary=qwen_deep_summary,
        order=QWEN_ORDER,
        labels=QWEN_LABELS,
        colors=QWEN_COLORS_ALT,
        markers=QWEN_MARKERS,
        title="DeepSeek-V3.2 Judge",
        family_title="Qwen judged alignment–coherence placement",
    )
    _save(
        fig,
        qwen_deep_summary,
        png_path=alt_out / "qwen_judged_alignment_coherence_deepseek_alt_cmaps.png",
        pdf_path=alt_out / "qwen_judged_alignment_coherence_deepseek_alt_cmaps.pdf",
        json_path=alt_out / "qwen_judged_alignment_coherence_deepseek_alt_cmaps_summary.json",
        tex_path=alt_out / "qwen_judged_alignment_coherence_deepseek_alt_cmaps_caption.tex",
        caption="\\caption{Judge-space placement for the Qwen comparisons under the DeepSeek-V3.2 judge, using the alternate Qwen family palette.}\n",
    )

    fig = _make_single_figure(
        summary=llama_deep_summary,
        order=LLAMA_ORDER,
        labels=LLAMA_LABELS,
        colors=LLAMA_COLORS_ALT,
        markers=LLAMA_MARKERS,
        title="DeepSeek-V3.2 Judge",
        family_title="LLaMA quantized judged alignment–coherence placement",
    )
    _save(
        fig,
        llama_deep_summary,
        png_path=alt_out / "llama_quant_judged_alignment_coherence_deepseek_alt_cmaps.png",
        pdf_path=alt_out / "llama_quant_judged_alignment_coherence_deepseek_alt_cmaps.pdf",
        json_path=alt_out / "llama_quant_judged_alignment_coherence_deepseek_alt_cmaps_summary.json",
        tex_path=alt_out / "llama_quant_judged_alignment_coherence_deepseek_alt_cmaps_caption.tex",
        caption="\\caption{Judge-space placement for the quantized LLaMA comparisons under the DeepSeek-V3.2 judge, using the alternate LLaMA quantization palette.}\n",
    )


if __name__ == "__main__":
    main()

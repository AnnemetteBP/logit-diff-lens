from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


ROOT = Path("/media/am/AM/logit-diff-lens")

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
    "sports": "^",
}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _layer_ticks(n: int) -> tuple[list[int], list[str]]:
    x = list(range(n))
    tick_positions = x[::2] if n > 14 else x
    if x and x[-1] not in tick_positions:
        tick_positions = tick_positions + [x[-1]]
    tick_labels = [str(v + 1) for v in tick_positions]
    if tick_labels:
        tick_labels[-1] = "Last"
    return tick_positions, tick_labels


def _gen_layerwise_top5(case: str) -> list[float]:
    payload = _load_json(
        ROOT / f"tmp/em_qwen/gen_lens/data/{case}/chat_template/64/{case}_chat_template_10_t64_layerwise.json"
    )
    rows = [row for row in payload["analysis_rows"] if row.get("is_generated", False)]
    by_layer: dict[int, list[float]] = {}
    for row in rows:
        layer = int(row["layer_absolute"])
        by_layer.setdefault(layer, []).append(float(row["top5_jaccard"]))
    return [float(np.mean(by_layer[layer])) for layer in sorted(by_layer)]


def _judge_alignment_deltas(summary_path: str | Path) -> dict[str, float]:
    rows = _load_json(summary_path)
    return {str(row["case"]): float(row["mean_delta_alignment_ft_minus_base"]) for row in rows}


def save_qwen_compact_paper_figure(
    *,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.4, 7.3),
        gridspec_kw={"height_ratios": [1.15, 0.95]},
        constrained_layout=False,
    )

    semibold_legend = FontProperties(weight="semibold", size=14.2)
    semibold_shared_legend = FontProperties(weight="semibold", size=15.4)

    ax = axes[0]
    series_len = None
    for case in MODEL_ORDER:
        y = _gen_layerwise_top5(case)
        x = list(range(len(y)))
        series_len = len(y)
        ax.plot(
            x,
            y,
            color=MODEL_COLORS[case],
            linewidth=2.8,
            marker=MODEL_MARKERS[case],
            markersize=7.8,
            markeredgewidth=1.0,
            markeredgecolor="#111111",
            label=MODEL_LABELS[case],
        )
        ax.scatter(
            [x[0], x[-1]],
            [y[0], y[-1]],
            s=92,
            color=MODEL_COLORS[case],
            marker=MODEL_MARKERS[case],
            edgecolors="#111111",
            linewidths=1.15,
            zorder=4,
        )
    ticks, labels = _layer_ticks(series_len or 0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.22)
    ax.set_title("Gen LogitDiff Lens", fontsize=15.2, fontweight="semibold", pad=7)
    ax.set_ylabel("Jaccard@5", fontsize=13.2, fontweight="semibold")
    ax.set_xlabel("Layer", fontsize=13.2, fontweight="semibold")
    ax.tick_params(axis="x", labelsize=10.3, rotation=32)
    ax.tick_params(axis="y", labelsize=10.6)

    qwen14 = _judge_alignment_deltas(ROOT / "tmp/em_qwen/arbiter_new/summaries/prompt_level_pair_faithful_summary.json")
    deepseek = _judge_alignment_deltas(ROOT / "tmp/em_qwen/arbiter_new/summaries_deepseek/prompt_level_pair_faithful_summary.json")

    ax = axes[1]
    y = np.arange(len(MODEL_ORDER))
    height = 0.34
    y_qwen = [qwen14[c] for c in MODEL_ORDER]
    y_deep = [deepseek[c] for c in MODEL_ORDER]
    ax.axvline(0.0, color="#666666", linewidth=1.0, alpha=0.55)
    ax.barh(
        y + height / 2,
        y_qwen,
        height=height,
        color="#374151",
        label="Qwen2.5-14B judge",
    )
    ax.barh(
        y - height / 2,
        y_deep,
        height=height,
        color="#9ca3af",
        label="DeepSeek-V3.2 judge",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(["Risky", "Medical", "Sports"], fontsize=11.0, fontweight="semibold")
    ax.invert_yaxis()
    ax.set_title("Judged Alignment Shift", fontsize=15.2, fontweight="semibold", pad=7)
    ax.set_xlabel("Mean FT - base alignment", fontsize=13.2, fontweight="semibold")
    ax.grid(True, axis="x", alpha=0.22)
    ax.tick_params(axis="x", labelsize=10.6)
    ax.tick_params(axis="y", length=0)
    x_min = min(y_qwen + y_deep)
    ax.set_xlim(x_min - 4.0, 2.0)
    ax.legend(frameon=False, loc="lower left", prop=semibold_legend)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        prop=semibold_shared_legend,
    )
    fig.suptitle(
        "Qwen summary across LogitDiff and judged behavior",
        fontsize=17.2,
        fontweight="semibold",
        y=0.992,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.subplots_adjust(hspace=0.40)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(Path(output_pdf), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "output_png": str(output_png),
        "output_pdf": str(output_pdf) if output_pdf else None,
        "judges": {
            "qwen2p5_14b_alignment_delta": qwen14,
            "deepseek_v3p2_alignment_delta": deepseek,
        },
    }
    if output_json is not None:
        Path(output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    save_qwen_compact_paper_figure(
        output_png=ROOT / "tmp/em_qwen/figures/qwen_compact_paper_figure.png",
        output_pdf=ROOT / "tmp/em_qwen/figures/qwen_compact_paper_figure.pdf",
        output_json=ROOT / "tmp/em_qwen/figures/qwen_compact_paper_figure_summary.json",
    )


if __name__ == "__main__":
    main()

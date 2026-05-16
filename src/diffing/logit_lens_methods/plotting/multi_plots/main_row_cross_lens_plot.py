from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


GEN_TEMPLATE_LABELS = {
    "Original Qwen template": "#d1495b",
    "Neutral chat template": "#2a9d8f",
    "No template": "#1d3557",
}
GEN_TEMPLATE_MARKERS = {
    "Original Qwen template": "o",
    "Neutral chat template": "s",
    "No template": "^",
}

MODE_COLORS = {
    "raw": "#d1495b",
    "model_norm": "#1d3557",
}
MODE_MARKERS = {
    "raw": "o",
    "model_norm": "s",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_layer_ticks(x: list[int]) -> tuple[list[int], list[str]]:
    tick_positions = list(x)
    tick_labels = [str(v + 1) for v in tick_positions]
    if tick_labels:
        tick_labels[-1] = "Last"
    return tick_positions, tick_labels


def save_main_row_cross_lens_plot(
    *,
    generation_summary_json: str | Path,
    tf_summary_json: str | Path,
    concept_summary_json: str | Path,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
) -> dict[str, Any]:
    gen_payload = _load_json(generation_summary_json)
    tf_payload = _load_json(tf_summary_json)
    concept_payload = _load_json(concept_summary_json)

    fig, axes = plt.subplots(1, 3, figsize=(18.9, 4.9), sharex=False)
    legend_prop = FontProperties(size=12.5, weight="semibold")

    # Panel 1: generation top-10 Jaccard by template
    ax = axes[0]
    for summary in gen_payload["summaries"]:
        x = [int(row["layer_absolute"]) for row in summary["layers"]]
        y = [float(row["top10_mean"]) for row in summary["layers"]]
        label = summary["template_label"]
        color = GEN_TEMPLATE_LABELS.get(label, None)
        ax.plot(
            x,
            y,
            label=label,
            linewidth=2.4,
            marker=GEN_TEMPLATE_MARKERS.get(label, "o"),
            markersize=6.0,
            markeredgewidth=0,
            color=color,
        )
    tick_positions, tick_labels = _format_layer_ticks(x)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title("Generation Top-10 Jaccard", fontsize=15, fontweight="semibold")
    ax.set_xlabel("Layer", fontsize=12.5, fontweight="semibold")
    ax.set_ylabel("Mean Jaccard overlap", fontsize=12.5, fontweight="semibold")
    ax.tick_params(axis="x", labelsize=9.5, rotation=36)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(frameon=False, loc="upper left", prop=legend_prop)

    # Panel 2: TF top-10 Jaccard by lens type
    ax = axes[1]
    for mode_key, mode_label in [("raw", "Raw LogitDiff Lens"), ("model_norm", "Model Norm LogitDiff Lens")]:
        y = tf_payload["modes"][mode_key]["jaccard_top10"]["layerwise_mean"]
        x = list(range(len(y)))
        ax.plot(
            x,
            y,
            label=mode_label,
            linewidth=2.4,
            marker=MODE_MARKERS[mode_key],
            markersize=6.0,
            markeredgewidth=0,
            color=MODE_COLORS[mode_key],
        )
    tick_positions, tick_labels = _format_layer_ticks(x)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title("Teacher-Forcing Top-10 Jaccard", fontsize=15, fontweight="semibold")
    ax.set_xlabel("Layer", fontsize=12.5, fontweight="semibold")
    ax.tick_params(axis="x", labelsize=9.5, rotation=36)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(frameon=False, loc="lower right", prop=legend_prop)
    ax.set_yticks(axes[0].get_yticks())
    ax.set_ylim(axes[0].get_ylim())
    ax.tick_params(axis="y", left=True, labelleft=True)

    # Panel 3: prescriptive onset marker-mass gap by lens type
    ax = axes[2]
    prescriptive = concept_payload["results"]["results"]["prescriptive"]["layerwise"]
    for mode_key, mode_label in [("raw", "Raw LogitDiff Lens"), ("model_norm", "Model Norm LogitDiff Lens")]:
        rows = prescriptive[mode_key]
        x = [int(r["layer_index"]) for r in rows]
        y = [float(r["gap_onset_marker_mass"]) for r in rows]
        ax.plot(
            x,
            y,
            label=mode_label,
            linewidth=2.4,
            marker=MODE_MARKERS[mode_key],
            markersize=6.0,
            markeredgewidth=0,
            color=MODE_COLORS[mode_key],
        )
    tick_positions, tick_labels = _format_layer_ticks(x)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.6)
    ax.grid(True, alpha=0.25)
    ax.set_title("Prescriptive Readout (FT - Base)", fontsize=15, fontweight="semibold")
    ax.set_xlabel("Layer", fontsize=12.5, fontweight="semibold")
    ax.tick_params(axis="x", labelsize=9.5, rotation=36)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(frameon=False, loc="upper left", prop=legend_prop)

    fig.suptitle(
        "Late-layer divergence and concept shifts across generation and teacher forcing",
        fontsize=17,
        fontweight="bold",
        y=0.90,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.subplots_adjust(wspace=0.08)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(Path(output_pdf), bbox_inches="tight")
    plt.close(fig)

    return {
        "output_png": str(output_png),
        "output_pdf": str(output_pdf) if output_pdf else None,
    }


def main() -> None:
    root = Path("/media/am/AM/logit-diff-lens")
    save_main_row_cross_lens_plot(
        generation_summary_json=root / "tmp/qwen_risky/logitdiff_gen/appendix_layer_summaries/generation_template_layerwise_jaccard_summary.json",
        tf_summary_json=root / "tmp/qwen_risky/tf_prompt_only/layerwise_comparison_light/layerwise_tf_comparison_summary.json",
        concept_summary_json=root / "tmp/qwen_risky/tf_prompt_only/concept_readouts/concept_readout_summary.json",
        output_png=root / "tmp/qwen_risky/main_row_cross_lens/main_row_cross_lens.png",
        output_pdf=root / "tmp/qwen_risky/main_row_cross_lens/main_row_cross_lens.pdf",
    )


if __name__ == "__main__":
    main()

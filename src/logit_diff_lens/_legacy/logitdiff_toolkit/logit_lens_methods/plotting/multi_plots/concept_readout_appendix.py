from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


CONCEPT_LABELS = {
    "prescriptive": "Prescriptive",
    "refusal": "Refusal",
    "risky_action": "Risky-Action",
}

MODE_ORDER = [
    ("raw", "Raw LogitDiff Lens"),
    ("model_norm", "Model Norm LogitDiff Lens"),
]

MODE_COLORS = {
    "raw": "#d1495b",
    "model_norm": "#1d3557",
}


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


def _format_layer_ticks(x: list[int]) -> tuple[list[int], list[str]]:
    tick_positions = x[::2] if len(x) > 14 else list(x)
    if x and x[-1] not in tick_positions:
        tick_positions = list(tick_positions) + [x[-1]]
    tick_labels = [str(v + 1) for v in tick_positions]
    if tick_positions:
        tick_labels[-1] = "Last"
    return tick_positions, tick_labels


def save_concept_readout_summary_plot(
    concept_summary_json: str | Path,
    *,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
) -> dict[str, Any]:
    payload = _load_json(concept_summary_json)
    concepts = payload["results"]["concepts"]

    fig, axes = plt.subplots(3, 2, figsize=(13.8, 11.8), sharex=True, sharey="row")

    for row_idx, concept in enumerate(concepts):
        concept_data = payload["results"]["results"][concept]
        for col_idx, (mode_key, mode_label) in enumerate(MODE_ORDER):
            ax = axes[row_idx, col_idx]
            rows = concept_data["layerwise"][mode_key]
            x = [int(r["layer_index"]) for r in rows]
            y = [float(r["gap_onset_marker_mass"]) for r in rows]
            tick_positions, tick_labels = _format_layer_ticks(x)

            ax.plot(
                x,
                y,
                color=MODE_COLORS[mode_key],
                linewidth=2.5,
                marker="o",
                markersize=5.0,
                markeredgewidth=0,
            )
            ax.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.6)
            ax.grid(True, alpha=0.25)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)

            if row_idx == 0:
                ax.set_title(mode_label, fontsize=16, fontweight="semibold")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{CONCEPT_LABELS.get(concept, concept)}\nFT - Base mass",
                    fontsize=12.5,
                    fontweight="semibold",
                )
            if row_idx == len(concepts) - 1:
                ax.set_xlabel("Layer", fontsize=13, fontweight="semibold")

    fig.suptitle(
        "Teacher-forcing concept readout: onset marker-mass gaps",
        fontsize=18,
        fontweight="bold",
        y=0.988,
    )
    fig.text(
        0.5,
        0.957,
        "Final prompt-token position predicting the first response token",
        ha="center",
        fontsize=12.5,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(Path(output_pdf), bbox_inches="tight")
    plt.close(fig)

    return {
        "summary_path": str(concept_summary_json),
        "figure_png": str(output_png),
        "figure_pdf": str(output_pdf) if output_pdf else None,
    }


def save_concept_readout_summary_table(
    concept_summary_json: str | Path,
    *,
    output_tex: str | Path,
) -> str:
    payload = _load_json(concept_summary_json)
    concepts = payload["results"]["concepts"]

    lines = [
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Concept & Lens & First & Early & Mid & Late & Last & Overall & Peak \\",
        r"\midrule",
    ]

    for concept in concepts:
        concept_data = payload["results"]["results"][concept]
        for mode_key, mode_label in MODE_ORDER:
            rows = concept_data["layerwise"][mode_key]
            num_layers = len(rows)
            picks = _pick_layer_indices(num_layers)
            values = {int(r["layer_index"]): r for r in rows}
            selected = [
                float(values[picks[name]]["gap_onset_marker_mass"])
                for name in ["first", "early", "mid", "late", "last"]
            ]
            overall = sum(float(r["gap_onset_marker_mass"]) for r in rows) / len(rows)
            peak = max(rows, key=lambda r: float(r["gap_onset_marker_mass"]))
            concept_label = CONCEPT_LABELS.get(concept, concept)
            lines.append(
                f"{concept_label} & {mode_label} & "
                f"{selected[0]:.3f} & {selected[1]:.3f} & {selected[2]:.3f} & {selected[3]:.3f} & {selected[4]:.3f} & "
                f"{overall:.3f} & {int(peak['layer_index']) + 1} ({float(peak['gap_onset_marker_mass']):.3f}) \\\\"
            )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tabular = "\n".join(lines)
    wrapped = _wrap_table(
        tabular,
        caption=(
            "Prompt-only teacher-forcing concept readout summary. Entries report finetuned-minus-base "
            "onset marker-mass gaps at the final prompt-token position, i.e., the position whose logits predict "
            "the first response token. Positive values indicate that the finetuned model assigns more probability "
            "mass to the given concept lexicon than the base model."
        ),
        label="tab:tf_concept_readout_summary",
    )
    output_tex = Path(output_tex)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(wrapped + "\n", encoding="utf-8")
    return str(output_tex)


def main() -> None:
    root = Path("/media/am/AM/logit-diff-lens")
    summary_path = root / "tmp/qwen_risky/tf_prompt_only/concept_readouts/concept_readout_summary.json"
    out_dir = root / "tmp/qwen_risky/tf_prompt_only/concept_readouts/appendix_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_concept_readout_summary_plot(
        summary_path,
        output_png=out_dir / "tf_concept_readout_summary.png",
        output_pdf=out_dir / "tf_concept_readout_summary.pdf",
    )
    save_concept_readout_summary_table(
        summary_path,
        output_tex=out_dir / "tf_concept_readout_summary_table.tex",
    )


if __name__ == "__main__":
    main()

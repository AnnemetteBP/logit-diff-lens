from __future__ import annotations

import json
from pathlib import Path


SELECTED_LAYERS = [
    ("First", 0),
    ("Early", 7),
    ("Mid", 14),
    ("Late", 20),
    ("Last", 27),
]


def _fmt(x: float) -> str:
    if abs(x) < 5e-4:
        x = 0.0
    return f"{x:.3f}"


def build_prescriptive_marker_table(marker_summary_path: Path) -> str:
    obj = json.loads(marker_summary_path.read_text(encoding="utf-8"))
    layerwise = obj["results"]["layerwise"]

    rows: list[str] = []
    for mode_key, mode_label in [
        ("raw", "Raw LogitDiff Lens"),
        ("model_norm", "Model Norm LogitDiff Lens"),
    ]:
        values = {int(row["layer_index"]): row for row in layerwise[mode_key]}
        overall_mass = sum(float(row["gap_onset_marker_mass"]) for row in values.values()) / len(values)
        overall_top10 = sum(float(row["gap_onset_top10_hit"]) for row in values.values()) / len(values)
        best_mass = max(values.values(), key=lambda row: float(row["gap_onset_marker_mass"]))
        best_top10 = max(values.values(), key=lambda row: float(row["gap_onset_top10_hit"]))

        mass_cells = [_fmt(float(values[layer]["gap_onset_marker_mass"])) for _, layer in SELECTED_LAYERS]
        top10_cells = [_fmt(float(values[layer]["gap_onset_top10_hit"])) for _, layer in SELECTED_LAYERS]

        rows.append(
            "        "
            + " & ".join(
                [
                    mode_label,
                    "Onset marker-mass gap",
                    *mass_cells,
                    _fmt(overall_mass),
                    f"{int(best_mass['layer_index'])} ({_fmt(float(best_mass['gap_onset_marker_mass']))})",
                ]
            )
            + r" \\"
        )
        rows.append(
            "        "
            + " & ".join(
                [
                    mode_label,
                    "Onset top-10 hit gap",
                    *top10_cells,
                    _fmt(overall_top10),
                    f"{int(best_top10['layer_index'])} ({_fmt(float(best_top10['gap_onset_top10_hit']))})",
                ]
            )
            + r" \\"
        )

    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{Prompt-only teacher-forcing prescriptive-marker summary. The probe is evaluated at the onset continuation position, i.e., the final prompt token position whose logits predict the first response token. A small prescriptive/advisory lexicon is used, including forms such as \texttt{{should}}, \texttt{{must}}, \texttt{{need}}, \texttt{{recommend}}, \texttt{{avoid}}, \texttt{{consider}}, and \texttt{{ensure}}. The onset marker-mass gap is computed as finetuned marker probability mass minus base marker probability mass at that position. The onset top-10 hit gap is computed as the finetuned indicator for whether at least one lexicon token appears in the top-10 predictions minus the corresponding base indicator, then averaged over prompts. Positive values therefore indicate a stronger finetuned tendency toward prescriptive/advisory continuations, negative values indicate a stronger base-model tendency, and near-zero values indicate little difference.}}
\label{{tab:tf-prescriptive-marker-summary}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llccccccc}}
\toprule
Lens & Metric & First & Early & Mid & Late & Last & Overall & Peak \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
"""


def build_findings_note(marker_summary_path: Path, generation_summary_path: Path) -> str:
    marker_obj = json.loads(marker_summary_path.read_text(encoding="utf-8"))
    gen_obj = json.loads(generation_summary_path.read_text(encoding="utf-8"))

    model_norm_peak = max(
        marker_obj["results"]["layerwise"]["model_norm"],
        key=lambda row: float(row["gap_onset_marker_mass"]),
    )
    raw_peak = max(
        marker_obj["results"]["layerwise"]["raw"],
        key=lambda row: float(row["gap_onset_marker_mass"]),
    )

    prompt_examples = []
    for row in marker_obj["results"]["prompt_rankings"]["model_norm"][:4]:
        prompt_text = (row.get("collection_text") or "").strip()
        if prompt_text:
            prompt_examples.append(prompt_text)

    gen_lines = []
    for summary in gen_obj["summaries"]:
        label = summary["template_label"]
        layers = summary["layers"]
        peak = max(layers, key=lambda row: float(row["top10_mean"]))
        overall = sum(float(row["top10_mean"]) for row in layers) / len(layers)
        gen_lines.append(
            f"{label} peaks near layer {int(peak['layer_absolute'])} with mean top-10 Jaccard {_fmt(float(peak['top10_mean']))}, "
            f"while its across-layer mean is {_fmt(overall)}."
        )

    examples_text = "; ".join(prompt_examples)
    gen_text = " ".join(gen_lines)

    return rf"""\paragraph{{How the prescriptive-marker probe is computed.}}
The prescriptive-marker analysis is a prompt-only teacher-forcing probe over the already collected TF payloads. For each prompt, each layer, and each lens type (\emph{{Raw LogitDiff Lens}} and \emph{{Model Norm LogitDiff Lens}}), the analysis reads the logits at the final prompt-token position, i.e., the position whose predictive distribution corresponds to the first response token. A small manually specified lexicon of prescriptive/advisory tokens is then used, including forms such as \texttt{{should}}, \texttt{{must}}, \texttt{{need}}, \texttt{{recommend}}, \texttt{{avoid}}, \texttt{{consider}}, and \texttt{{ensure}}. Two quantities are computed: (i) the total probability mass assigned to this lexicon, and (ii) whether at least one lexicon token appears in the top-10 predicted tokens. Reported gaps are always finetuned minus base, so positive values indicate a stronger finetuned tendency toward prescriptive/advisory continuations, negative values indicate a stronger base-model tendency, and values near zero indicate little difference. Because this probe is lexicon-based and context-light, it should be treated as a screening device rather than a standalone semantic claim.

\paragraph{{Cross-lens interpretation.}}
Across the generation runs, overlap is a late-layer phenomenon rather than an early-layer one. {gen_text} This suggests that late internal layers are more informative than the final readout layer alone when diagnosing whether the base and finetuned models are converging on the same continuation.

\paragraph{{Raw versus model-norm teacher forcing.}}
The prompt-only teacher-forcing comparisons suggest that the \emph{{Model Norm LogitDiff Lens}} is the cleaner descriptive probe, whereas the raw lens remains the more ecologically direct comparison to generation. In the prescriptive-marker analysis, the model-norm onset marker-mass gap peaks at layer {int(model_norm_peak['layer_index'])} with value {_fmt(float(model_norm_peak['gap_onset_marker_mass']))}, while the raw onset marker-mass gap peaks only weakly at layer {int(raw_peak['layer_index'])} with value {_fmt(float(raw_peak['gap_onset_marker_mass']))}. This is consistent with the broader TF summaries, where model-norm Jaccard aligns more cleanly with distributional divergence metrics such as total variation distance and Jensen--Shannon divergence.

\paragraph{{What the marker probe adds.}}
The lexicon-based marker probe should be treated as a screening device rather than a final semantic claim, but it highlights several prompts where the finetuned model appears more advice-oriented than the base model. The strongest model-norm cases in the current prompt-only set include: {examples_text}. Taken together, the TF probe and the generation lens support a cautious interpretation: the finetuned model often appears to develop a stronger late-layer tendency toward prescriptive or action-oriented continuations, and this tendency sometimes surfaces in generation as more direct advice.
"""


def main() -> None:
    root = Path("/media/am/AM/logit-diff-lens")
    marker_summary_path = root / "tmp/qwen_risky/tf_prompt_only/prescriptive_marker_analysis/prescriptive_marker_summary.json"
    generation_summary_path = root / "tmp/qwen_risky/logitdiff_gen/appendix_layer_summaries/generation_template_layerwise_jaccard_summary.json"
    out_dir = root / "tmp/qwen_risky/appendix_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "tf_prescriptive_marker_summary_table.tex").write_text(
        build_prescriptive_marker_table(marker_summary_path),
        encoding="utf-8",
    )
    (out_dir / "cross_lens_findings_summary.tex").write_text(
        build_findings_note(marker_summary_path, generation_summary_path),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

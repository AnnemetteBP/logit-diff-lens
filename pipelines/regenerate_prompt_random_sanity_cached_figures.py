from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch

from logit_diff_lens.plotting.prompt_null_sanity import (
    save_prompt_four_view_summary_pdf,
    save_prompt_jsd_calibration_gap_overlay_pdf,
    save_prompt_jsd_calibration_overlay_pdf,
    save_prompt_metric_actual_summary_pdf,
    save_prompt_metric_bundle_actual_pdf,
    save_prompt_metric_bundle_gap_pdf,
    save_prompt_metric_score_and_gap_pdf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate cache-backed prompt random-sanity PDFs.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _notice(path: Path) -> None:
    print(f"[write] {path}", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache = _load(output_dir / "90_prompt_random_sanity_summary_cache.pt")
    summaries_jsd = _load(output_dir / "91_jsd_layerwise_summaries.pt")
    gaps_jsd = _load(output_dir / "92_jsd_gap_summaries.pt")
    top1_ece_summaries = _load(output_dir / "93_top1_ece_layerwise_summaries.pt")
    top1_ece_gaps = _load(output_dir / "94_top1_ece_gap_summaries.pt")
    brier_summaries = _load(output_dir / "95_brier_layerwise_summaries.pt")
    brier_gaps = _load(output_dir / "96_brier_gap_summaries.pt")
    nll_summaries = _load(output_dir / "97_nll_layerwise_summaries.pt")
    nll_gaps = _load(output_dir / "98_nll_gap_summaries.pt")
    js_similarity_summaries = _load(output_dir / "101_js_similarity_layerwise_summaries.pt")
    js_similarity_gaps = _load(output_dir / "102_js_similarity_gap_summaries.pt")

    actual_family_labels = cache["actual_family_labels"]
    gap_family_labels = cache["gap_family_labels"]

    specs_actual = [
        ("20_non_calibrated_jsd_raw_vs_model_norm.pdf", summaries_jsd, "non_calibrated", "JSD", "Non-Calibrated JSD | Raw & ModelNorm LogitDiff"),
        ("21_calibrated_top1_ece_raw_vs_model_norm.pdf", top1_ece_summaries, "calibrated", "Top-1 ECE", "Calibrated Top-1 ECE | Raw & ModelNorm LogitDiff"),
        ("22_calibrated_brier_raw_vs_model_norm.pdf", brier_summaries, "calibrated", "Brier", "Calibrated Brier | Raw & ModelNorm LogitDiff"),
        ("23_calibrated_nll_raw_vs_model_norm.pdf", nll_summaries, "calibrated", "Negative Log-Likelihood", "Calibrated Negative Log-Likelihood | Raw & ModelNorm LogitDiff"),
        ("40_non_calibrated_js_similarity_raw_vs_model_norm.pdf", js_similarity_summaries, "non_calibrated", "JS Similarity", "Non-Calibrated JS Similarity | Raw & ModelNorm LogitDiff"),
        ("41_calibrated_js_similarity_raw_vs_model_norm.pdf", js_similarity_summaries, "calibrated", "JS Similarity", "Calibrated JS Similarity | Raw & ModelNorm LogitDiff"),
    ]
    for filename, summaries, metric_mode, label, title in specs_actual:
        path = output_dir / filename
        save_prompt_metric_actual_summary_pdf(
            summaries,
            path=path,
            metric_mode=metric_mode,
            metric_label_override=label,
            family_labels=actual_family_labels,
            title=title,
        )
        _notice(path)

    specs_score_gap = [
        ("10_calibrated_top1_ece_raw_vs_model_norm.pdf", top1_ece_summaries, top1_ece_gaps, "calibrated", "Top-1 ECE", "Calibrated Top-1 ECE | Raw & ModelNorm LogitDiff"),
        ("11_calibrated_brier_raw_vs_model_norm.pdf", brier_summaries, brier_gaps, "calibrated", "Brier", "Calibrated Brier | Raw & ModelNorm LogitDiff"),
        ("12_calibrated_nll_raw_vs_model_norm.pdf", nll_summaries, nll_gaps, "calibrated", "Negative Log-Likelihood", "Calibrated Negative Log-Likelihood | Raw & ModelNorm LogitDiff"),
        ("42_non_calibrated_js_similarity_score_and_gap_raw_vs_model_norm.pdf", js_similarity_summaries, js_similarity_gaps, "non_calibrated", "JS Similarity", "Non-Calibrated JS Similarity | Raw & ModelNorm LogitDiff"),
        ("43_calibrated_js_similarity_score_and_gap_raw_vs_model_norm.pdf", js_similarity_summaries, js_similarity_gaps, "calibrated", "JS Similarity", "Calibrated JS Similarity | Raw & ModelNorm LogitDiff"),
    ]
    for filename, summaries, gaps, metric_mode, label, title in specs_score_gap:
        path = output_dir / filename
        save_prompt_metric_score_and_gap_pdf(
            summaries,
            gaps,
            path=path,
            metric_mode=metric_mode,
            metric_label_override=label,
            actual_family_labels=actual_family_labels,
            gap_family_labels=gap_family_labels,
            title=title,
        )
        _notice(path)

    path = output_dir / "30_non_calibrated_and_calibrated_raw_vs_model_norm.pdf"
    save_prompt_metric_bundle_actual_pdf(
        [
            ("non_calibrated", summaries_jsd, "JSD"),
            ("calibrated", top1_ece_summaries, "Top-1 ECE"),
            ("calibrated", brier_summaries, "Brier"),
            ("calibrated", nll_summaries, "Negative Log-Likelihood"),
        ],
        path=path,
        family_labels=actual_family_labels,
        title="Non-Calibrated JSD & Calibrated Top-1 ECE, Brier, & Negative Log-Likelihood | Raw & ModelNorm LogitDiff",
    )
    _notice(path)

    path = output_dir / "31_gap_metric_bundle_raw_vs_model_norm.pdf"
    save_prompt_metric_bundle_gap_pdf(
        [
            ("non_calibrated", gaps_jsd, "JSD"),
            ("calibrated", top1_ece_gaps, "Top-1 ECE"),
            ("calibrated", brier_gaps, "Brier"),
            ("calibrated", nll_gaps, "Negative Log-Likelihood"),
        ],
        path=path,
        family_labels=gap_family_labels,
        title="Non-Calibrated ΔJSD & Calibrated ΔTop-1 ECE, ΔBrier, & ΔNegative Log-Likelihood | Raw & ModelNorm LogitDiff",
    )
    _notice(path)

    overlay_specs = [
        ("32_overlay_jsd_vs_top1_ece_raw_modelnorm.pdf", top1_ece_summaries, "Top-1 ECE"),
        ("33_overlay_jsd_vs_brier_raw_modelnorm.pdf", brier_summaries, "Brier"),
        ("34_overlay_jsd_vs_nll_raw_modelnorm.pdf", nll_summaries, "Negative Log-Likelihood"),
    ]
    for filename, calibrated_summaries, label in overlay_specs:
        path = output_dir / filename
        save_prompt_jsd_calibration_overlay_pdf(
            summaries_jsd,
            calibrated_summaries,
            path=path,
            calibrated_label=label,
            family_titles_override=actual_family_labels,
            title=f"Non-Calibrated JSD & Calibrated {label} | Raw & ModelNorm LogitDiff",
        )
        _notice(path)

    gap_overlay_specs = [
        ("35_overlay_gap_jsd_vs_top1_ece_raw_modelnorm.pdf", top1_ece_gaps, "Top-1 ECE"),
        ("36_overlay_gap_jsd_vs_brier_raw_modelnorm.pdf", brier_gaps, "Brier"),
        ("37_overlay_gap_jsd_vs_nll_raw_modelnorm.pdf", nll_gaps, "Negative Log-Likelihood"),
    ]
    for filename, calibrated_gaps, label in gap_overlay_specs:
        path = output_dir / filename
        save_prompt_jsd_calibration_gap_overlay_pdf(
            gaps_jsd,
            calibrated_gaps,
            path=path,
            calibrated_label=label,
            family_titles_override=gap_family_labels,
            title=f"Non-Calibrated ΔJSD & Calibrated Δ{label} | Raw & ModelNorm LogitDiff",
        )
        _notice(path)

    path = output_dir / "38_four_view_jsd_vs_top1_ece.pdf"
    save_prompt_four_view_summary_pdf(
        summaries_jsd,
        top1_ece_summaries,
        path=path,
        calibrated_label="Top-1 ECE",
        family_labels=actual_family_labels,
        title="Non-Calibrated JSD & Calibrated Top-1 ECE | Raw & ModelNorm LogitDiff",
    )
    _notice(path)

    path = output_dir / "44_non_calibrated_and_calibrated_js_similarity_four_view.pdf"
    save_prompt_four_view_summary_pdf(
        js_similarity_summaries,
        js_similarity_summaries,
        path=path,
        calibrated_label="JS Similarity",
        family_labels=actual_family_labels,
        title="Non-Calibrated & Calibrated JS Similarity | Raw & ModelNorm LogitDiff",
    )
    _notice(path)

    path = output_dir / "45_non_calibrated_and_calibrated_js_similarity_raw_vs_model_norm.pdf"
    save_prompt_metric_bundle_actual_pdf(
        [
            ("non_calibrated", js_similarity_summaries, "JS Similarity"),
            ("calibrated", js_similarity_summaries, "JS Similarity"),
        ],
        path=path,
        family_labels=actual_family_labels,
        title="Non-Calibrated & Calibrated JS Similarity | Raw & ModelNorm LogitDiff",
    )
    _notice(path)

    path = output_dir / "46_non_calibrated_and_calibrated_js_similarity_gap_raw_vs_model_norm.pdf"
    save_prompt_metric_bundle_gap_pdf(
        [
            ("non_calibrated", js_similarity_gaps, "JS Similarity"),
            ("calibrated", js_similarity_gaps, "JS Similarity"),
        ],
        path=path,
        family_labels=gap_family_labels,
        title="Non-Calibrated & Calibrated ΔJS Similarity | Raw & ModelNorm LogitDiff",
    )
    _notice(path)


if __name__ == "__main__":
    main()

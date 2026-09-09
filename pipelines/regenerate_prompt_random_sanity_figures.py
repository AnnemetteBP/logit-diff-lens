from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import torch

from logit_diff_lens.plotting.prompt_null_sanity import (
    build_prompt_pairwise_jsd_curves,
    build_prompt_pairwise_jsd_gap_curves,
    build_prompt_pairwise_metric_curves,
    build_prompt_pairwise_metric_gap_curves,
    build_prompt_pairwise_similarity_curves,
    build_prompt_pairwise_similarity_gap_curves,
    save_combined_pairwise_jsd_gap_pdf,
    save_combined_pairwise_jsd_pdf,
    save_combined_pairwise_metric_gap_pdf,
    save_combined_pairwise_metric_pdf,
    save_prompt_four_view_summary_pdf,
    save_prompt_jsd_calibration_gap_overlay_pdf,
    save_prompt_jsd_calibration_overlay_pdf,
    save_prompt_metric_actual_summary_pdf,
    save_prompt_metric_bundle_actual_pdf,
    save_prompt_metric_bundle_gap_pdf,
    save_prompt_metric_score_and_gap_pdf,
    save_robust_prompt_jsd_overview_pdf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate prompt random-sanity PDFs from saved caches.")
    parser.add_argument("--trained-artifact", required=True)
    parser.add_argument("--baseline-artifact", required=True)
    parser.add_argument("--random-artifact", action="append", dest="random_artifacts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_notice(path: Path) -> None:
    print(f"[write] {path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.size": 21,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 21,
            "figure.titlesize": 26,
        }
    )

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
    common_kwargs = dict(
        trained_artifact=args.trained_artifact,
        baseline_artifact=args.baseline_artifact,
        random_artifacts=list(args.random_artifacts),
        top_k=args.top_k,
    )

    raw_curves, raw_labels = build_prompt_pairwise_jsd_curves(readout_mode="raw", **common_kwargs)
    model_norm_curves, model_norm_labels = build_prompt_pairwise_jsd_curves(readout_mode="model_norm", **common_kwargs)
    raw_gap_curves, _ = build_prompt_pairwise_jsd_gap_curves(readout_mode="raw", **common_kwargs)
    model_norm_gap_curves, _ = build_prompt_pairwise_jsd_gap_curves(readout_mode="model_norm", **common_kwargs)

    path = output_dir / "00_jsd_overview_4panel.pdf"
    save_robust_prompt_jsd_overview_pdf(
        raw_curves=raw_curves,
        raw_layer_labels=raw_labels,
        raw_gap_curves=raw_gap_curves,
        model_norm_curves=model_norm_curves,
        model_norm_layer_labels=model_norm_labels,
        model_norm_gap_curves=model_norm_gap_curves,
        path=path,
        title="Non-Calibrated JSD | Raw & ModelNorm LogitDiff",
    )
    _save_notice(path)

    path = output_dir / "01_non_calibrated_jsd_pairwise_raw_vs_model_norm.pdf"
    save_combined_pairwise_jsd_pdf(
        raw_curves=raw_curves,
        raw_layer_labels=raw_labels,
        model_norm_curves=model_norm_curves,
        model_norm_layer_labels=model_norm_labels,
        path=path,
        title="Non-Calibrated JSD | Raw & ModelNorm LogitDiff",
    )
    _save_notice(path)

    path = output_dir / "02_jsd_gap_vs_random_null_raw_vs_model_norm.pdf"
    save_combined_pairwise_jsd_gap_pdf(
        raw_gap_curves=raw_gap_curves,
        raw_layer_labels=raw_labels,
        model_norm_gap_curves=model_norm_gap_curves,
        model_norm_layer_labels=model_norm_labels,
        path=path,
        title="Non-Calibrated ΔJSD | Raw & ModelNorm LogitDiff",
    )
    _save_notice(path)

    pairwise_metric_specs = [
        ("top1_ece", "Top-1 ECE", "03_calibrated_top1_ece_pairwise_raw_vs_model_norm.pdf", "06_calibrated_top1_ece_gap_pairwise_raw_vs_model_norm.pdf"),
        ("brier", "Brier", "04_calibrated_brier_pairwise_raw_vs_model_norm.pdf", "07_calibrated_brier_gap_pairwise_raw_vs_model_norm.pdf"),
        ("nll", "Negative Log-Likelihood", "05_calibrated_nll_pairwise_raw_vs_model_norm.pdf", "08_calibrated_nll_gap_pairwise_raw_vs_model_norm.pdf"),
    ]
    for metric_name, metric_label, pairwise_name, pairwise_gap_name in pairwise_metric_specs:
        raw_metric_curves, raw_metric_labels = build_prompt_pairwise_metric_curves(
            readout_mode="raw",
            metric_mode="calibrated",
            calibrated_metric_name=metric_name,
            **common_kwargs,
        )
        model_norm_metric_curves, model_norm_metric_labels = build_prompt_pairwise_metric_curves(
            readout_mode="model_norm",
            metric_mode="calibrated",
            calibrated_metric_name=metric_name,
            **common_kwargs,
        )
        path = output_dir / pairwise_name
        save_combined_pairwise_metric_pdf(
            raw_curves=raw_metric_curves,
            raw_layer_labels=raw_metric_labels,
            model_norm_curves=model_norm_metric_curves,
            model_norm_layer_labels=model_norm_metric_labels,
            path=path,
            metric_label=f"Calibrated {metric_label}",
            y_label=f"Mean {metric_label}",
            title=f"Calibrated {metric_label} | Raw & ModelNorm LogitDiff | Individual Seed/Pair Comparisons",
        )
        _save_notice(path)

        raw_metric_gap_curves, raw_metric_gap_labels = build_prompt_pairwise_metric_gap_curves(
            readout_mode="raw",
            metric_mode="calibrated",
            calibrated_metric_name=metric_name,
            **common_kwargs,
        )
        model_norm_metric_gap_curves, model_norm_metric_gap_labels = build_prompt_pairwise_metric_gap_curves(
            readout_mode="model_norm",
            metric_mode="calibrated",
            calibrated_metric_name=metric_name,
            **common_kwargs,
        )
        path = output_dir / pairwise_gap_name
        save_combined_pairwise_metric_gap_pdf(
            raw_gap_curves=raw_metric_gap_curves,
            raw_layer_labels=raw_metric_gap_labels,
            model_norm_gap_curves=model_norm_metric_gap_curves,
            model_norm_layer_labels=model_norm_metric_gap_labels,
            path=path,
            metric_label=f"Calibrated {metric_label}",
            y_label=f"Δ{metric_label}",
            title=f"Calibrated Δ{metric_label} | Raw & ModelNorm LogitDiff | Individual Seed/Pair Comparisons",
        )
        _save_notice(path)

    single_actual_specs = [
        ("20_non_calibrated_jsd_raw_vs_model_norm.pdf", summaries_jsd, "non_calibrated", "JSD", "Non-Calibrated JSD | Raw & ModelNorm LogitDiff"),
        ("21_calibrated_top1_ece_raw_vs_model_norm.pdf", top1_ece_summaries, "calibrated", "Top-1 ECE", "Calibrated Top-1 ECE | Raw & ModelNorm LogitDiff"),
        ("22_calibrated_brier_raw_vs_model_norm.pdf", brier_summaries, "calibrated", "Brier", "Calibrated Brier | Raw & ModelNorm LogitDiff"),
        ("23_calibrated_nll_raw_vs_model_norm.pdf", nll_summaries, "calibrated", "Negative Log-Likelihood", "Calibrated Negative Log-Likelihood | Raw & ModelNorm LogitDiff"),
        ("40_non_calibrated_js_similarity_raw_vs_model_norm.pdf", js_similarity_summaries, "non_calibrated", "JS Similarity", "Non-Calibrated JS Similarity | Raw & ModelNorm LogitDiff"),
        ("41_calibrated_js_similarity_raw_vs_model_norm.pdf", js_similarity_summaries, "calibrated", "JS Similarity", "Calibrated JS Similarity | Raw & ModelNorm LogitDiff"),
    ]
    for filename, summaries, metric_mode, label, title in single_actual_specs:
        path = output_dir / filename
        save_prompt_metric_actual_summary_pdf(
            summaries,
            path=path,
            metric_mode=metric_mode,
            metric_label_override=label,
            family_labels=actual_family_labels,
            title=title,
        )
        _save_notice(path)

    score_gap_specs = [
        ("10_calibrated_top1_ece_raw_vs_model_norm.pdf", top1_ece_summaries, top1_ece_gaps, "calibrated", "Top-1 ECE", "Calibrated Top-1 ECE | Raw & ModelNorm LogitDiff"),
        ("11_calibrated_brier_raw_vs_model_norm.pdf", brier_summaries, brier_gaps, "calibrated", "Brier", "Calibrated Brier | Raw & ModelNorm LogitDiff"),
        ("12_calibrated_nll_raw_vs_model_norm.pdf", nll_summaries, nll_gaps, "calibrated", "Negative Log-Likelihood", "Calibrated Negative Log-Likelihood | Raw & ModelNorm LogitDiff"),
        ("42_non_calibrated_js_similarity_score_and_gap_raw_vs_model_norm.pdf", js_similarity_summaries, js_similarity_gaps, "non_calibrated", "JS Similarity", "Non-Calibrated JS Similarity | Raw & ModelNorm LogitDiff"),
        ("43_calibrated_js_similarity_score_and_gap_raw_vs_model_norm.pdf", js_similarity_summaries, js_similarity_gaps, "calibrated", "JS Similarity", "Calibrated JS Similarity | Raw & ModelNorm LogitDiff"),
    ]
    for filename, summaries, gaps, metric_mode, label, title in score_gap_specs:
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
        _save_notice(path)

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
    _save_notice(path)

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
    _save_notice(path)

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
        _save_notice(path)

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
        _save_notice(path)

    path = output_dir / "38_four_view_jsd_vs_top1_ece.pdf"
    save_prompt_four_view_summary_pdf(
        summaries_jsd,
        top1_ece_summaries,
        path=path,
        calibrated_label="Top-1 ECE",
        family_labels=actual_family_labels,
        title="Non-Calibrated JSD & Calibrated Top-1 ECE | Raw & ModelNorm LogitDiff",
    )
    _save_notice(path)

    path = output_dir / "44_non_calibrated_and_calibrated_js_similarity_four_view.pdf"
    save_prompt_four_view_summary_pdf(
        js_similarity_summaries,
        js_similarity_summaries,
        path=path,
        calibrated_label="JS Similarity",
        family_labels=actual_family_labels,
        title="Non-Calibrated & Calibrated JS Similarity | Raw & ModelNorm LogitDiff",
    )
    _save_notice(path)

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
    _save_notice(path)

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
    _save_notice(path)

    for calibrated, stem, title_prefix, seed_base in [
        (False, "47_non_calibrated", "Non-Calibrated JS Similarity", 21),
        (True, "49_calibrated", "Calibrated JS Similarity", 25),
    ]:
        raw_curves, raw_labels = build_prompt_pairwise_similarity_curves(
            trained_artifact=args.trained_artifact,
            baseline_artifact=args.baseline_artifact,
            random_artifacts=list(args.random_artifacts),
            readout_mode="raw",
            similarity_metric_name="js_similarity",
            calibrated=calibrated,
            top_k=args.top_k,
            num_permutations=1000,
            alpha=0.05,
            seed=seed_base,
        )
        model_norm_curves, model_norm_labels = build_prompt_pairwise_similarity_curves(
            trained_artifact=args.trained_artifact,
            baseline_artifact=args.baseline_artifact,
            random_artifacts=list(args.random_artifacts),
            readout_mode="model_norm",
            similarity_metric_name="js_similarity",
            calibrated=calibrated,
            top_k=args.top_k,
            num_permutations=1000,
            alpha=0.05,
            seed=seed_base + 1,
        )
        path = output_dir / f"{stem}_js_similarity_pairwise_raw_vs_model_norm.pdf"
        save_combined_pairwise_metric_pdf(
            raw_curves=raw_curves,
            raw_layer_labels=raw_labels,
            model_norm_curves=model_norm_curves,
            model_norm_layer_labels=model_norm_labels,
            path=path,
            metric_label=title_prefix,
            y_label=title_prefix,
            title=f"{title_prefix} | Raw & ModelNorm LogitDiff | Individual Seed/Pair Comparisons",
        )
        _save_notice(path)

        raw_gap_curves, raw_gap_labels = build_prompt_pairwise_similarity_gap_curves(
            trained_artifact=args.trained_artifact,
            baseline_artifact=args.baseline_artifact,
            random_artifacts=list(args.random_artifacts),
            readout_mode="raw",
            similarity_metric_name="js_similarity",
            calibrated=calibrated,
            top_k=args.top_k,
            num_permutations=1000,
            alpha=0.05,
            seed=seed_base + 2,
        )
        model_norm_gap_curves, model_norm_gap_labels = build_prompt_pairwise_similarity_gap_curves(
            trained_artifact=args.trained_artifact,
            baseline_artifact=args.baseline_artifact,
            random_artifacts=list(args.random_artifacts),
            readout_mode="model_norm",
            similarity_metric_name="js_similarity",
            calibrated=calibrated,
            top_k=args.top_k,
            num_permutations=1000,
            alpha=0.05,
            seed=seed_base + 3,
        )
        path = output_dir / f"{int(stem[:2]) + 1:02d}_{'non_calibrated' if not calibrated else 'calibrated'}_js_similarity_gap_pairwise_raw_vs_model_norm.pdf"
        save_combined_pairwise_metric_gap_pdf(
            raw_gap_curves=raw_gap_curves,
            raw_layer_labels=raw_gap_labels,
            model_norm_gap_curves=model_norm_gap_curves,
            model_norm_layer_labels=model_norm_gap_labels,
            path=path,
            metric_label=title_prefix,
            y_label=f"Δ{title_prefix}",
            title=f"Δ{title_prefix} | Raw & ModelNorm LogitDiff | Individual Seed/Pair Comparisons",
        )
        _save_notice(path)


if __name__ == "__main__":
    main()

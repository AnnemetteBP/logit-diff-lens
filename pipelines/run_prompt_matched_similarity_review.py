from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import torch

from logit_diff_lens.plotting.prompt_null_sanity import (
    build_prompt_null_gap_summaries,
    build_prompt_pairwise_similarity_curves,
    build_prompt_pairwise_similarity_gap_curves,
    build_prompt_similarity_summaries,
    save_combined_pairwise_metric_gap_pdf,
    save_combined_pairwise_metric_pdf,
    save_prompt_four_view_summary_pdf,
    save_prompt_metric_actual_summary_pdf,
    save_prompt_metric_bundle_actual_pdf,
    save_prompt_metric_bundle_gap_pdf,
    save_prompt_metric_score_and_gap_pdf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate matched non-calibrated/calibrated prompt LogitDiff similarity PDFs."
    )
    parser.add_argument("--trained-artifact", required=True)
    parser.add_argument("--baseline-artifact", required=True)
    parser.add_argument("--random-artifact", action="append", dest="random_artifacts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-permutations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    family_labels = {
        "trained_vs_trained": "70m-143k vs 70m-71k",
        "trained_vs_random": f"70m-143k vs random-seed mean (n={len(args.random_artifacts)})",
        "random_vs_random": "random-seed-pair mean (n=3)",
    }
    gap_family_labels = {
        "trained_vs_trained_minus_random_vs_random": "70m-143k vs 70m-71k - random-seed-pair mean (n=3)",
        "trained_vs_random_minus_random_vs_random": "70m-143k vs random-seed mean (n=3) - random-seed-pair mean (n=3)",
    }

    summaries = build_prompt_similarity_summaries(
        trained_artifact=args.trained_artifact,
        baseline_artifact=args.baseline_artifact,
        random_artifacts=list(args.random_artifacts),
        similarity_metric_name="js_similarity",
        top_k=args.top_k,
        num_permutations=args.num_permutations,
        alpha=args.alpha,
        seed=11,
    )
    gaps = build_prompt_null_gap_summaries(summaries)
    torch.save(summaries, output_dir / "101_js_similarity_layerwise_summaries.pt")
    torch.save(gaps, output_dir / "102_js_similarity_gap_summaries.pt")

    save_prompt_metric_actual_summary_pdf(
        summaries,
        path=output_dir / "40_non_calibrated_js_similarity_raw_vs_model_norm.pdf",
        metric_mode="non_calibrated",
        metric_label_override="JS Similarity",
        family_labels=family_labels,
        title="Non-Calibrated JS Similarity | Raw & ModelNorm LogitDiff",
    )
    save_prompt_metric_actual_summary_pdf(
        summaries,
        path=output_dir / "41_calibrated_js_similarity_raw_vs_model_norm.pdf",
        metric_mode="calibrated",
        metric_label_override="JS Similarity",
        family_labels=family_labels,
        title="Calibrated JS Similarity | Raw & ModelNorm LogitDiff",
    )
    save_prompt_metric_score_and_gap_pdf(
        summaries,
        gaps,
        path=output_dir / "42_non_calibrated_js_similarity_score_and_gap_raw_vs_model_norm.pdf",
        metric_mode="non_calibrated",
        metric_label_override="JS Similarity",
        actual_family_labels=family_labels,
        gap_family_labels=gap_family_labels,
        title="Non-Calibrated JS Similarity | Raw & ModelNorm LogitDiff",
    )
    save_prompt_metric_score_and_gap_pdf(
        summaries,
        gaps,
        path=output_dir / "43_calibrated_js_similarity_score_and_gap_raw_vs_model_norm.pdf",
        metric_mode="calibrated",
        metric_label_override="JS Similarity",
        actual_family_labels=family_labels,
        gap_family_labels=gap_family_labels,
        title="Calibrated JS Similarity | Raw & ModelNorm LogitDiff",
    )
    save_prompt_four_view_summary_pdf(
        summaries,
        summaries,
        path=output_dir / "44_non_calibrated_and_calibrated_js_similarity_four_view.pdf",
        calibrated_label="JS Similarity",
        family_labels=family_labels,
        title="Non-Calibrated & Calibrated JS Similarity | Raw & ModelNorm LogitDiff",
    )
    save_prompt_metric_bundle_actual_pdf(
        [
            ("non_calibrated", summaries, "JS Similarity"),
            ("calibrated", summaries, "JS Similarity"),
        ],
        path=output_dir / "45_non_calibrated_and_calibrated_js_similarity_raw_vs_model_norm.pdf",
        family_labels=family_labels,
        title="Non-Calibrated & Calibrated JS Similarity | Raw & ModelNorm LogitDiff",
    )
    save_prompt_metric_bundle_gap_pdf(
        [
            ("non_calibrated", gaps, "JS Similarity"),
            ("calibrated", gaps, "JS Similarity"),
        ],
        path=output_dir / "46_non_calibrated_and_calibrated_js_similarity_gap_raw_vs_model_norm.pdf",
        family_labels=gap_family_labels,
        title="Non-Calibrated & Calibrated JS Similarity Gap | Raw & ModelNorm LogitDiff",
    )

    for calibrated, stem, title_prefix in [
        (False, "47_non_calibrated", "Non-Calibrated JS Similarity"),
        (True, "49_calibrated", "Calibrated JS Similarity"),
    ]:
        raw_curves, raw_labels = build_prompt_pairwise_similarity_curves(
            trained_artifact=args.trained_artifact,
            baseline_artifact=args.baseline_artifact,
            random_artifacts=list(args.random_artifacts),
            readout_mode="raw",
            similarity_metric_name="js_similarity",
            calibrated=calibrated,
            top_k=args.top_k,
            num_permutations=args.num_permutations,
            alpha=args.alpha,
            seed=21 if not calibrated else 25,
        )
        model_norm_curves, model_norm_labels = build_prompt_pairwise_similarity_curves(
            trained_artifact=args.trained_artifact,
            baseline_artifact=args.baseline_artifact,
            random_artifacts=list(args.random_artifacts),
            readout_mode="model_norm",
            similarity_metric_name="js_similarity",
            calibrated=calibrated,
            top_k=args.top_k,
            num_permutations=args.num_permutations,
            alpha=args.alpha,
            seed=22 if not calibrated else 26,
        )
        save_combined_pairwise_metric_pdf(
            raw_curves=raw_curves,
            raw_layer_labels=raw_labels,
            model_norm_curves=model_norm_curves,
            model_norm_layer_labels=model_norm_labels,
            path=output_dir / f"{stem}_js_similarity_pairwise_raw_vs_model_norm.pdf",
            metric_label=title_prefix,
            y_label=f"Mean {title_prefix.lower()}",
            title=f"{title_prefix} | Raw & ModelNorm LogitDiff | Individual Seed/Pair Comparisons",
        )

        raw_gap_curves, raw_gap_labels = build_prompt_pairwise_similarity_gap_curves(
            trained_artifact=args.trained_artifact,
            baseline_artifact=args.baseline_artifact,
            random_artifacts=list(args.random_artifacts),
            readout_mode="raw",
            similarity_metric_name="js_similarity",
            calibrated=calibrated,
            top_k=args.top_k,
            num_permutations=args.num_permutations,
            alpha=args.alpha,
            seed=23 if not calibrated else 27,
        )
        model_norm_gap_curves, model_norm_gap_labels = build_prompt_pairwise_similarity_gap_curves(
            trained_artifact=args.trained_artifact,
            baseline_artifact=args.baseline_artifact,
            random_artifacts=list(args.random_artifacts),
            readout_mode="model_norm",
            similarity_metric_name="js_similarity",
            calibrated=calibrated,
            top_k=args.top_k,
            num_permutations=args.num_permutations,
            alpha=args.alpha,
            seed=24 if not calibrated else 28,
        )
        save_combined_pairwise_metric_gap_pdf(
            raw_gap_curves=raw_gap_curves,
            raw_layer_labels=raw_gap_labels,
            model_norm_gap_curves=model_norm_gap_curves,
            model_norm_layer_labels=model_norm_gap_labels,
            path=output_dir / f"{int(stem[:2]) + 1:02d}_{'non_calibrated' if not calibrated else 'calibrated'}_js_similarity_gap_pairwise_raw_vs_model_norm.pdf",
            metric_label=title_prefix,
            y_label=f"{title_prefix} gap",
            title=f"{title_prefix} Gap | Raw & ModelNorm LogitDiff | Individual Seed/Pair Comparisons",
        )


if __name__ == "__main__":
    main()

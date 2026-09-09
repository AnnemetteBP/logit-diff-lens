from __future__ import annotations

import argparse
import json
import matplotlib as mpl
from pathlib import Path
import torch
import re

from logit_diff_lens.plotting.prompt_null_sanity import (
    build_prompt_hidden_metric_summaries,
    build_prompt_null_gap_summaries,
    build_prompt_null_sanity_summaries,
    build_prompt_pairwise_jsd_curves,
    build_prompt_pairwise_jsd_gap_curves,
    build_prompt_pairwise_metric_curves,
    build_prompt_pairwise_metric_gap_curves,
    build_prompt_pairwise_similarity_curves,
    build_prompt_pairwise_similarity_gap_curves,
    build_prompt_similarity_summaries,
    save_combined_pairwise_jsd_gap_pdf,
    save_combined_pairwise_jsd_pdf,
    save_combined_pairwise_metric_gap_pdf,
    save_combined_pairwise_metric_pdf,
    save_prompt_metric_actual_summary_pdf,
    save_prompt_metric_bundle_actual_pdf,
    save_prompt_metric_bundle_gap_pdf,
    save_prompt_jsd_calibration_overlay_pdf,
    save_prompt_jsd_calibration_gap_overlay_pdf,
    save_prompt_four_view_summary_pdf,
    save_prompt_metric_score_and_gap_pdf,
    save_robust_prompt_jsd_overview_pdf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prompt random-sanity comparison PDFs from saved prompt bundles."
    )
    parser.add_argument("--trained-artifact", required=True, help="Saved trained prompt bundle (.pt).")
    parser.add_argument("--baseline-artifact", required=True, help="Saved baseline prompt bundle (.pt).")
    parser.add_argument(
        "--random-artifact",
        action="append",
        dest="random_artifacts",
        required=True,
        help="Saved random-seed prompt bundle (.pt). Repeat for each random seed.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated PDFs.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for non-calibrated JSD comparisons.")
    parser.add_argument(
        "--artifact-label-map",
        help="Optional JSON mapping artifact filename or stem to display label.",
    )
    return parser.parse_args()


def _short_label(path: str) -> str:
    stem = Path(path).stem.replace("_prompt_bundle", "")
    size_match = re.search(r"pythia_(\d+[a-z]+)", stem)
    size_label = size_match.group(1) if size_match is not None else "model"
    step_match = re.search(r"step(\d+)", stem)
    if step_match is not None:
        step_value = int(step_match.group(1))
        ckpt = f"{step_value // 1000}k" if step_value % 1000 == 0 else str(step_value)
        return f"{size_label}-{ckpt}"
    seed_match = re.search(r"seed(\d+)", stem)
    if seed_match is not None:
        return f"{size_label}-seed{seed_match.group(1)}"
    return stem


def _load_artifact_label_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("artifact-label-map must be a JSON object")
    return {str(key): str(value) for key, value in payload.items()}


def _artifact_display_label(path: str, label_map: dict[str, str]) -> str:
    artifact_path = Path(path)
    for key in (artifact_path.name, artifact_path.stem):
        if key in label_map:
            return label_map[key]
    return _short_label(path)


def _build_family_labels(
    trained_artifact: str,
    baseline_artifact: str,
    random_artifacts: list[str],
    *,
    artifact_label_map: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    label_map = artifact_label_map or {}
    trained_label = _artifact_display_label(trained_artifact, label_map)
    baseline_label = _artifact_display_label(baseline_artifact, label_map)
    size_prefix = trained_label.split("-", 1)[0] if "-" in trained_label else trained_label
    random_count = len(random_artifacts)
    pair_count = (random_count * (random_count - 1)) // 2
    random_mean_label = f"{trained_label} vs random-seed mean (n={random_count})"
    random_pair_mean_label = f"random-seed-pair mean (n={pair_count})"
    actual_labels = {
        "trained_vs_trained": f"{trained_label} vs {baseline_label}",
        "trained_vs_random": random_mean_label,
        "random_vs_random": random_pair_mean_label,
    }
    gap_labels = {
        "trained_vs_trained_minus_random_vs_random": f"{trained_label} vs {baseline_label} - {random_pair_mean_label}",
        "trained_vs_random_minus_random_vs_random": f"{random_mean_label} - {random_pair_mean_label}",
    }
    return actual_labels, gap_labels


def _load_cached(path: Path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


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
    artifact_label_map = _load_artifact_label_map(args.artifact_label_map)

    common_kwargs = dict(
        trained_artifact=args.trained_artifact,
        baseline_artifact=args.baseline_artifact,
        random_artifacts=list(args.random_artifacts),
        top_k=args.top_k,
    )
    actual_family_labels, gap_family_labels = _build_family_labels(
        args.trained_artifact,
        args.baseline_artifact,
        list(args.random_artifacts),
        artifact_label_map=artifact_label_map,
    )

    raw_curves, raw_labels = build_prompt_pairwise_jsd_curves(readout_mode="raw", **common_kwargs)
    model_norm_curves, model_norm_labels = build_prompt_pairwise_jsd_curves(
        readout_mode="model_norm", **common_kwargs
    )
    raw_gap_curves, _ = build_prompt_pairwise_jsd_gap_curves(readout_mode="raw", **common_kwargs)
    model_norm_gap_curves, _ = build_prompt_pairwise_jsd_gap_curves(
        readout_mode="model_norm", **common_kwargs
    )

    save_robust_prompt_jsd_overview_pdf(
        raw_curves=raw_curves,
        raw_layer_labels=raw_labels,
        raw_gap_curves=raw_gap_curves,
        model_norm_curves=model_norm_curves,
        model_norm_layer_labels=model_norm_labels,
        model_norm_gap_curves=model_norm_gap_curves,
        path=output_dir / "00_jsd_overview_4panel.pdf",
        title="Non-Calibrated JSD | Raw & ModelNorm LogitDiff",
    )
    save_combined_pairwise_jsd_pdf(
        raw_curves=raw_curves,
        raw_layer_labels=raw_labels,
        model_norm_curves=model_norm_curves,
        model_norm_layer_labels=model_norm_labels,
        path=output_dir / "01_non_calibrated_jsd_pairwise_raw_vs_model_norm.pdf",
        title="Non-Calibrated JSD | Raw & ModelNorm LogitDiff",
    )
    save_combined_pairwise_jsd_gap_pdf(
        raw_gap_curves=raw_gap_curves,
        raw_layer_labels=raw_labels,
        model_norm_gap_curves=model_norm_gap_curves,
        model_norm_layer_labels=model_norm_labels,
        path=output_dir / "02_jsd_gap_vs_random_null_raw_vs_model_norm.pdf",
        title="Non-Calibrated JSD Gap | Raw & ModelNorm LogitDiff",
    )

    pairwise_metric_specs = [
        ("top1_ece", "Top-1 ECE", "03_calibrated_top1_ece_pairwise_raw_vs_model_norm.pdf", "06_calibrated_top1_ece_gap_pairwise_raw_vs_model_norm.pdf"),
        ("brier", "Brier", "04_calibrated_brier_pairwise_raw_vs_model_norm.pdf", "07_calibrated_brier_gap_pairwise_raw_vs_model_norm.pdf"),
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
        save_combined_pairwise_metric_pdf(
            raw_curves=raw_metric_curves,
            raw_layer_labels=raw_metric_labels,
            model_norm_curves=model_norm_metric_curves,
            model_norm_layer_labels=model_norm_metric_labels,
            path=output_dir / pairwise_name,
            metric_label=f"Calibrated {metric_label}",
            y_label=f"Mean {metric_label}",
            title=f"Calibrated {metric_label} | Raw & ModelNorm LogitDiff | Individual Seed/Pair Comparisons",
        )

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
        save_combined_pairwise_metric_gap_pdf(
            raw_gap_curves=raw_metric_gap_curves,
            raw_layer_labels=raw_metric_gap_labels,
            model_norm_gap_curves=model_norm_metric_gap_curves,
            model_norm_layer_labels=model_norm_metric_gap_labels,
            path=output_dir / pairwise_gap_name,
            metric_label=f"Calibrated {metric_label}",
            y_label=f"{metric_label} gap",
            title=f"Calibrated {metric_label} Gap | Raw & ModelNorm LogitDiff | Individual Seed/Pair Comparisons",
        )

    summaries_jsd = _load_cached(output_dir / "91_jsd_layerwise_summaries.pt")
    if summaries_jsd is None:
        summaries_jsd = build_prompt_null_sanity_summaries(calibrated_metric_name="top1_ece", **common_kwargs)
        torch.save(summaries_jsd, output_dir / "91_jsd_layerwise_summaries.pt")
    gaps_jsd = _load_cached(output_dir / "92_jsd_gap_summaries.pt")
    if gaps_jsd is None:
        gaps_jsd = build_prompt_null_gap_summaries(summaries_jsd)
        torch.save(gaps_jsd, output_dir / "92_jsd_gap_summaries.pt")
    save_prompt_metric_actual_summary_pdf(
        summaries_jsd,
        path=output_dir / "20_non_calibrated_jsd_raw_vs_model_norm.pdf",
        metric_mode="non_calibrated",
        metric_label_override="JSD",
        family_labels=actual_family_labels,
        title="Non-Calibrated JSD | Raw & ModelNorm LogitDiff",
    )

    bundle_actual = [("non_calibrated", summaries_jsd, "JSD")]
    bundle_gap: list[tuple[str, dict[tuple[str, str, str], object], str]] = [("non_calibrated", gaps_jsd, "JSD")]
    summary_cache: dict[str, object] = {
        "jsd_summaries": summaries_jsd,
        "jsd_gap_summaries": gaps_jsd,
        "inputs": common_kwargs,
    }

    for metric_name, stem, label, summaries_file, gaps_file, score_gap_pdf, actual_pdf in [
        ("top1_ece", "top1_ece", "Top-1 ECE", "93_top1_ece_layerwise_summaries.pt", "94_top1_ece_gap_summaries.pt", "10_calibrated_top1_ece_raw_vs_model_norm.pdf", "21_calibrated_top1_ece_raw_vs_model_norm.pdf"),
        ("brier", "brier", "Brier", "95_brier_layerwise_summaries.pt", "96_brier_gap_summaries.pt", "11_calibrated_brier_raw_vs_model_norm.pdf", "22_calibrated_brier_raw_vs_model_norm.pdf"),
    ]:
        summaries_path = output_dir / summaries_file
        gaps_path = output_dir / gaps_file
        summaries = _load_cached(summaries_path)
        if summaries is None:
            summaries = build_prompt_null_sanity_summaries(
                calibrated_metric_name=metric_name,
                **common_kwargs,
            )
            torch.save(summaries, summaries_path)
        gaps = _load_cached(gaps_path)
        if gaps is None:
            gaps = build_prompt_null_gap_summaries(summaries)
            torch.save(gaps, gaps_path)
        save_prompt_metric_score_and_gap_pdf(
            summaries,
            gaps,
            path=output_dir / score_gap_pdf,
            metric_mode="calibrated",
            metric_label_override=label,
            actual_family_labels=actual_family_labels,
            gap_family_labels=gap_family_labels,
            title=f"Calibrated {label} | Raw & ModelNorm LogitDiff",
        )
        save_prompt_metric_actual_summary_pdf(
            summaries,
            path=output_dir / actual_pdf,
            metric_mode="calibrated",
            metric_label_override=label,
            family_labels=actual_family_labels,
            title=f"Calibrated {label} | Raw & ModelNorm LogitDiff",
        )
        bundle_actual.append(("calibrated", summaries, label))
        bundle_gap.append(("calibrated", gaps, label))
        summary_cache[f"{stem}_summaries"] = summaries
        summary_cache[f"{stem}_gap_summaries"] = gaps

    topk_ece_specs = [
        (5, "103_top5_ece_layerwise_summaries.pt", "104_top5_ece_gap_summaries.pt"),
        (10, "105_top10_ece_layerwise_summaries.pt", "106_top10_ece_gap_summaries.pt"),
    ]
    for k, summaries_name, gaps_name in topk_ece_specs:
        metric_name = f"top{k}_mass_ece"
        summaries_path = output_dir / summaries_name
        gaps_path = output_dir / gaps_name
        summaries = _load_cached(summaries_path)
        if summaries is None:
            summaries = build_prompt_null_sanity_summaries(
                calibrated_metric_name=metric_name,
                **common_kwargs,
            )
            torch.save(summaries, summaries_path)
        gaps = _load_cached(gaps_path)
        if gaps is None:
            gaps = build_prompt_null_gap_summaries(summaries)
            torch.save(gaps, gaps_path)
        summary_cache[f"top{k}_ece_summaries"] = summaries
        summary_cache[f"top{k}_ece_gap_summaries"] = gaps

    hidden_metric_specs = [
        ("hidden_cosine_similarity", "115_hidden_cosine_similarity_layerwise_summaries.pt", "116_hidden_cosine_similarity_gap_summaries.pt"),
        ("hidden_l2_distance", "117_hidden_l2_distance_layerwise_summaries.pt", "118_hidden_l2_distance_gap_summaries.pt"),
        ("hidden_normalized_l2_distance", "119_hidden_normalized_l2_distance_layerwise_summaries.pt", "120_hidden_normalized_l2_distance_gap_summaries.pt"),
    ]
    for metric_name, summaries_name, gaps_name in hidden_metric_specs:
        summaries_path = output_dir / summaries_name
        gaps_path = output_dir / gaps_name
        summaries = _load_cached(summaries_path)
        if summaries is None:
            summaries = build_prompt_hidden_metric_summaries(
                trained_artifact=args.trained_artifact,
                baseline_artifact=args.baseline_artifact,
                random_artifacts=list(args.random_artifacts),
                hidden_metric_name=metric_name,
                num_permutations=1000,
                alpha=0.05,
                seed=31,
            )
            torch.save(summaries, summaries_path)
        gaps = _load_cached(gaps_path)
        if gaps is None:
            gaps = build_prompt_null_gap_summaries(summaries)
            torch.save(gaps, gaps_path)
        summary_cache[f"{metric_name}_summaries"] = summaries
        summary_cache[f"{metric_name}_gap_summaries"] = gaps

    save_prompt_metric_bundle_actual_pdf(
        bundle_actual,
        path=output_dir / "30_non_calibrated_and_calibrated_raw_vs_model_norm.pdf",
        family_labels=actual_family_labels,
        title="Non-Calibrated JSD & Calibrated Top-1 ECE & Brier | Raw & ModelNorm LogitDiff",
    )
    save_prompt_metric_bundle_gap_pdf(
        bundle_gap,
        path=output_dir / "31_gap_metric_bundle_raw_vs_model_norm.pdf",
        family_labels=gap_family_labels,
        title="Non-Calibrated JSD Gap & Calibrated Top-1 ECE & Brier Gaps | Raw & ModelNorm LogitDiff",
    )

    save_prompt_jsd_calibration_overlay_pdf(
        summaries_jsd,
        summary_cache["top1_ece_summaries"],  # type: ignore[arg-type]
        path=output_dir / "32_overlay_jsd_vs_top1_ece_raw_modelnorm.pdf",
        calibrated_label="Top-1 ECE",
        family_titles_override=actual_family_labels,
        title="Non-Calibrated JSD & Calibrated Top-1 ECE | Raw & ModelNorm LogitDiff",
    )
    save_prompt_four_view_summary_pdf(
        summaries_jsd,
        summary_cache["top1_ece_summaries"],  # type: ignore[arg-type]
        path=output_dir / "38_four_view_jsd_vs_top1_ece.pdf",
        calibrated_label="Top-1 ECE",
        family_labels=actual_family_labels,
        title="Non-Calibrated JSD & Calibrated Top-1 ECE | Raw & ModelNorm LogitDiff",
    )
    save_prompt_jsd_calibration_overlay_pdf(
        summaries_jsd,
        summary_cache["brier_summaries"],  # type: ignore[arg-type]
        path=output_dir / "33_overlay_jsd_vs_brier_raw_modelnorm.pdf",
        calibrated_label="Brier",
        family_titles_override=actual_family_labels,
        title="Non-Calibrated JSD & Calibrated Brier | Raw & ModelNorm LogitDiff",
    )
    save_prompt_jsd_calibration_gap_overlay_pdf(
        gaps_jsd,
        summary_cache["top1_ece_gap_summaries"],  # type: ignore[arg-type]
        path=output_dir / "35_overlay_gap_jsd_vs_top1_ece_raw_modelnorm.pdf",
        calibrated_label="Top-1 ECE",
        family_titles_override=gap_family_labels,
        title="Non-Calibrated JSD Gap & Calibrated Top-1 ECE Gap | Raw & ModelNorm LogitDiff",
    )
    save_prompt_jsd_calibration_gap_overlay_pdf(
        gaps_jsd,
        summary_cache["brier_gap_summaries"],  # type: ignore[arg-type]
        path=output_dir / "36_overlay_gap_jsd_vs_brier_raw_modelnorm.pdf",
        calibrated_label="Brier",
        family_titles_override=gap_family_labels,
        title="Non-Calibrated JSD Gap & Calibrated Brier Gap | Raw & ModelNorm LogitDiff",
    )

    torch.save(summary_cache, output_dir / "90_prompt_random_sanity_summary_cache.pt")


if __name__ == "__main__":
    main()

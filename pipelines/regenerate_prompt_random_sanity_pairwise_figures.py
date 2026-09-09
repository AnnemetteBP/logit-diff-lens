from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
    save_robust_prompt_jsd_overview_pdf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate pairwise prompt random-sanity PDFs.")
    parser.add_argument("--trained-artifact", required=True)
    parser.add_argument("--baseline-artifact", required=True)
    parser.add_argument("--random-artifact", action="append", dest="random_artifacts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _notice(path: Path) -> None:
    print(f"[write] {path}", flush=True)


def _load_pairwise_cache(path: Path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_pairwise_cache(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "99_pairwise_curve_cache.pt"
    common_kwargs = dict(
        trained_artifact=args.trained_artifact,
        baseline_artifact=args.baseline_artifact,
        random_artifacts=list(args.random_artifacts),
        top_k=args.top_k,
    )
    cache_key = {
        "trained_artifact": args.trained_artifact,
        "baseline_artifact": args.baseline_artifact,
        "random_artifacts": list(args.random_artifacts),
        "top_k": args.top_k,
    }
    cache = _load_pairwise_cache(cache_path)
    if not isinstance(cache, dict) or cache.get("cache_key") != cache_key:
        cache = {"cache_key": cache_key}

    if "pairwise_jsd_raw" not in cache:
        raw_curves, raw_labels = build_prompt_pairwise_jsd_curves(readout_mode="raw", **common_kwargs)
        cache["pairwise_jsd_raw"] = (raw_curves, raw_labels)
        _save_pairwise_cache(cache_path, cache)
    raw_curves, raw_labels = cache["pairwise_jsd_raw"]

    if "pairwise_jsd_model_norm" not in cache:
        model_norm_curves, model_norm_labels = build_prompt_pairwise_jsd_curves(readout_mode="model_norm", **common_kwargs)
        cache["pairwise_jsd_model_norm"] = (model_norm_curves, model_norm_labels)
        _save_pairwise_cache(cache_path, cache)
    model_norm_curves, model_norm_labels = cache["pairwise_jsd_model_norm"]

    if "pairwise_jsd_gap_raw" not in cache:
        raw_gap_curves, _ = build_prompt_pairwise_jsd_gap_curves(readout_mode="raw", **common_kwargs)
        cache["pairwise_jsd_gap_raw"] = raw_gap_curves
        _save_pairwise_cache(cache_path, cache)
    raw_gap_curves = cache["pairwise_jsd_gap_raw"]

    if "pairwise_jsd_gap_model_norm" not in cache:
        model_norm_gap_curves, _ = build_prompt_pairwise_jsd_gap_curves(readout_mode="model_norm", **common_kwargs)
        cache["pairwise_jsd_gap_model_norm"] = model_norm_gap_curves
        _save_pairwise_cache(cache_path, cache)
    model_norm_gap_curves = cache["pairwise_jsd_gap_model_norm"]

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
    _notice(path)

    path = output_dir / "01_non_calibrated_jsd_pairwise_raw_vs_model_norm.pdf"
    save_combined_pairwise_jsd_pdf(
        raw_curves=raw_curves,
        raw_layer_labels=raw_labels,
        model_norm_curves=model_norm_curves,
        model_norm_layer_labels=model_norm_labels,
        path=path,
        title="Non-Calibrated JSD | Raw & ModelNorm LogitDiff",
    )
    _notice(path)

    path = output_dir / "02_jsd_gap_vs_random_null_raw_vs_model_norm.pdf"
    save_combined_pairwise_jsd_gap_pdf(
        raw_gap_curves=raw_gap_curves,
        raw_layer_labels=raw_labels,
        model_norm_gap_curves=model_norm_gap_curves,
        model_norm_layer_labels=model_norm_labels,
        path=path,
        title="Non-Calibrated ΔJSD | Raw & ModelNorm LogitDiff",
    )
    _notice(path)

    pairwise_metric_specs = [
        ("top1_ece", "Top-1 ECE", "03_calibrated_top1_ece_pairwise_raw_vs_model_norm.pdf", "06_calibrated_top1_ece_gap_pairwise_raw_vs_model_norm.pdf"),
        ("brier", "Brier", "04_calibrated_brier_pairwise_raw_vs_model_norm.pdf", "07_calibrated_brier_gap_pairwise_raw_vs_model_norm.pdf"),
        ("nll", "Negative Log-Likelihood", "05_calibrated_nll_pairwise_raw_vs_model_norm.pdf", "08_calibrated_nll_gap_pairwise_raw_vs_model_norm.pdf"),
    ]
    for metric_name, metric_label, pairwise_name, pairwise_gap_name in pairwise_metric_specs:
        raw_metric_key = f"pairwise_metric_raw::{metric_name}"
        if raw_metric_key not in cache:
            raw_metric_curves, raw_metric_labels = build_prompt_pairwise_metric_curves(
                readout_mode="raw",
                metric_mode="calibrated",
                calibrated_metric_name=metric_name,
                **common_kwargs,
            )
            cache[raw_metric_key] = (raw_metric_curves, raw_metric_labels)
            _save_pairwise_cache(cache_path, cache)
        raw_metric_curves, raw_metric_labels = cache[raw_metric_key]

        model_norm_metric_key = f"pairwise_metric_model_norm::{metric_name}"
        if model_norm_metric_key not in cache:
            model_norm_metric_curves, model_norm_metric_labels = build_prompt_pairwise_metric_curves(
                readout_mode="model_norm",
                metric_mode="calibrated",
                calibrated_metric_name=metric_name,
                **common_kwargs,
            )
            cache[model_norm_metric_key] = (model_norm_metric_curves, model_norm_metric_labels)
            _save_pairwise_cache(cache_path, cache)
        model_norm_metric_curves, model_norm_metric_labels = cache[model_norm_metric_key]
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
        _notice(path)

        raw_metric_gap_key = f"pairwise_metric_gap_raw::{metric_name}"
        if raw_metric_gap_key not in cache:
            raw_metric_gap_curves, raw_metric_gap_labels = build_prompt_pairwise_metric_gap_curves(
                readout_mode="raw",
                metric_mode="calibrated",
                calibrated_metric_name=metric_name,
                **common_kwargs,
            )
            cache[raw_metric_gap_key] = (raw_metric_gap_curves, raw_metric_gap_labels)
            _save_pairwise_cache(cache_path, cache)
        raw_metric_gap_curves, raw_metric_gap_labels = cache[raw_metric_gap_key]

        model_norm_metric_gap_key = f"pairwise_metric_gap_model_norm::{metric_name}"
        if model_norm_metric_gap_key not in cache:
            model_norm_metric_gap_curves, model_norm_metric_gap_labels = build_prompt_pairwise_metric_gap_curves(
                readout_mode="model_norm",
                metric_mode="calibrated",
                calibrated_metric_name=metric_name,
                **common_kwargs,
            )
            cache[model_norm_metric_gap_key] = (model_norm_metric_gap_curves, model_norm_metric_gap_labels)
            _save_pairwise_cache(cache_path, cache)
        model_norm_metric_gap_curves, model_norm_metric_gap_labels = cache[model_norm_metric_gap_key]
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
        _notice(path)

    for calibrated, stem, title_prefix, seed_base in [
        (False, "47_non_calibrated", "Non-Calibrated JS Similarity", 21),
        (True, "49_calibrated", "Calibrated JS Similarity", 25),
    ]:
        similarity_mode = "calibrated" if calibrated else "non_calibrated"
        raw_similarity_key = f"pairwise_similarity_raw::{similarity_mode}"
        if raw_similarity_key not in cache:
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
            cache[raw_similarity_key] = (raw_curves, raw_labels)
            _save_pairwise_cache(cache_path, cache)
        raw_curves, raw_labels = cache[raw_similarity_key]

        model_norm_similarity_key = f"pairwise_similarity_model_norm::{similarity_mode}"
        if model_norm_similarity_key not in cache:
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
            cache[model_norm_similarity_key] = (model_norm_curves, model_norm_labels)
            _save_pairwise_cache(cache_path, cache)
        model_norm_curves, model_norm_labels = cache[model_norm_similarity_key]
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
        _notice(path)

        raw_similarity_gap_key = f"pairwise_similarity_gap_raw::{similarity_mode}"
        if raw_similarity_gap_key not in cache:
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
            cache[raw_similarity_gap_key] = (raw_gap_curves, raw_gap_labels)
            _save_pairwise_cache(cache_path, cache)
        raw_gap_curves, raw_gap_labels = cache[raw_similarity_gap_key]

        model_norm_similarity_gap_key = f"pairwise_similarity_gap_model_norm::{similarity_mode}"
        if model_norm_similarity_gap_key not in cache:
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
            cache[model_norm_similarity_gap_key] = (model_norm_gap_curves, model_norm_gap_labels)
            _save_pairwise_cache(cache_path, cache)
        model_norm_gap_curves, model_norm_gap_labels = cache[model_norm_similarity_gap_key]
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
        _notice(path)


if __name__ == "__main__":
    main()

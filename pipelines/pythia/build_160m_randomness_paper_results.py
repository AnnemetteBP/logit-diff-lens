from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from logit_diff_lens.plotting.prompt_null_sanity import (
    FamilySummary,
    build_prompt_null_gap_summaries,
    build_prompt_null_sanity_summaries,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tmp" / "pythia_160m_randomness_paper"
ARTIFACT_DIR = ROOT / "tmp" / "pythia_160m_randomness_pilot" / "artifacts"


@dataclass(frozen=True)
class ComparisonSpec:
    slug: str
    label: str
    trained_artifact: Path
    baseline_artifact: Path


COMPARISONS = [
    ComparisonSpec(
        slug="160m_1k_vs_143k",
        label="160M 1k vs 143k",
        trained_artifact=ARTIFACT_DIR / "pythia_160m_step1000_prompt_bundle.pt",
        baseline_artifact=ARTIFACT_DIR / "pythia_160m_step143000_prompt_bundle.pt",
    ),
    ComparisonSpec(
        slug="160m_71k_vs_143k",
        label="160M 71k vs 143k",
        trained_artifact=ARTIFACT_DIR / "pythia_160m_step71000_prompt_bundle.pt",
        baseline_artifact=ARTIFACT_DIR / "pythia_160m_step143000_prompt_bundle.pt",
    ),
]

RANDOM_ARTIFACTS = [
    ARTIFACT_DIR / "pythia_160m_seed123_prompt_bundle.pt",
    ARTIFACT_DIR / "pythia_160m_seed456_prompt_bundle.pt",
    ARTIFACT_DIR / "pythia_160m_seed789_prompt_bundle.pt",
]


def _short_labels(labels: list[str]) -> list[str]:
    short = []
    for label in labels:
        if label == "Embedding":
            short.append("Emb")
        elif label == "Output":
            short.append("Out")
        elif label.startswith("L") and "/" in label:
            short.append(label.split("/", 1)[0])
        else:
            short.append(label)
    return short


def _layer_blocks(labels: list[str]) -> dict[str, list[int]]:
    emb_idx = [i for i, label in enumerate(labels) if label == "Embedding"]
    out_idx = [i for i, label in enumerate(labels) if label == "Output"]
    core = [i for i, label in enumerate(labels) if label not in {"Embedding", "Output"}]
    n = len(core)
    if n == 0:
        thirds = ([], [], [])
    else:
        cut1 = max(1, n // 3)
        cut2 = max(cut1 + 1, (2 * n) // 3)
        thirds = (core[:cut1], core[cut1:cut2], core[cut2:])
    return {
        "Emb": emb_idx,
        "Early": thirds[0],
        "Mid": thirds[1],
        "Late": thirds[2],
        "Out": out_idx,
    }


def _mean_for_indices(values: torch.Tensor, indices: list[int]) -> float:
    if not indices:
        return float("nan")
    subset = values[indices]
    return float(torch.nanmean(subset).item())


def _summarize_gap(summary: FamilySummary) -> dict[str, float]:
    blocks = _layer_blocks(summary.layer_labels)
    return {
        "all": float(torch.nanmean(summary.mean).item()),
        "emb": _mean_for_indices(summary.mean, blocks["Emb"]),
        "early": _mean_for_indices(summary.mean, blocks["Early"]),
        "mid": _mean_for_indices(summary.mean, blocks["Mid"]),
        "late": _mean_for_indices(summary.mean, blocks["Late"]),
        "out": _mean_for_indices(summary.mean, blocks["Out"]),
    }


def _sign_flip_pvalue(values: torch.Tensor, *, num_permutations: int = 2500, seed: int = 0) -> float:
    scores = values.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    scores = scores[torch.isfinite(scores)]
    if scores.numel() == 0:
        return float("nan")
    observed = float(scores.mean().item())
    if scores.numel() == 1:
        return 1.0 if observed <= 0.0 else 0.0
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    null_means = []
    for _ in range(num_permutations):
        flips = torch.randint(0, 2, scores.shape, generator=gen, dtype=torch.int64)
        signs = torch.where(flips == 0, -torch.ones_like(scores), torch.ones_like(scores))
        null_means.append(float((scores * signs).mean().item()))
    null_tensor = torch.tensor(null_means, dtype=torch.float32)
    exceed = int((null_tensor >= observed).sum().item())
    return float((exceed + 1) / (num_permutations + 1))


def _prompt_mean_ci(values: torch.Tensor, *, num_bootstrap: int = 2500, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    scores = values.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    scores = scores[torch.isfinite(scores)]
    if scores.numel() == 0:
        return float("nan"), float("nan")
    if scores.numel() == 1:
        val = float(scores.item())
        return val, val
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    draws = []
    for _ in range(num_bootstrap):
        idx = torch.randint(0, scores.numel(), (scores.numel(),), generator=gen)
        draws.append(float(scores[idx].mean().item()))
    draw_tensor = torch.tensor(draws, dtype=torch.float32)
    lo = float(torch.quantile(draw_tensor, alpha / 2.0).item())
    hi = float(torch.quantile(draw_tensor, 1.0 - alpha / 2.0).item())
    return lo, hi


def _write_randomness_table(
    all_gap_results: dict[str, dict[tuple[str, str, str], FamilySummary]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in COMPARISONS:
        gaps = all_gap_results[spec.slug]
        for readout_mode, readout_label in [("raw", "Raw"), ("model_norm", "ModelNorm")]:
            for metric_mode, regime_label in [("non_calibrated", "NC"), ("calibrated", "C")]:
                key = (readout_mode, metric_mode, "trained_vs_trained_minus_random_vs_random")
                summary = gaps[key]
                stats = _summarize_gap(summary)
                p_perm = _sign_flip_pvalue(summary.per_prompt_curves.mean(dim=1), seed=17)
                rows.append(
                    (
                        spec.label,
                        readout_label,
                        regime_label,
                        stats["all"],
                        stats["emb"],
                        stats["early"],
                        stats["mid"],
                        stats["late"],
                        stats["out"],
                        p_perm,
                    )
                )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllrrrrrrr}",
        "\\toprule",
        "Cmp. & Lens & Regime & \\multicolumn{6}{c}{Observed-vs-null separation} & \\\\",
        "\\cmidrule(lr){4-9}",
        " &  &  & $\\bar{\\Delta}$ & Emb. & Early & Mid & Late & Out & $p_{\\mathrm{perm}}$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row[0]} & {row[1]} & {row[2]} & {row[3]:.4f} & {row[4]:.4f} & {row[5]:.4f} & {row[6]:.4f} & {row[7]:.4f} & {row[8]:.4f} & {row[9]:.4f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Observed-vs-null randomness summary. `Cmp.` denotes the checkpoint comparison against the corresponding 143k base checkpoint. $\\bar{\\Delta}$ is the prompt-aggregated mean observed-vs-null separation across the full analyzed layer axis. Emb., Early, Mid, Late, and Out report the same separation aggregated over the embedding row, the early-depth block, the middle-depth block, the late-depth block, and the synthetic output row. `NC` and `C` denote the non-calibrated and calibrated LogitDiff measurements, and the final column reports the permutation-test significance of the observed-vs-null separation.}",
            "\\label{tab:randomness-main}",
            "\\end{table*}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_appendix_randomness_separation_table(
    all_gap_results: dict[str, dict[tuple[str, str, str], FamilySummary]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in COMPARISONS:
        gaps = all_gap_results[spec.slug]
        for readout_mode, readout_label in [("raw", "Raw"), ("model_norm", "ModelNorm")]:
            for metric_mode, regime_label in [("non_calibrated", "NC"), ("calibrated", "C")]:
                key = (readout_mode, metric_mode, "trained_vs_trained_minus_random_vs_random")
                summary = gaps[key]
                stats = _summarize_gap(summary)
                prompt_means = summary.per_prompt_curves.mean(dim=1)
                ci_low, ci_high = _prompt_mean_ci(prompt_means, seed=23)
                p_perm = _sign_flip_pvalue(prompt_means, seed=29)
                rows.append(
                    (
                        spec.label,
                        readout_label,
                        regime_label,
                        stats["all"],
                        stats["emb"],
                        stats["early"],
                        stats["mid"],
                        stats["late"],
                        stats["out"],
                        ci_low,
                        ci_high,
                        p_perm,
                    )
                )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllrrrrrrrr}",
        "\\toprule",
        "Cmp. & Lens & Regime & Mean $\\Delta$ & Emb. & Early & Mid & Late & Out & 95\\% CI & $p_{\\mathrm{perm}}$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row[0]} & {row[1]} & {row[2]} & {row[3]:.4f} & {row[4]:.4f} & {row[5]:.4f} & {row[6]:.4f} & {row[7]:.4f} & {row[8]:.4f} & [{row[9]:.4f}, {row[10]:.4f}] & {row[11]:.4f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Observed-vs-null summary for the randomness sanity check. For each comparison, lens, and regime, $\\Delta$ denotes the mean separation across prompts aggregated over \\emph{all} probed layers between the observed layerwise signal and the matching architecture-matched random baseline, relative to the null-family self-separation. The observed signal is built from the readout object in Eq.~\\eqref{eq:logitdiff-logits}--Eq.~\\eqref{eq:logitdiff-distribution}, while the random baseline is defined by Eq.~\\eqref{eq:logitdiff-random-baseline-appendix}. Emb., Early, Mid, Late, and Out denote the corresponding blockwise mean separations over the embedding row, early-depth block, middle-depth block, late-depth block, and synthetic output row, where the depth blocks are defined by Eq.~\\eqref{eq:appendix-depth-blocks}--Eq.~\\eqref{eq:appendix-layer-block-aggregation}. NC/C denote non-calibrated and calibrated similarity estimation.}",
            "\\label{tab:app-randomness-separation}",
            "\\end{table*}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_gap_results: dict[str, dict[tuple[str, str, str], FamilySummary]] = {}
    for spec in COMPARISONS:
        print(f"[randomness] building summaries for {spec.slug}", flush=True)
        comparison_dir = OUT_DIR / spec.slug
        comparison_dir.mkdir(parents=True, exist_ok=True)
        summaries = build_prompt_null_sanity_summaries(
            trained_artifact=str(spec.trained_artifact),
            baseline_artifact=str(spec.baseline_artifact),
            random_artifacts=[str(path) for path in RANDOM_ARTIFACTS],
            calibrated_metric_name="top1_ece",
        )
        gaps = build_prompt_null_gap_summaries(summaries)
        all_gap_results[spec.slug] = gaps
        torch.save(summaries, comparison_dir / "randomness_summaries.pt")
        torch.save(gaps, comparison_dir / "randomness_gap_summaries.pt")
        print(f"[randomness] wrote summary tensors for {spec.slug}", flush=True)
    _write_randomness_table(all_gap_results, OUT_DIR / "randomness_main_table.tex")
    _write_appendix_randomness_separation_table(all_gap_results, OUT_DIR / "appendix_randomness_separation_table.tex")
    print("[randomness] wrote paper tables", flush=True)


if __name__ == "__main__":
    main()

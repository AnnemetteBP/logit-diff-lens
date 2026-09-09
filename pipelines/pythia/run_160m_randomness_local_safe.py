import argparse
import csv
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from logit_diff_lens.diffing.io import load_prompt_decode_artifact_bundle, load_prompt_decode_artifact
from logit_diff_lens.plotting.prompt_null_sanity import _non_calibrated_curve, _similarity_curve
from logit_diff_lens.schemas import PromptDecodeArtifact


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "tmp" / "pythia_160m_randomness_pilot" / "artifacts"
OUT_DIR = ROOT / "tmp" / "pythia_160m_randomness_local_safe"
SHARD_DIR = OUT_DIR / "prompt_shards"
REDUCED_DIR = OUT_DIR / "reduced_curves"


@dataclass(frozen=True)
class BundleSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class ComparisonSpec:
    slug: str
    label: str
    comparison_name: str
    baseline_name: str


BUNDLES = [
    BundleSpec("step1000", ARTIFACT_DIR / "pythia_160m_step1000_prompt_bundle.pt"),
    BundleSpec("step71000", ARTIFACT_DIR / "pythia_160m_step71000_prompt_bundle.pt"),
    BundleSpec("step143000", ARTIFACT_DIR / "pythia_160m_step143000_prompt_bundle.pt"),
    BundleSpec("seed123", ARTIFACT_DIR / "pythia_160m_seed123_prompt_bundle.pt"),
    BundleSpec("seed456", ARTIFACT_DIR / "pythia_160m_seed456_prompt_bundle.pt"),
    BundleSpec("seed789", ARTIFACT_DIR / "pythia_160m_seed789_prompt_bundle.pt"),
]

COMPARISONS = [
    ComparisonSpec("160m_1k_vs_143k", "160M 1k--143k", "step1000", "step143000"),
    ComparisonSpec("160m_71k_vs_143k", "160M 71k--143k", "step71000", "step143000"),
]

RANDOM_NAMES = ["seed123", "seed456", "seed789"]
READOUTS = [("raw", "Raw"), ("model_norm", "ModelNorm")]
REGIMES = [("non_calibrated", "NC"), ("calibrated", "C")]
NULL_PAIR_LABELS = {
    ("seed123", "seed456"): "seed123--seed456",
    ("seed123", "seed789"): "seed123--seed789",
    ("seed456", "seed789"): "seed456--seed789",
}


def _artifact_key(artifact: PromptDecodeArtifact) -> str:
    key = artifact.prompt_id if artifact.prompt_id is not None else artifact.prompt_text
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key.strip())[:120]
    return key or "prompt"


def _shard_path(bundle_name: str, key: str) -> Path:
    return SHARD_DIR / bundle_name / f"{key}.pt"


def _cache_path(*parts: str) -> Path:
    safe_parts = [re.sub(r"[^A-Za-z0-9_.-]+", "_", part) for part in parts]
    return REDUCED_DIR / ("__".join(safe_parts) + ".pt")


def _shard_bundle(spec: BundleSpec, *, force: bool = False) -> list[str]:
    out_dir = SHARD_DIR / spec.name
    manifest = out_dir / "manifest.txt"
    if manifest.exists() and not force:
        return [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[shard] loading one bundle: {spec.path}", flush=True)
    bundle = load_prompt_decode_artifact_bundle(spec.path)
    keys: list[str] = []
    for artifact in bundle["artifacts"]:
        key = _artifact_key(artifact)
        keys.append(key)
        torch.save(artifact.to_dict(), _shard_path(spec.name, key))
    manifest.write_text("\n".join(keys) + "\n", encoding="utf-8")
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[shard] wrote {len(keys)} prompt shards for {spec.name}", flush=True)
    return keys


def _load_shard(bundle_name: str, key: str) -> PromptDecodeArtifact:
    return load_prompt_decode_artifact(_shard_path(bundle_name, key))


def _layer_labels(artifact: PromptDecodeArtifact) -> list[str]:
    positive = [r.layer_index for r in artifact.layer_records if r.layer_index >= 0 and r.layer_name != "output"]
    total_layers = max(positive) + 1 if positive else max(1, len(artifact.layer_records))
    labels = []
    for record in artifact.layer_records:
        if record.layer_index < 0:
            labels.append("Embedding")
        elif record.layer_name == "output":
            labels.append("Output")
        else:
            labels.append(f"L{record.layer_index + 1}/{total_layers}")
    return labels


def _curve(
    comparison: PromptDecodeArtifact,
    baseline: PromptDecodeArtifact,
    *,
    readout: str,
    regime: str,
) -> torch.Tensor:
    if regime == "non_calibrated":
        return _non_calibrated_curve(comparison, baseline, readout_mode=readout, top_k=10)
    if regime == "calibrated":
        calibrated_similarity = _similarity_curve(
            comparison,
            baseline,
            readout_mode=readout,
            similarity_metric_name="js_similarity",
            top_k=10,
            num_permutations=250,
            alpha=0.05,
            seed=17,
            calibrated=True,
        )
        return 1.0 - calibrated_similarity
    raise ValueError(f"Unknown regime: {regime}")


def _curves_for_pairs(
    pair_names: list[tuple[str, str]],
    keys: list[str],
    *,
    readout: str,
    regime: str,
) -> tuple[torch.Tensor, list[str], list[str]]:
    per_prompt = []
    kept_keys = []
    layer_labels: list[str] | None = None
    for idx, key in enumerate(keys, start=1):
        curves = []
        for left_name, right_name in pair_names:
            left = _load_shard(left_name, key)
            right = _load_shard(right_name, key)
            if layer_labels is None:
                layer_labels = _layer_labels(left)
            curves.append(_curve(left, right, readout=readout, regime=regime))
            del left
            del right
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        per_prompt.append(torch.stack(curves, dim=0))
        kept_keys.append(key)
        if idx % 10 == 0:
            print(f"[reduce] {readout}/{regime}: {idx}/{len(keys)} prompts", flush=True)
    return torch.stack(per_prompt, dim=0), (layer_labels or []), kept_keys


def _mean_curve_for_pairs(
    pair_names: list[tuple[str, str]],
    keys: list[str],
    *,
    readout: str,
    regime: str,
) -> tuple[torch.Tensor, list[str], list[str]]:
    pair_curves, layer_labels, kept_keys = _curves_for_pairs(
        pair_names,
        keys,
        readout=readout,
        regime=regime,
    )
    return pair_curves.mean(dim=1), layer_labels, kept_keys


def _load_or_compute_pair_curves(
    cache_name: str,
    pair_names: list[tuple[str, str]],
    keys: list[str],
    *,
    readout: str,
    regime: str,
) -> tuple[torch.Tensor, list[str], list[str]]:
    path = _cache_path(cache_name, readout, regime)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload["curves"], payload["layer_labels"], payload["keys"]

    curves, layer_labels, kept_keys = _curves_for_pairs(
        pair_names,
        keys,
        readout=readout,
        regime=regime,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "curves": curves.cpu(),
            "layer_labels": layer_labels,
            "keys": kept_keys,
            "pair_names": pair_names,
        },
        path,
    )
    return curves, layer_labels, kept_keys


def _load_or_compute_mean_curves(
    cache_name: str,
    pair_names: list[tuple[str, str]],
    keys: list[str],
    *,
    readout: str,
    regime: str,
) -> tuple[torch.Tensor, list[str], list[str]]:
    pair_curves, layer_labels, kept_keys = _load_or_compute_pair_curves(
        cache_name,
        pair_names,
        keys,
        readout=readout,
        regime=regime,
    )
    return pair_curves.mean(dim=1), layer_labels, kept_keys


def _blocks(labels: list[str]) -> dict[str, list[int]]:
    emb = [i for i, label in enumerate(labels) if label == "Embedding"]
    out = [i for i, label in enumerate(labels) if label == "Output"]
    core = [i for i, label in enumerate(labels) if label not in {"Embedding", "Output"}]
    n = len(core)
    cut1 = max(1, n // 3) if n else 0
    cut2 = max(cut1 + 1, (2 * n) // 3) if n else 0
    return {"Emb": emb, "Early": core[:cut1], "Mid": core[cut1:cut2], "Late": core[cut2:], "Out": out}


def _mean_at(values: torch.Tensor, idx: list[int]) -> float:
    if not idx:
        return float("nan")
    return float(torch.nanmean(values[idx]).item())


def _sign_flip_pvalue(prompt_values: torch.Tensor, *, num_permutations: int = 2500, seed: int = 19) -> float:
    vals = prompt_values.detach().to(dtype=torch.float32, device="cpu")
    vals = vals[torch.isfinite(vals)]
    if vals.numel() < 2:
        return float("nan")
    observed = abs(float(vals.mean().item()))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    null = []
    for _ in range(num_permutations):
        flips = torch.randint(0, 2, vals.shape, generator=gen)
        signs = torch.where(flips == 0, -torch.ones_like(vals), torch.ones_like(vals))
        null.append(abs(float((vals * signs).mean().item())))
    null_t = torch.tensor(null, dtype=torch.float32)
    return float(((null_t >= observed).sum().item() + 1) / (num_permutations + 1))


def _bootstrap_mean_ci(
    prompt_values: torch.Tensor,
    *,
    num_bootstrap: int = 2500,
    alpha: float = 0.05,
    seed: int = 23,
) -> tuple[float, float]:
    vals = prompt_values.detach().to(dtype=torch.float32, device="cpu")
    vals = vals[torch.isfinite(vals)]
    if vals.numel() < 2:
        value = float(vals.mean().item()) if vals.numel() else float("nan")
        return value, value
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    means = torch.empty(num_bootstrap, dtype=torch.float32)
    for idx in range(num_bootstrap):
        sample_idx = torch.randint(0, vals.numel(), (vals.numel(),), generator=gen)
        means[idx] = vals[sample_idx].mean()
    low = float(torch.quantile(means, alpha / 2.0).item())
    high = float(torch.quantile(means, 1.0 - alpha / 2.0).item())
    return low, high


def _row_stats(
    cmp_label: str,
    lens: str,
    regime: str,
    observed_curves: torch.Tensor,
    null_curves: torch.Tensor,
    gap_mean: torch.Tensor,
    labels: list[str],
    prompt_gap: torch.Tensor,
) -> dict[str, float | str]:
    blocks = _blocks(labels)
    signed_prompt_means = prompt_gap.mean(dim=1)
    separation_by_layer = prompt_gap.abs().mean(dim=0)
    separation_prompt_means = prompt_gap.abs().mean(dim=1)
    observed_by_layer = observed_curves.mean(dim=0)
    null_by_layer = null_curves.mean(dim=0)
    ci_low, ci_high = _bootstrap_mean_ci(separation_prompt_means)
    return {
        "comparison": cmp_label,
        "lens": lens,
        "regime": regime,
        "observed_mean": float(torch.nanmean(observed_by_layer).item()),
        "null_mean": float(torch.nanmean(null_by_layer).item()),
        "mean_delta": float(torch.nanmean(separation_by_layer).item()),
        "signed_mean_delta": float(torch.nanmean(gap_mean).item()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "emb": _mean_at(separation_by_layer, blocks["Emb"]),
        "early": _mean_at(separation_by_layer, blocks["Early"]),
        "mid": _mean_at(separation_by_layer, blocks["Mid"]),
        "late": _mean_at(separation_by_layer, blocks["Late"]),
        "out": _mean_at(separation_by_layer, blocks["Out"]),
        "p_perm": _sign_flip_pvalue(signed_prompt_means),
    }


def _null_pair_stats(
    pair_label: str,
    lens: str,
    regime: str,
    labels: list[str],
    prompt_curves: torch.Tensor,
) -> dict[str, float | str]:
    blocks = _blocks(labels)
    mean_by_layer = prompt_curves.mean(dim=0)
    prompt_means = prompt_curves.mean(dim=1)
    ci_low, ci_high = _bootstrap_mean_ci(prompt_means)
    return {
        "null_pair": pair_label,
        "lens": lens,
        "regime": regime,
        "mean_null": float(torch.nanmean(mean_by_layer).item()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "emb": _mean_at(mean_by_layer, blocks["Emb"]),
        "early": _mean_at(mean_by_layer, blocks["Early"]),
        "mid": _mean_at(mean_by_layer, blocks["Mid"]),
        "late": _mean_at(mean_by_layer, blocks["Late"]),
        "out": _mean_at(mean_by_layer, blocks["Out"]),
    }


def _write_jsonl(rows: list[dict[str, float | str]], path: Path) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _add_bh_q_values(rows: list[dict[str, float | str]]) -> None:
    indexed = sorted(
        [(idx, float(row["p_perm"])) for idx, row in enumerate(rows) if torch.isfinite(torch.tensor(float(row["p_perm"])))],
        key=lambda item: item[1],
        reverse=True,
    )
    m = len(indexed)
    running = 1.0
    q_values: dict[int, float] = {}
    for rank_from_high, (idx, p_value) in enumerate(indexed):
        rank = m - rank_from_high
        running = min(running, p_value * m / max(rank, 1))
        q_values[idx] = min(1.0, max(0.0, running))
    for idx, row in enumerate(rows):
        row["q_bh"] = q_values.get(idx, float("nan"))


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    fieldnames = [
        "comparison",
        "lens",
        "regime",
        "observed_mean",
        "null_mean",
        "mean_delta",
        "signed_mean_delta",
        "ci_low",
        "ci_high",
        "emb",
        "early",
        "mid",
        "late",
        "out",
        "p_perm",
        "q_bh",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_main_table(rows: list[dict[str, float | str]], path: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllrrrrrrrrrrrrr}",
        "\\toprule",
        "Cmp. & Lens & Regime & \\multicolumn{2}{c}{Mean signal} & \\multicolumn{7}{c}{Observed-vs-null separation} & \\multicolumn{2}{c}{Test} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-12}\\cmidrule(lr){13-14}",
        " &  &  & Obs. & Null & $\\overline{|\\Delta|}$ & 95\\% CI & Emb. & Early & Mid & Late & Out & $p_{\\mathrm{perm}}$ & $q_{\\mathrm{BH}}$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison']} & {row['lens']} & {row['regime']} & "
            f"{float(row['observed_mean']):.4f} & "
            f"{float(row['null_mean']):.4f} & "
            f"{float(row['mean_delta']):.4f} & "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] & "
            f"{float(row['emb']):.4f} & "
            f"{float(row['early']):.4f} & "
            f"{float(row['mid']):.4f} & "
            f"{float(row['late']):.4f} & "
            f"{float(row['out']):.4f} & "
            f"{float(row['p_perm']):.4f} & "
            f"{float(row['q_bh']):.4f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Observed-vs-null randomness summary for the local Pythia-160M run.}",
            "\\label{tab:randomness-main}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_null_seed_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    fieldnames = [
        "null_pair",
        "lens",
        "regime",
        "mean_null",
        "ci_low",
        "ci_high",
        "emb",
        "early",
        "mid",
        "late",
        "out",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_null_seed_table(rows: list[dict[str, float | str]], path: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllrrrrrrr}",
        "\\toprule",
        "Null pair & Lens & Regime & $\\overline{M}_{\\mathrm{null}}$ & 95\\% CI & Emb. & Early & Mid & Late & Out \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['null_pair']} & {row['lens']} & {row['regime']} & "
            f"{float(row['mean_null']):.4f} & "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] & "
            f"{float(row['emb']):.4f} & "
            f"{float(row['early']):.4f} & "
            f"{float(row['mid']):.4f} & "
            f"{float(row['late']):.4f} & "
            f"{float(row['out']):.4f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Architecture-matched random seed-pair null summaries for the local Pythia-160M run. Each row reports the random-model pair trajectory used to build the null family for the observed-vs-null table.}",
            "\\label{tab:randomness-null-seeds}",
            "\\end{table*}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local-safe Pythia-160M randomness paper tables from existing 100-prompt bundles."
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for all shared prompts.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    REDUCED_DIR.mkdir(parents=True, exist_ok=True)

    manifests = {spec.name: _shard_bundle(spec) for spec in BUNDLES}
    shared_keys = sorted(set.intersection(*(set(v) for v in manifests.values())))
    if not shared_keys:
        raise RuntimeError("No shared prompt keys across Pythia-160M bundles.")
    if args.max_prompts is not None:
        if args.max_prompts <= 0:
            raise ValueError("--max-prompts must be positive when provided")
        shared_keys = shared_keys[: args.max_prompts]
    print(f"[run] using {len(shared_keys)} shared prompts", flush=True)

    random_pairs = [(RANDOM_NAMES[i], RANDOM_NAMES[j]) for i in range(len(RANDOM_NAMES)) for j in range(i + 1, len(RANDOM_NAMES))]
    table_rows: list[dict[str, float | str]] = []
    null_seed_rows: list[dict[str, float | str]] = []
    random_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, list[str]]] = {}

    for spec in COMPARISONS:
        comparison_dir = OUT_DIR / spec.slug
        comparison_dir.mkdir(parents=True, exist_ok=True)
        prompt_rows_path = comparison_dir / "randomness_prompt_rows.jsonl"
        prompt_rows_path.write_text("", encoding="utf-8")
        for readout, _readout_label in READOUTS:
            for regime, _regime_label in REGIMES:
                print(f"[run] {spec.slug}: {readout}/{regime}", flush=True)
                trained_curves, labels, keys = _load_or_compute_mean_curves(
                    f"observed__{spec.slug}",
                    [(spec.comparison_name, spec.baseline_name)],
                    shared_keys,
                    readout=readout,
                    regime=regime,
                )
                cache_key = (readout, regime)
                if cache_key not in random_cache:
                    random_pair_curves, random_labels, _ = _load_or_compute_pair_curves(
                        "null_seed_pairs",
                        random_pairs,
                        shared_keys,
                        readout=readout,
                        regime=regime,
                    )
                    random_curves_cached = random_pair_curves.mean(dim=1)
                    random_cache[cache_key] = (random_pair_curves, random_curves_cached, random_labels)
                    for pair_idx, pair in enumerate(random_pairs):
                        null_seed_rows.append(
                            _null_pair_stats(
                                NULL_PAIR_LABELS[pair],
                                dict(READOUTS)[readout],
                                dict(REGIMES)[regime],
                                random_labels,
                                random_pair_curves[:, pair_idx, :],
                            )
                        )
                random_pair_curves, random_curves, _ = random_cache[cache_key]
                gap_curves = trained_curves - random_curves
                table_rows.append(
                    _row_stats(
                        spec.label,
                        dict(READOUTS)[readout],
                        dict(REGIMES)[regime],
                        trained_curves,
                        random_curves,
                        gap_curves.mean(dim=0),
                        labels,
                        gap_curves,
                    )
                )
                with prompt_rows_path.open("a", encoding="utf-8") as handle:
                    for prompt_key, prompt_curve in zip(keys, gap_curves):
                        handle.write(
                            json.dumps(
                                {
                                    "comparison": spec.label,
                                    "lens": dict(READOUTS)[readout],
                                    "regime": dict(REGIMES)[regime],
                                    "prompt_key": prompt_key,
                                    "mean_abs_delta": float(torch.nanmean(prompt_curve.abs()).item()),
                                    "signed_mean_delta": float(torch.nanmean(prompt_curve).item()),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                del trained_curves
                del gap_curves
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    _add_bh_q_values(table_rows)
    _write_jsonl(table_rows, OUT_DIR / "randomness_main_table.jsonl")
    _write_csv(table_rows, OUT_DIR / "randomness_main_table.csv")
    _write_main_table(table_rows, OUT_DIR / "randomness_main_table.tex")
    _write_jsonl(null_seed_rows, OUT_DIR / "randomness_null_seed_table.jsonl")
    _write_null_seed_csv(null_seed_rows, OUT_DIR / "randomness_null_seed_table.csv")
    _write_null_seed_table(null_seed_rows, OUT_DIR / "randomness_null_seed_table.tex")
    print(f"[done] wrote {OUT_DIR / 'randomness_main_table.tex'}", flush=True)


if __name__ == "__main__":
    main()

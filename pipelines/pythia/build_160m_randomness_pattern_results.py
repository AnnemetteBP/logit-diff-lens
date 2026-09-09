from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "tmp" / "pythia_160m_randomness_local_safe"
CURVE_DIR = DATA_DIR / "reduced_curves"
FIG_DIR = ROOT / "Figures" / "RealResults" / "randomness_pythia160m_pattern"
RNG_SEED = 17
N_BOOT = 2000
N_PERM = 5000


COMPARISONS = [
    ("160M 1k--143k", "160m_1k_vs_143k"),
    ("160M 71k--143k", "160m_71k_vs_143k"),
]
LENSES = [("R", "raw"), ("MN", "model_norm")]
REGIMES = [("NC", "non_calibrated"), ("C", "calibrated")]


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(unique) != len(x):
        for group in range(len(unique)):
            idx = inverse == group
            ranks[idx] = ranks[idx].mean()
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _prompt_distance_samples(obs_curves: np.ndarray, null_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    null_mean = null_curves.mean(axis=(0, 1))
    obs_dist = np.sqrt(np.mean((obs_curves[:, 0, :] - null_mean) ** 2, axis=1))
    null_dist = np.sqrt(np.mean((null_curves - null_mean[None, None, :]) ** 2, axis=2)).reshape(-1)
    return obs_dist, null_dist


def _bootstrap_profile_stats(
    obs_curves: np.ndarray,
    null_curves: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict[str, tuple[float, float]]:
    n_prompts = obs_curves.shape[0]
    r_vals: list[float] = []
    rho_vals: list[float] = []
    d_vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        obs_profile = obs_curves[idx, 0, :].mean(axis=0)
        null_profile = null_curves[idx, :, :].mean(axis=(0, 1))
        r_vals.append(_pearson(obs_profile, null_profile))
        rho_vals.append(_spearman(obs_profile, null_profile))
        d_vals.append(_rmse(obs_profile, null_profile))
    return {
        "r_p": _ci(np.asarray(r_vals, dtype=float)),
        "rho_s": _ci(np.asarray(rho_vals, dtype=float)),
        "d_obs": _ci(np.asarray(d_vals, dtype=float)),
    }


def _permutation_pvalue(
    obs_dist: np.ndarray,
    null_dist: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> float:
    observed = float(obs_dist.mean() - null_dist.mean())
    pooled = np.concatenate([obs_dist, null_dist])
    n_obs = len(obs_dist)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        stat = float(perm[:n_obs].mean() - perm[n_obs:].mean())
        if stat >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def _block_indices(layer_labels: list[str]) -> dict[str, list[int]]:
    n = len(layer_labels)
    transformer = list(range(1, n - 1))
    chunks = np.array_split(transformer, 3)
    return {
        "emb": [0],
        "early": [int(i) for i in chunks[0]],
        "mid": [int(i) for i in chunks[1]],
        "late": [int(i) for i in chunks[2]],
        "out": [n - 1],
    }


def _block_means(profile: np.ndarray, layer_labels: list[str]) -> dict[str, float]:
    blocks = _block_indices(layer_labels)
    return {name: float(np.mean(profile[idx])) for name, idx in blocks.items()}


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fmt(x: object, digits: int = 3) -> str:
    if isinstance(x, float):
        if np.isnan(x):
            return "--"
        return f"{x:.{digits}f}"
    return str(x)


def _write_pattern_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        r"Cmp. & Lens & Regime & $r_{\mathrm{P}}$ & 95\% CI & $\rho_{\mathrm{S}}$ & 95\% CI & $d_{\mathrm{obs}}$ & 95\% CI & $p_{\mathrm{perm}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison']} & {row['lens']} & {row['regime']} & "
            f"{_fmt(row['r_p'])} & [{_fmt(row['r_p_ci_low'])}, {_fmt(row['r_p_ci_high'])}] & "
            f"{_fmt(row['rho_s'])} & [{_fmt(row['rho_s_ci_low'])}, {_fmt(row['rho_s_ci_high'])}] & "
            f"{_fmt(row['d_obs'])} & [{_fmt(row['d_obs_ci_low'])}, {_fmt(row['d_obs_ci_high'])}] & "
            f"{_fmt(row['p_perm'], 4)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Layer-wise randomness pattern test for the Pythia-160M analysis. $r_{\mathrm{P}}$ and $\rho_{\mathrm{S}}$ compare the observed layer-wise LogitDiff profile with the mean architecture-matched random seed-pair profile under the same lens and measurement regime. $d_{\mathrm{obs}}$ is the root-mean-square distance from the observed profile to the random null mean. Confidence intervals are prompt-bootstrap intervals. $p_{\mathrm{perm}}$ is a prompt-level permutation test comparing observed profile distances against random seed-pair profile distances.}",
            r"\label{tab:randomness-pattern-test}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_seed_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Seed pair & Lens & Regime & $\bar{D}_{\mathrm{null}}$ & Emb. & Early & Mid & Late & Out \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['seed_pair']} & {row['lens']} & {row['regime']} & "
            f"{_fmt(row['mean_d'])} & {_fmt(row['emb'])} & {_fmt(row['early'])} & "
            f"{_fmt(row['mid'])} & {_fmt(row['late'])} & {_fmt(row['out'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Random seed-pair null profiles for the Pythia-160M randomness analysis. Each row reports the same layer-wise LogitDiff metric, lens, regime, and layer partitions used by the observed comparisons, but computed between two independently initialized architecture-matched random models.}",
            r"\label{tab:randomness-null-profiles}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_profile_figure(records: list[dict[str, object]], layer_labels: list[str]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.8,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x = np.arange(len(layer_labels))
    tick_positions = [0, 1, 4, 8, 12, 13]
    tick_labels = ["Emb", "L1", "L4", "L8", "L12", "Out"]
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 7.4), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.955, bottom=0.075, wspace=0.17, hspace=0.42)

    for ax, rec in zip(axes.ravel(), records):
        obs = rec["obs_profile"]
        null_profiles = rec["null_profiles"]
        null_mean = null_profiles.mean(axis=0)
        null_low = np.percentile(null_profiles, 2.5, axis=0)
        null_high = np.percentile(null_profiles, 97.5, axis=0)

        ax.fill_between(x, null_low, null_high, color="#d7ecf4", alpha=0.90, linewidth=0, label="Random null band")
        for profile in null_profiles:
            ax.plot(x, profile, color="#6e9fb4", alpha=0.45, linewidth=1.0)
        ax.plot(x, null_mean, color="#2f6f8f", linewidth=1.35, linestyle="--", label="Random null mean")
        ax.plot(x, obs, color="#b94b5f", linewidth=1.85, label="Observed")

        ax.set_title(f"{rec['comparison_short']} {rec['lens']} {rec['regime']}", loc="left", fontweight=600, pad=3)
        ax.grid(color="#e6e1dd", linewidth=0.65, alpha=0.9)
        ax.set_xlim(0, len(layer_labels) - 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontweight=600)
        ax.text(
            0.98,
            0.94,
            rf"$r_P={rec['r_p']:.2f}$, $\rho_S={rec['rho_s']:.2f}$" + "\n" + rf"$p={rec['p_perm']:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.6,
            color="#252525",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#d5d0cc", "linewidth": 0.55, "alpha": 0.86},
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#4b4b4b")

    for ax in axes[:, 0]:
        ax.set_ylabel("Mean JSD")
    for ax in axes[-1, :]:
        ax.set_xlabel("Layer")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.995), ncol=3, frameon=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "randomness_pattern_profiles_pythia160m"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pattern_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    figure_records: list[dict[str, object]] = []
    layer_labels: list[str] | None = None

    for lens_label, lens_slug in LENSES:
        for regime_label, regime_slug in REGIMES:
            null_obj = _load(CURVE_DIR / f"null_seed_pairs__{lens_slug}__{regime_slug}.pt")
            null_curves = null_obj["curves"].detach().cpu().numpy()
            null_profiles = null_curves.mean(axis=0)
            pair_names = [f"{a}--{b}" for a, b in null_obj["pair_names"]]
            labels = list(null_obj["layer_labels"])
            if layer_labels is None:
                layer_labels = labels

            for pair_name, profile in zip(pair_names, null_profiles):
                row = {
                    "seed_pair": pair_name,
                    "lens": lens_label,
                    "regime": regime_label,
                    "mean_d": float(np.mean(profile)),
                }
                row.update(_block_means(profile, labels))
                seed_rows.append(row)

            null_mean = null_profiles.mean(axis=0)
            null_dist = []
            for idx, profile in enumerate(null_profiles):
                others = np.delete(null_profiles, idx, axis=0)
                null_dist.append(_rmse(profile, others.mean(axis=0)))
            null_dist_arr = np.array(null_dist, dtype=float)

            for comparison_label, comparison_slug in COMPARISONS:
                obs_obj = _load(CURVE_DIR / f"observed__{comparison_slug}__{lens_slug}__{regime_slug}.pt")
                obs_curves = obs_obj["curves"].detach().cpu().numpy()
                obs_profile = obs_curves.mean(axis=(0, 1))
                d_obs = _rmse(obs_profile, null_mean)
                obs_dist_samples, null_dist_samples = _prompt_distance_samples(obs_curves, null_curves)
                boot = _bootstrap_profile_stats(obs_curves, null_curves, rng)
                p_perm = _permutation_pvalue(obs_dist_samples, null_dist_samples, rng)
                r_p = _pearson(obs_profile, null_mean)
                rho_s = _spearman(obs_profile, null_mean)
                row = {
                    "comparison": comparison_label,
                    "lens": lens_label,
                    "regime": regime_label,
                    "metric": "JSD",
                    "r_p": r_p,
                    "r_p_ci_low": boot["r_p"][0],
                    "r_p_ci_high": boot["r_p"][1],
                    "rho_s": rho_s,
                    "rho_s_ci_low": boot["rho_s"][0],
                    "rho_s_ci_high": boot["rho_s"][1],
                    "d_obs": d_obs,
                    "d_obs_ci_low": boot["d_obs"][0],
                    "d_obs_ci_high": boot["d_obs"][1],
                    "null_d_low": float(np.percentile(null_dist_arr, 2.5)),
                    "null_d_high": float(np.percentile(null_dist_arr, 97.5)),
                    "null_prompt_d_mean": float(null_dist_samples.mean()),
                    "null_prompt_d_ci_low": _ci(null_dist_samples)[0],
                    "null_prompt_d_ci_high": _ci(null_dist_samples)[1],
                    "p_perm": p_perm,
                }
                pattern_rows.append(row)
                figure_records.append(
                    {
                        **row,
                        "comparison_short": comparison_label.replace("160M ", ""),
                        "obs_profile": obs_profile,
                        "null_profiles": null_profiles,
                    }
                )

    assert layer_labels is not None
    pattern_fields = [
        "comparison",
        "lens",
        "regime",
        "metric",
        "r_p",
        "r_p_ci_low",
        "r_p_ci_high",
        "rho_s",
        "rho_s_ci_low",
        "rho_s_ci_high",
        "d_obs",
        "d_obs_ci_low",
        "d_obs_ci_high",
        "null_d_low",
        "null_d_high",
        "null_prompt_d_mean",
        "null_prompt_d_ci_low",
        "null_prompt_d_ci_high",
        "p_perm",
    ]
    seed_fields = ["seed_pair", "lens", "regime", "mean_d", "emb", "early", "mid", "late", "out"]
    _write_csv(DATA_DIR / "randomness_pattern_test_table.csv", pattern_rows, pattern_fields)
    _write_jsonl(DATA_DIR / "randomness_pattern_test_table.jsonl", pattern_rows)
    _write_pattern_tex(DATA_DIR / "randomness_pattern_test_table.tex", pattern_rows)
    _write_csv(DATA_DIR / "randomness_null_profile_table.csv", seed_rows, seed_fields)
    _write_jsonl(DATA_DIR / "randomness_null_profile_table.jsonl", seed_rows)
    _write_seed_tex(DATA_DIR / "randomness_null_profile_table.tex", seed_rows)
    _make_profile_figure(figure_records, layer_labels)

    print(FIG_DIR / "randomness_pattern_profiles_pythia160m.pdf")
    print(FIG_DIR / "randomness_pattern_profiles_pythia160m.png")
    print(DATA_DIR / "randomness_pattern_test_table.tex")
    print(DATA_DIR / "randomness_null_profile_table.tex")


if __name__ == "__main__":
    main()

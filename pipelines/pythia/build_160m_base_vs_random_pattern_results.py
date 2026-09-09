from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "tmp" / "pythia_160m_randomness_local_safe"
SHARD_DIR = DATA_DIR / "prompt_shards"
OUT_DIR = DATA_DIR / "base_vs_random_pattern"
FIG_DIR = ROOT / "Figures" / "RealResults" / "randomness_pythia160m_base_vs_random"

BASE = "step143000"
REAL_MODELS = [("160M 1k--143k", "step1000"), ("160M 71k--143k", "step71000")]
RANDOM_MODELS = ["seed123", "seed456", "seed789"]
LENSES = [("R", "logits_raw"), ("MN", "logits_model_norm")]
TOPK = [1, 5, 10]
N_BOOT = 2000
N_PERM = 5000
RNG_SEED = 23


def _prompt_ids(model: str) -> list[str]:
    return sorted(p.stem for p in (SHARD_DIR / model).glob("*.pt") if p.stem.isdigit())


def _load(model: str, prompt_id: str) -> dict:
    return torch.load(SHARD_DIR / model / f"{prompt_id}.pt", map_location="cpu")


def _softmax_np(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits.float(), dim=-1).cpu().numpy()


def _jsd_from_probs(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * (np.log(p) - np.log(m)), axis=-1) + 0.5 * np.sum(q * (np.log(q) - np.log(m)), axis=-1)


def _topk_disagreement_from_logits(a: torch.Tensor, b: torch.Tensor, k: int) -> np.ndarray:
    a_idx = torch.topk(a.float(), k=k, dim=-1).indices.cpu().numpy()
    b_idx = torch.topk(b.float(), k=k, dim=-1).indices.cpu().numpy()
    out = np.zeros(a_idx.shape[:-1], dtype=np.float32)
    for idx in np.ndindex(out.shape):
        sa = set(int(x) for x in a_idx[idx])
        sb = set(int(x) for x in b_idx[idx])
        out[idx] = 1.0 - (len(sa & sb) / max(1, len(sa | sb)))
    return out


def _layer_curve(base_artifact: dict, other_artifact: dict, lens_key: str) -> dict[str, np.ndarray | list[str]]:
    layer_labels: list[str] = []
    jsd: list[float] = []
    topk: dict[int, list[float]] = {k: [] for k in TOPK}
    for base_rec, other_rec in zip(base_artifact["layer_records"], other_artifact["layer_records"]):
        layer_labels.append(str(base_rec["layer_name"]).replace("embedding", "Emb").replace("output", "Out"))
        base_logits = base_rec[lens_key]
        other_logits = other_rec[lens_key]
        mask = base_rec["attention_mask"].bool().cpu().numpy()[0]
        if mask.size > 1:
            mask[-1] = False
        valid = np.where(mask)[0]
        if valid.size == 0:
            valid = np.arange(base_logits.shape[1])
        p = _softmax_np(base_logits[0, valid, :])
        q = _softmax_np(other_logits[0, valid, :])
        jsd.append(float(_jsd_from_probs(p, q).mean()))
        for k in TOPK:
            topk[k].append(float(_topk_disagreement_from_logits(base_logits[0, valid, :], other_logits[0, valid, :], k).mean()))
    result: dict[str, np.ndarray | list[str]] = {"layer_labels": layer_labels, "JSD": np.asarray(jsd, dtype=np.float32)}
    for k in TOPK:
        result[f"J@{k}_dist"] = np.asarray(topk[k], dtype=np.float32)
    return result


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    _, inverse = np.unique(x, return_inverse=True)
    for group in range(inverse.max() + 1):
        idx = inverse == group
        if idx.sum() > 1:
            ranks[idx] = ranks[idx].mean()
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _diff_corrs(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    da = np.diff(a)
    db = np.diff(b)
    return _pearson(da, db), _spearman(da, db)


def _ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _bootstrap_stats(obs: np.ndarray, rand: np.ndarray, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    n = obs.shape[0]
    rp, rs, rp_diff, rs_diff, dist, mean_real, mean_rand, mean_gap = [], [], [], [], [], [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        obs_profile = obs[idx].mean(axis=0)
        rand_profile = rand[idx].mean(axis=(0, 1))
        rp.append(_pearson(obs_profile, rand_profile))
        rs.append(_spearman(obs_profile, rand_profile))
        diff_rp, diff_rs = _diff_corrs(obs_profile, rand_profile)
        rp_diff.append(diff_rp)
        rs_diff.append(diff_rs)
        dist.append(_rmse(obs_profile, rand_profile))
        mean_real.append(float(obs_profile.mean()))
        mean_rand.append(float(rand_profile.mean()))
        mean_gap.append(float(obs_profile.mean() - rand_profile.mean()))
    return {
        "r_p": _ci(np.asarray(rp)),
        "rho_s": _ci(np.asarray(rs)),
        "r_p_diff": _ci(np.asarray(rp_diff)),
        "rho_s_diff": _ci(np.asarray(rs_diff)),
        "d": _ci(np.asarray(dist)),
        "mean_real": _ci(np.asarray(mean_real)),
        "mean_rand": _ci(np.asarray(mean_rand)),
        "mean_gap": _ci(np.asarray(mean_gap)),
    }


def _permutation_p(obs_prompt_dist: np.ndarray, rand_prompt_dist: np.ndarray, rng: np.random.Generator) -> float:
    observed = float(obs_prompt_dist.mean() - rand_prompt_dist.mean())
    pooled = np.concatenate([obs_prompt_dist, rand_prompt_dist])
    n_obs = len(obs_prompt_dist)
    count = 0
    for _ in range(N_PERM):
        perm = rng.permutation(pooled)
        stat = float(perm[:n_obs].mean() - perm[n_obs:].mean())
        if stat >= observed:
            count += 1
    return float((count + 1) / (N_PERM + 1))


def _sign_flip_pvalue(diff: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(diff.mean()))
    count = 0
    for _ in range(N_PERM):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(diff), replace=True)
        stat = abs(float((diff * signs).mean()))
        if stat >= observed:
            count += 1
    return float((count + 1) / (N_PERM + 1))


def _bootstrap_seed_gap(
    obs_prompt_curves: np.ndarray,
    rand_seed_prompt_curves: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = obs_prompt_curves.shape[0]
    vals: list[float] = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        vals.append(float(obs_prompt_curves[idx].mean() - rand_seed_prompt_curves[idx].mean()))
    return _ci(np.asarray(vals, dtype=float))


def _block_indices(n_layers: int) -> dict[str, list[int]]:
    transformer = list(range(1, n_layers - 1))
    chunks = np.array_split(transformer, 3)
    return {
        "emb": [0],
        "early": [int(i) for i in chunks[0]],
        "mid": [int(i) for i in chunks[1]],
        "late": [int(i) for i in chunks[2]],
        "out": [n_layers - 1],
    }


def _block_means(profile: np.ndarray) -> dict[str, float]:
    return {k: float(profile[idx].mean()) for k, idx in _block_indices(len(profile)).items()}


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fmt(x: object, digits: int = 3) -> str:
    if isinstance(x, float):
        return f"{x:.{digits}f}" if np.isfinite(x) else "--"
    return str(x)


def _write_pattern_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lllrrrrrrrrrrr}",
        r"\toprule",
        r"Cmp. & Lens & Metric & $\bar{D}_{\mathrm{real}}$ & $\bar{D}_{\mathrm{rand}}$ & $\Delta$ & 95\% CI$_{\Delta}$ & $r_{\mathrm{P}}$ & $\rho_{\mathrm{S}}$ & $r_{\mathrm{P}}^{\nabla}$ & $\rho_{\mathrm{S}}^{\nabla}$ & $d_{\mathrm{prof}}$ & $p_{\mathrm{perm}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison']} & {row['lens']} & {row['metric']} & "
            f"{_fmt(row['mean_real'])} & {_fmt(row['mean_rand'])} & {_fmt(row['mean_gap'])} & "
            f"[{_fmt(row['mean_gap_ci_low'])}, {_fmt(row['mean_gap_ci_high'])}] & "
            f"{_fmt(row['r_p'])} & {_fmt(row['rho_s'])} & {_fmt(row['r_p_diff'])} & {_fmt(row['rho_s_diff'])} & {_fmt(row['d_obs'])} & "
            f"{_fmt(row['p_perm'], 4)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Base-vs-random layer-wise pattern test. $\bar{D}_{\mathrm{real}}$ is the mean base-vs-real divergence/disagreement profile and $\bar{D}_{\mathrm{rand}}$ is the mean base-vs-random profile under the same prompts, lens, metric, and layer axis. $\Delta=\bar{D}_{\mathrm{real}}-\bar{D}_{\mathrm{rand}}$ reports whether the real comparison is above or below the random replacement baseline, with prompt-bootstrap confidence intervals. $r_{\mathrm{P}}$ and $\rho_{\mathrm{S}}$ compare the layer-wise profile shapes, while $r_{\mathrm{P}}^{\nabla}$ and $\rho_{\mathrm{S}}^{\nabla}$ compare first-difference profiles across adjacent layers. $d_{\mathrm{prof}}$ is the profile distance from base-vs-real to the base-vs-random mean, and $p_{\mathrm{perm}}$ is a prompt-level permutation test comparing base-vs-real distances against base-vs-random distances.}",
            r"\label{tab:randomness-base-vs-random-pattern}",
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
        r"Random model & Lens & Metric & $\bar{D}_{\mathrm{base,rand}}$ & Emb. & Early & Mid & Late & Out \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['random_model']} & {row['lens']} & {row['metric']} & "
            f"{_fmt(row['mean_d'])} & {_fmt(row['emb'])} & {_fmt(row['early'])} & {_fmt(row['mid'])} & "
            f"{_fmt(row['late'])} & {_fmt(row['out'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Base-vs-random null profiles. Each row compares the same Pythia-160M base checkpoint against one independently initialized random model under the same prompt set, lens, metric, and layer partitioning used for the observed base-vs-real comparisons.}",
            r"\label{tab:randomness-base-vs-random-seeds}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_seed_comparison_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llllrrrrrrrr}",
        r"\toprule",
        r"Cmp. & Random & Lens & Metric & $\bar{D}_{\mathrm{real}}$ & $\bar{D}_{\mathrm{rand}}$ & $\Delta$ & 95\% CI$_{\Delta}$ & $r_{\mathrm{P}}$ & $\rho_{\mathrm{S}}$ & $d_{\mathrm{prof}}$ & $p_{\mathrm{perm}}$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison']} & {row['random_model']} & {row['lens']} & {row['metric']} & "
            f"{_fmt(row['mean_real'])} & {_fmt(row['mean_rand'])} & {_fmt(row['mean_gap'])} & "
            f"[{_fmt(row['mean_gap_ci_low'])}, {_fmt(row['mean_gap_ci_high'])}] & "
            f"{_fmt(row['r_p'])} & {_fmt(row['rho_s'])} & {_fmt(row['d_prof'])} & {_fmt(row['p_perm'], 4)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Per-seed base-vs-random comparison table. Each row compares one base-vs-real profile against one base-vs-random profile under the same prompt set, lens, metric, and layer axis. $\Delta=\bar{D}_{\mathrm{real}}-\bar{D}_{\mathrm{rand}}$ has prompt-bootstrap confidence intervals. $r_{\mathrm{P}}$, $\rho_{\mathrm{S}}$, and $d_{\mathrm{prof}}$ compare the layer-wise profile shapes, and $p_{\mathrm{perm}}$ is a paired prompt-level sign-flip test for the real-minus-random mean gap.}",
            r"\label{tab:randomness-base-vs-random-per-seed}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_block_gap_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Cmp. & Lens & Metric & $\Delta_{\mathrm{total}}$ & Emb. & Early & Mid & Late & Out \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['comparison']} & {row['lens']} & {row['metric']} & "
            f"{_fmt(row['gap_total'])} & {_fmt(row['gap_emb'])} & {_fmt(row['gap_early'])} & "
            f"{_fmt(row['gap_mid'])} & {_fmt(row['gap_late'])} & {_fmt(row['gap_out'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Blockwise base-vs-random gap summary. Each entry is the base-vs-real profile minus the mean base-vs-random profile for the same lens, metric, prompts, and layer partition. Positive values indicate that the real comparison is more divergent than the random replacement baseline in that layer block; negative values indicate that the random replacement baseline is more divergent.}",
            r"\label{tab:randomness-base-vs-random-block-gaps}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(records: list[dict[str, object]], layer_labels: list[str]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10.4,
            "axes.titlesize": 11.2,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 9.4,
            "legend.fontsize": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plot_records = [r for r in records if r["metric"] == "JSD"]
    x = np.arange(len(layer_labels))
    tick_positions = [0, 1, 4, 8, 12, 13]
    tick_labels = ["Emb", "1", "4", "8", "12", "Out"]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.95), dpi=300, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.995, top=0.855, bottom=0.185, wspace=0.055, hspace=0.24)
    global_max = max(float(np.max(r["rand_profiles"])) for r in plot_records)
    global_max = max(global_max, max(float(np.max(r["obs_profile"])) for r in plot_records))
    for panel_idx, (ax, rec) in enumerate(zip(axes.ravel(), plot_records)):
        obs = rec["obs_profile"]
        rand = rec["rand_profiles"]
        rand_mean = rand.mean(axis=0)
        rand_std = rand.std(axis=0)
        ax.fill_between(
            x,
            rand_mean - rand_std,
            rand_mean + rand_std,
            color="#d7ecf4",
            alpha=0.70,
            linewidth=0,
            label=r"143k--Rand. seeds $\mu\pm\sigma$",
        )
        ax.plot(
            x,
            rand_mean,
            color="#2f6f8f",
            linewidth=1.55,
            linestyle="--",
            label=r"143k--Rand. seeds $\mu$",
        )
        ax.fill_between(
            x,
            obs,
            rand_mean,
            color="#efb6ad",
            alpha=0.28,
            linewidth=0,
            label="Gap",
        )
        ax.plot(x, obs, color="#b94b5f", linewidth=2.05, label="143k--real")
        comparison_title = rec["comparison_short"].replace("--143k", "")
        ax.set_title(f"{comparison_title} {rec['lens']} JSD", loc="left", fontweight=600, pad=3)
        ax.grid(color="#e6e1dd", linewidth=0.65, alpha=0.9)
        ax.set_xlim(0, len(layer_labels) - 1)
        ax.set_ylim(-0.02, global_max * 1.08)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontweight="normal", rotation=0, ha="center")
        for tick in ax.get_xticklabels():
            text = tick.get_text()
            if text == "Emb":
                tick.set_rotation(90)
                tick.set_ha("center")
                tick.set_va("top")
            elif text == "Out":
                tick.set_rotation(90)
                tick.set_ha("center")
                tick.set_va("top")
        is_top = panel_idx < 2
        box_x = 0.03
        box_y = 0.94 if is_top else 0.06
        box_va = "top" if is_top else "bottom"
        ax.text(
            box_x,
            box_y,
            rf"$\bar{{\Delta}}={rec['mean_gap']:.2f}$, $d_{{\mathrm{{prof}}}}={rec['d_obs']:.2f}$"
            + "\n"
            + rf"$r_P={rec['r_p']:.2f}$, $\rho_S={rec['rho_s']:.2f}$"
            + "\n"
            + rf"$p_{{\mathrm{{perm}}}}={rec['p_perm']:.4f}$",
            transform=ax.transAxes,
            ha="left",
            va=box_va,
            fontsize=9.0,
            color="#252525",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d5d0cc", "linewidth": 0.55, "alpha": 0.86},
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#4b4b4b")
    fig.supylabel(r"$\mu\,\mathrm{JSD}(\ell)$", x=0.015, fontweight=700)
    fig.supxlabel("Layer", y=0.055, fontweight=600)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=0.75,
        fontsize=11.5,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "base_vs_random_pattern_pythia160m"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_ids = sorted(set(_prompt_ids(BASE)).intersection(*[_prompt_ids(m) for _, m in REAL_MODELS], *[_prompt_ids(m) for m in RANDOM_MODELS]), key=int)
    rng = np.random.default_rng(RNG_SEED)
    curves: dict[tuple[str, str, str], list[np.ndarray]] = {}
    layer_labels: list[str] | None = None

    for prompt_id in prompt_ids:
        base_art = _load(BASE, prompt_id)
        others = {model: _load(model, prompt_id) for _, model in REAL_MODELS}
        others.update({model: _load(model, prompt_id) for model in RANDOM_MODELS})
        for lens_label, lens_key in LENSES:
            for model, artifact in others.items():
                curve = _layer_curve(base_art, artifact, lens_key)
                if layer_labels is None:
                    layer_labels = list(curve["layer_labels"])  # type: ignore[arg-type]
                for metric in ["JSD", "J@1_dist", "J@5_dist", "J@10_dist"]:
                    curves.setdefault((model, lens_label, metric), []).append(np.asarray(curve[metric], dtype=np.float32))

    assert layer_labels is not None
    curves_arr = {k: np.stack(v, axis=0) for k, v in curves.items()}
    pattern_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    seed_comparison_rows: list[dict[str, object]] = []
    block_gap_rows: list[dict[str, object]] = []
    figure_records: list[dict[str, object]] = []

    for lens_label, _ in LENSES:
        for metric in ["JSD", "J@1_dist", "J@5_dist", "J@10_dist"]:
            rand_prompt_curves = np.stack([curves_arr[(seed, lens_label, metric)] for seed in RANDOM_MODELS], axis=1)
            rand_profiles = rand_prompt_curves.mean(axis=0)
            rand_mean = rand_profiles.mean(axis=0)
            for seed, profile in zip(RANDOM_MODELS, rand_profiles):
                row = {"random_model": seed, "lens": lens_label, "metric": metric.replace("_dist", ""), "mean_d": float(profile.mean())}
                row.update(_block_means(profile))
                seed_rows.append(row)
            rand_prompt_dist = np.sqrt(np.mean((rand_prompt_curves - rand_mean[None, None, :]) ** 2, axis=2)).reshape(-1)
            for comparison_label, real_model in REAL_MODELS:
                obs_prompt_curves = curves_arr[(real_model, lens_label, metric)]
                obs_profile = obs_prompt_curves.mean(axis=0)
                rand_mean_profile = rand_profiles.mean(axis=0)
                obs_prompt_dist = np.sqrt(np.mean((obs_prompt_curves - rand_mean[None, :]) ** 2, axis=1))
                boot = _bootstrap_stats(obs_prompt_curves, rand_prompt_curves, rng)
                p_perm = _permutation_p(obs_prompt_dist, rand_prompt_dist, rng)
                mean_real = float(obs_profile.mean())
                mean_rand = float(rand_mean_profile.mean())
                mean_gap = mean_real - mean_rand
                r_p_diff, rho_s_diff = _diff_corrs(obs_profile, rand_mean)
                block_gap = _block_means(obs_profile - rand_mean_profile)
                block_gap_rows.append(
                    {
                        "comparison": comparison_label,
                        "lens": lens_label,
                        "metric": metric.replace("_dist", ""),
                        "gap_total": mean_gap,
                        "gap_emb": block_gap["emb"],
                        "gap_early": block_gap["early"],
                        "gap_mid": block_gap["mid"],
                        "gap_late": block_gap["late"],
                        "gap_out": block_gap["out"],
                    }
                )
                for seed_index, seed in enumerate(RANDOM_MODELS):
                    seed_prompt_curves = rand_prompt_curves[:, seed_index, :]
                    seed_profile = seed_prompt_curves.mean(axis=0)
                    seed_gap_ci = _bootstrap_seed_gap(obs_prompt_curves, seed_prompt_curves, rng)
                    prompt_gap = obs_prompt_curves.mean(axis=1) - seed_prompt_curves.mean(axis=1)
                    seed_comparison_rows.append(
                        {
                            "comparison": comparison_label,
                            "random_model": seed,
                            "lens": lens_label,
                            "metric": metric.replace("_dist", ""),
                            "mean_real": float(obs_profile.mean()),
                            "mean_rand": float(seed_profile.mean()),
                            "mean_gap": float(obs_profile.mean() - seed_profile.mean()),
                            "mean_gap_ci_low": seed_gap_ci[0],
                            "mean_gap_ci_high": seed_gap_ci[1],
                            "r_p": _pearson(obs_profile, seed_profile),
                            "rho_s": _spearman(obs_profile, seed_profile),
                            "d_prof": _rmse(obs_profile, seed_profile),
                            "p_perm": _sign_flip_pvalue(prompt_gap, rng),
                        }
                    )
                row = {
                    "comparison": comparison_label,
                    "comparison_short": comparison_label.replace("160M ", ""),
                    "lens": lens_label,
                    "metric": metric.replace("_dist", ""),
                    "mean_real": mean_real,
                    "mean_real_ci_low": boot["mean_real"][0],
                    "mean_real_ci_high": boot["mean_real"][1],
                    "mean_rand": mean_rand,
                    "mean_rand_ci_low": boot["mean_rand"][0],
                    "mean_rand_ci_high": boot["mean_rand"][1],
                    "mean_gap": mean_gap,
                    "mean_gap_ci_low": boot["mean_gap"][0],
                    "mean_gap_ci_high": boot["mean_gap"][1],
                    "r_p": _pearson(obs_profile, rand_mean),
                    "r_p_ci_low": boot["r_p"][0],
                    "r_p_ci_high": boot["r_p"][1],
                    "rho_s": _spearman(obs_profile, rand_mean),
                    "rho_s_ci_low": boot["rho_s"][0],
                    "rho_s_ci_high": boot["rho_s"][1],
                    "r_p_diff": r_p_diff,
                    "r_p_diff_ci_low": boot["r_p_diff"][0],
                    "r_p_diff_ci_high": boot["r_p_diff"][1],
                    "rho_s_diff": rho_s_diff,
                    "rho_s_diff_ci_low": boot["rho_s_diff"][0],
                    "rho_s_diff_ci_high": boot["rho_s_diff"][1],
                    "d_obs": _rmse(obs_profile, rand_mean),
                    "d_ci_low": boot["d"][0],
                    "d_ci_high": boot["d"][1],
                    "p_perm": p_perm,
                    "obs_profile": obs_profile,
                    "rand_profiles": rand_profiles,
                }
                pattern_rows.append({k: v for k, v in row.items() if k not in {"obs_profile", "rand_profiles", "comparison_short"}})
                figure_records.append(row)

    pattern_fields = [
        "comparison",
        "lens",
        "metric",
        "mean_real",
        "mean_real_ci_low",
        "mean_real_ci_high",
        "mean_rand",
        "mean_rand_ci_low",
        "mean_rand_ci_high",
        "mean_gap",
        "mean_gap_ci_low",
        "mean_gap_ci_high",
        "r_p",
        "r_p_ci_low",
        "r_p_ci_high",
        "rho_s",
        "rho_s_ci_low",
        "rho_s_ci_high",
        "r_p_diff",
        "r_p_diff_ci_low",
        "r_p_diff_ci_high",
        "rho_s_diff",
        "rho_s_diff_ci_low",
        "rho_s_diff_ci_high",
        "d_obs",
        "d_ci_low",
        "d_ci_high",
        "p_perm",
    ]
    seed_fields = ["random_model", "lens", "metric", "mean_d", "emb", "early", "mid", "late", "out"]
    seed_comparison_fields = [
        "comparison",
        "random_model",
        "lens",
        "metric",
        "mean_real",
        "mean_rand",
        "mean_gap",
        "mean_gap_ci_low",
        "mean_gap_ci_high",
        "r_p",
        "rho_s",
        "d_prof",
        "p_perm",
    ]
    block_gap_fields = ["comparison", "lens", "metric", "gap_total", "gap_emb", "gap_early", "gap_mid", "gap_late", "gap_out"]
    _write_csv(OUT_DIR / "base_vs_random_pattern_test_table.csv", pattern_rows, pattern_fields)
    _write_jsonl(OUT_DIR / "base_vs_random_pattern_test_table.jsonl", pattern_rows)
    _write_pattern_tex(OUT_DIR / "base_vs_random_pattern_test_table.tex", pattern_rows)
    _write_csv(OUT_DIR / "base_vs_random_block_gap_table.csv", block_gap_rows, block_gap_fields)
    _write_jsonl(OUT_DIR / "base_vs_random_block_gap_table.jsonl", block_gap_rows)
    _write_block_gap_tex(OUT_DIR / "base_vs_random_block_gap_table.tex", block_gap_rows)
    _write_csv(OUT_DIR / "base_vs_random_seed_profile_table.csv", seed_rows, seed_fields)
    _write_jsonl(OUT_DIR / "base_vs_random_seed_profile_table.jsonl", seed_rows)
    _write_seed_tex(OUT_DIR / "base_vs_random_seed_profile_table.tex", seed_rows)
    _write_csv(OUT_DIR / "base_vs_random_per_seed_comparison_table.csv", seed_comparison_rows, seed_comparison_fields)
    _write_jsonl(OUT_DIR / "base_vs_random_per_seed_comparison_table.jsonl", seed_comparison_rows)
    _write_seed_comparison_tex(OUT_DIR / "base_vs_random_per_seed_comparison_table.tex", seed_comparison_rows)
    _plot(figure_records, layer_labels)
    print(FIG_DIR / "base_vs_random_pattern_pythia160m.pdf")
    print(FIG_DIR / "base_vs_random_pattern_pythia160m.png")
    print(OUT_DIR / "base_vs_random_pattern_test_table.tex")
    print(OUT_DIR / "base_vs_random_block_gap_table.tex")
    print(OUT_DIR / "base_vs_random_seed_profile_table.tex")
    print(OUT_DIR / "base_vs_random_per_seed_comparison_table.tex")


if __name__ == "__main__":
    main()

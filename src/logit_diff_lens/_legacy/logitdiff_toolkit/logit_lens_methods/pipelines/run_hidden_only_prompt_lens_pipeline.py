from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from diffing.logit_lens_methods.base_collector_scripts.prompt_lens.collect_prompt_lens_activations import run_collection
from diffing.logit_lens_methods.plotting.multi_plots.mode_specific_summary_cli import save_mode_specific_figures
from diffing.logit_lens_methods.prompt_lens.analysis_scripts.summarize_mode_specific import summarize_mode_specific


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


@dataclass
class UCloudPromptLensConfig:
    dataset_path: Path
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    force_cpu: bool = False
    base_model_revision: str | None = None
    comparison_model_revision: str | None = None
    base_tokenizer_revision: str | None = None
    comparison_tokenizer_revision: str | None = None
    data_dir: str = "data"
    summaries_dir: str = "summaries"
    figures_dir: str = "figures"
    base_output_stem: str = "base_qwen_prompt_lens"
    comparison_output_stem: str = "comparison_qwen_prompt_lens"
    base_reuse_path: Path | None = None
    reuse_existing_base: bool = True
    quantization_config_name: str | None = None
    quantization_config_source: str | None = None
    title_prefix: str = "Model comparison"
    table_label: str = "tab:prompt_lens_summary"
    enabled: bool = True


def _load_json_config(path: Path) -> dict[str, Any]:
    _require_path(path, "config")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_config(raw: dict[str, Any]) -> UCloudPromptLensConfig:
    cfg = raw["prompt_lens"]
    base_reuse_path = cfg.get("base_reuse_path")
    return UCloudPromptLensConfig(
        dataset_path=Path(cfg["dataset_path"]),
        dtype=cfg.get("dtype", "bfloat16"),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        force_cpu=bool(cfg.get("force_cpu", False)),
        base_model_revision=raw.get("base_model_revision"),
        comparison_model_revision=raw.get("comparison_model_revision"),
        base_tokenizer_revision=raw.get("base_tokenizer_revision"),
        comparison_tokenizer_revision=raw.get("comparison_tokenizer_revision"),
        data_dir=cfg.get("data_dir", "data"),
        summaries_dir=cfg.get("summaries_dir", "summaries"),
        figures_dir=cfg.get("figures_dir", "figures"),
        base_output_stem=cfg.get("base_output_stem", "base_qwen_prompt_lens"),
        comparison_output_stem=cfg.get("comparison_output_stem", "comparison_qwen_prompt_lens"),
        base_reuse_path=Path(base_reuse_path) if base_reuse_path else None,
        reuse_existing_base=bool(cfg.get("reuse_existing_base", True)),
        quantization_config_name=raw.get("quantization_config_name"),
        quantization_config_source=raw.get("quantization_config_source"),
        title_prefix=cfg.get("title_prefix", "Model comparison"),
        table_label=cfg.get("table_label", "tab:prompt_lens_summary"),
        enabled=bool(cfg.get("enabled", True)),
    )


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1)


def _jaccard_from_topk(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> torch.Tensor:
    a = torch.topk(logits_a, k=k, dim=-1).indices
    b = torch.topk(logits_b, k=k, dim=-1).indices
    out = []
    for row_a, row_b in zip(a.tolist(), b.tolist()):
        sa, sb = set(row_a), set(row_b)
        denom = len(sa | sb)
        out.append(0.0 if denom == 0 else len(sa & sb) / denom)
    return torch.tensor(out, dtype=torch.float32)


def _tvd(pa: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.abs(pa - pb).sum(dim=-1)


def _kl(pa: torch.Tensor, pb: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pa = pa.clamp_min(eps)
    pb = pb.clamp_min(eps)
    return (pa * (torch.log(pa) - torch.log(pb))).sum(dim=-1)


def _js(pa: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (pa + pb)
    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    an = torch.linalg.norm(a, dim=-1).clamp_min(1e-12)
    bn = torch.linalg.norm(b, dim=-1).clamp_min(1e-12)
    return (a * b).sum(dim=-1) / (an * bn)


def _l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(b - a, dim=-1)


def _prepare_projection_metadata(payload: dict[str, Any]) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    lm_head_weight = payload.get("lm_head_weight")
    final_norm = payload.get("final_norm")
    lm_head_tensor = None
    if torch.is_tensor(lm_head_weight):
        lm_head_tensor = lm_head_weight.detach().to(device="cpu", dtype=torch.float32)
    return lm_head_tensor, final_norm if isinstance(final_norm, dict) else None


def _apply_saved_final_norm(hidden: torch.Tensor, final_norm: dict[str, Any] | None) -> torch.Tensor:
    if final_norm is None:
        return hidden
    eps = float(final_norm.get("eps", 1e-5))
    mean = hidden.mean(dim=-1, keepdim=True)
    var = hidden.var(dim=-1, keepdim=True, unbiased=False)
    out = (hidden - mean) / torch.sqrt(var + eps)
    weight = final_norm.get("weight")
    bias = final_norm.get("bias")
    if torch.is_tensor(weight):
        out = out * weight.detach().to(device="cpu", dtype=torch.float32)
    if torch.is_tensor(bias):
        out = out + bias.detach().to(device="cpu", dtype=torch.float32)
    return out


def _resolve_logits(
    rec: dict[str, Any],
    *,
    mode: str,
    usable: int,
    lm_head_weight: torch.Tensor | None,
    final_norm: dict[str, Any] | None,
) -> torch.Tensor:
    direct = rec.get(f"logits_{mode}")
    if torch.is_tensor(direct):
        return direct[0, :usable, :].float()
    if lm_head_weight is None:
        raise ValueError(f"Missing persisted logits_{mode} and lm_head_weight needed for projection")
    hidden = rec["hidden"][0, :usable, :].float()
    if mode == "model_norm":
        hidden = _apply_saved_final_norm(hidden, final_norm)
    return hidden @ lm_head_weight.T


def _topk_next_token_accuracy(logits: torch.Tensor, target_ids: torch.Tensor, k: int) -> torch.Tensor:
    topk = torch.topk(logits, k=k, dim=-1).indices
    return (topk == target_ids.unsqueeze(-1)).any(dim=-1).to(dtype=torch.float32)


def _bootstrap_ci(values: list[float], num_bootstrap: int = 2000) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [float("nan"), float("nan")]
    if arr.size == 1:
        value = float(arr[0])
        return [value, value]
    rng = np.random.default_rng(0)
    samples = rng.choice(arr, size=(num_bootstrap, arr.size), replace=True)
    means = samples.mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _fisher_ci(corr: float, n: int) -> list[float]:
    if n <= 3 or not np.isfinite(corr) or abs(corr) >= 1.0:
        return [float("nan"), float("nan")]
    z = np.arctanh(corr)
    se = 1.0 / np.sqrt(n - 3)
    z_low = z - 1.96 * se
    z_high = z + 1.96 * se
    return [float(np.tanh(z_low)), float(np.tanh(z_high))]


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("group_id")), str(row.get("variant"))


def _write_observation_and_summary_outputs(
    *,
    base_path: Path,
    comparison_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base_payload = torch.load(base_path, map_location="cpu")
    comparison_payload = torch.load(comparison_path, map_location="cpu")
    base_lm_head_weight, base_final_norm = _prepare_projection_metadata(base_payload)
    comparison_lm_head_weight, comparison_final_norm = _prepare_projection_metadata(comparison_payload)

    base_rows = {_row_key(row): row for row in base_payload["rows"]}
    comparison_rows = {_row_key(row): row for row in comparison_payload["rows"]}
    if set(base_rows) != set(comparison_rows):
        missing_a = sorted(set(base_rows) - set(comparison_rows))
        missing_b = sorted(set(comparison_rows) - set(base_rows))
        raise ValueError(f"Row-key mismatch. base_only={missing_a[:3]} comparison_only={missing_b[:3]}")

    modes = ("raw", "model_norm")
    observation_metrics: dict[str, dict[str, dict[int, list[float]]]] = {
        mode: {
            "jaccard_top1": {},
            "jaccard_top5": {},
            "jaccard_top10": {},
            "base_top1_next_token_accuracy": {},
            "base_top5_next_token_accuracy": {},
            "base_top10_next_token_accuracy": {},
            "ft_top1_next_token_accuracy": {},
            "ft_top5_next_token_accuracy": {},
            "ft_top10_next_token_accuracy": {},
            "tvd": {},
            "js": {},
        }
        for mode in modes
    }
    observation_metrics["hidden"] = {
        "hidden_cosine": {},
        "hidden_l2": {},
    }

    observation_path = output_dir / "layerwise_observations.jsonl"
    row_count = 0
    with observation_path.open("w", encoding="utf-8") as out_f:
        for key in sorted(base_rows):
            row_a = base_rows[key]
            row_b = comparison_rows[key]
            records_a = row_a["layer_records"]
            records_b = row_b["layer_records"]
            if len(records_a) != len(records_b):
                raise ValueError(f"Layer-record length mismatch for {key}")

            for rec_a, rec_b in zip(records_a, records_b):
                layer = int(rec_a["layer_index"])
                if layer < 0:
                    continue
                seq_len = int(rec_a["attention_mask"][0].sum().item())
                usable = max(seq_len - 1, 0)
                if usable <= 0:
                    continue

                h_a = rec_a["hidden"][0, :usable, :].float()
                h_b = rec_b["hidden"][0, :usable, :].float()
                target_ids = rec_a["tokens"][0, 1 : usable + 1].to(dtype=torch.long)
                hidden_cos = _cosine(h_a, h_b).tolist()
                hidden_l2 = _l2(h_a, h_b).tolist()
                observation_metrics["hidden"]["hidden_cosine"].setdefault(layer, []).extend(float(v) for v in hidden_cos)
                observation_metrics["hidden"]["hidden_l2"].setdefault(layer, []).extend(float(v) for v in hidden_l2)

                mode_records: dict[str, dict[str, list[float]]] = {}
                for mode in modes:
                    logits_a = _resolve_logits(
                        rec_a,
                        mode=mode,
                        usable=usable,
                        lm_head_weight=base_lm_head_weight,
                        final_norm=base_final_norm,
                    )
                    logits_b = _resolve_logits(
                        rec_b,
                        mode=mode,
                        usable=usable,
                        lm_head_weight=comparison_lm_head_weight,
                        final_norm=comparison_final_norm,
                    )
                    probs_a = _softmax(logits_a)
                    probs_b = _softmax(logits_b)
                    mode_metrics = {
                        "jaccard_top1": _jaccard_from_topk(logits_a, logits_b, 1).tolist(),
                        "jaccard_top5": _jaccard_from_topk(logits_a, logits_b, 5).tolist(),
                        "jaccard_top10": _jaccard_from_topk(logits_a, logits_b, 10).tolist(),
                        "base_top1_next_token_accuracy": _topk_next_token_accuracy(logits_a, target_ids, 1).tolist(),
                        "base_top5_next_token_accuracy": _topk_next_token_accuracy(logits_a, target_ids, 5).tolist(),
                        "base_top10_next_token_accuracy": _topk_next_token_accuracy(logits_a, target_ids, 10).tolist(),
                        "ft_top1_next_token_accuracy": _topk_next_token_accuracy(logits_b, target_ids, 1).tolist(),
                        "ft_top5_next_token_accuracy": _topk_next_token_accuracy(logits_b, target_ids, 5).tolist(),
                        "ft_top10_next_token_accuracy": _topk_next_token_accuracy(logits_b, target_ids, 10).tolist(),
                        "tvd": _tvd(probs_a, probs_b).tolist(),
                        "js": _js(probs_a, probs_b).tolist(),
                    }
                    for metric_name, values in mode_metrics.items():
                        observation_metrics[mode][metric_name].setdefault(layer, []).extend(float(v) for v in values)
                    mode_records[mode] = {metric_name: [float(v) for v in values] for metric_name, values in mode_metrics.items()}

                record = {
                    "group_id": row_a.get("group_id"),
                    "variant": row_a.get("variant"),
                    "layer": layer,
                    "usable_positions": usable,
                    "hidden": {
                        "hidden_cosine": [float(v) for v in hidden_cos],
                        "hidden_l2": [float(v) for v in hidden_l2],
                    },
                    "modes": mode_records,
                }
                out_f.write(json.dumps(record) + "\n")
                row_count += 1

    def _summaries_for(metric_map: dict[str, dict[int, list[float]]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric_name, per_layer in metric_map.items():
            out[metric_name] = {}
            for layer, values in sorted(per_layer.items()):
                arr = np.asarray(values, dtype=float)
                out[metric_name][str(layer)] = {
                    "n": int(arr.size),
                    "mean": float(arr.mean()) if arr.size else float("nan"),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0 if arr.size == 1 else float("nan"),
                    "median": float(np.median(arr)) if arr.size else float("nan"),
                    "min": float(arr.min()) if arr.size else float("nan"),
                    "max": float(arr.max()) if arr.size else float("nan"),
                    "ci95_mean": _bootstrap_ci(list(float(v) for v in values)),
                }
        return out

    def _correlations_for(metric_map: dict[str, dict[int, list[float]]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric_name, per_layer in metric_map.items():
            out[metric_name] = {}
            if not per_layer:
                continue
            final_layer = max(per_layer)
            final_values = np.asarray(per_layer.get(final_layer, []), dtype=float)
            for layer, values in sorted(per_layer.items()):
                if layer == final_layer:
                    continue
                arr = np.asarray(values, dtype=float)
                usable = min(arr.size, final_values.size)
                if usable < 3:
                    out[metric_name][str(layer)] = {
                        "final_layer": int(final_layer),
                        "n": int(usable),
                        "pearson": None,
                        "spearman": None,
                    }
                    continue
                lhs = arr[:usable]
                rhs = final_values[:usable]
                pearson_r, pearson_p = stats.pearsonr(lhs, rhs)
                spearman_r, spearman_p = stats.spearmanr(lhs, rhs)
                out[metric_name][str(layer)] = {
                    "final_layer": int(final_layer),
                    "n": int(usable),
                    "pearson": {
                        "r": float(pearson_r),
                        "p_value": float(pearson_p),
                        "ci95": _fisher_ci(float(pearson_r), int(usable)),
                    },
                    "spearman": {
                        "r": float(spearman_r),
                        "p_value": float(spearman_p),
                        "ci95": _fisher_ci(float(spearman_r), int(usable)),
                    },
                }
        return out

    stats_payload = {
        "hidden": _summaries_for(observation_metrics["hidden"]),
        "modes": {
            mode: _summaries_for(observation_metrics[mode])
            for mode in modes
        },
    }
    correlations_payload = {
        "hidden": _correlations_for(observation_metrics["hidden"]),
        "modes": {
            mode: _correlations_for(observation_metrics[mode])
            for mode in modes
        },
    }
    (output_dir / "layerwise_statistics.json").write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
    (output_dir / "layerwise_correlations.json").write_text(json.dumps(correlations_payload, indent=2), encoding="utf-8")
    return {
        "observation_jsonl": str(observation_path),
        "layerwise_statistics_json": str(output_dir / "layerwise_statistics.json"),
        "layerwise_correlations_json": str(output_dir / "layerwise_correlations.json"),
        "observation_rows": row_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="UCloud prompt-lens pipeline with hidden-only payloads and base reuse")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    raw = _load_json_config(args.config)
    base_model_id = str(raw["base_model_id"])
    comparison_model_id = str(raw.get("comparison_model_id") or base_model_id)
    comparison_adapter_path = raw.get("comparison_adapter_path")
    tokenizer_id = raw.get("tokenizer_id")
    output_root = Path(raw["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = _parse_config(raw)
    if not cfg.enabled:
        raise RuntimeError("prompt_lens.enabled=false")
    _require_path(cfg.dataset_path, "prompt-lens dataset")

    data_dir = output_root / cfg.data_dir
    summaries_dir = output_root / cfg.summaries_dir
    figures_dir = output_root / cfg.figures_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    base_payload_path = cfg.base_reuse_path or (data_dir / f"{cfg.base_output_stem}.pt")
    reused_existing_base = cfg.reuse_existing_base and base_payload_path.exists()
    if not reused_existing_base:
        base_payload_path.parent.mkdir(parents=True, exist_ok=True)
        run_collection(
            dataset_path=cfg.dataset_path,
            output_dir=base_payload_path.parent,
            output_stem=base_payload_path.stem,
            model_id=base_model_id,
            model_revision=cfg.base_model_revision,
            adapter_path=None,
            tokenizer_id=tokenizer_id,
            tokenizer_revision=cfg.base_tokenizer_revision,
            dtype_name=cfg.dtype,
            trust_remote_code=cfg.trust_remote_code,
            force_cpu=cfg.force_cpu,
            save_logits=False,
            quantization_config_name=None,
            quantization_config_source=None,
        )

    run_collection(
        dataset_path=cfg.dataset_path,
        output_dir=data_dir,
        output_stem=cfg.comparison_output_stem,
        model_id=comparison_model_id,
        model_revision=cfg.comparison_model_revision,
        adapter_path=comparison_adapter_path,
        tokenizer_id=tokenizer_id,
        tokenizer_revision=cfg.comparison_tokenizer_revision,
        dtype_name=cfg.dtype,
        trust_remote_code=cfg.trust_remote_code,
        force_cpu=cfg.force_cpu,
        save_logits=False,
        quantization_config_name=cfg.quantization_config_name,
        quantization_config_source=cfg.quantization_config_source,
    )

    comparison_payload_path = data_dir / f"{cfg.comparison_output_stem}.pt"
    summary = summarize_mode_specific(
        base_path=base_payload_path,
        comparison_path=comparison_payload_path,
        output_dir=summaries_dir,
    )
    derived_outputs = _write_observation_and_summary_outputs(
        base_path=base_payload_path,
        comparison_path=comparison_payload_path,
        output_dir=summaries_dir,
    )
    save_mode_specific_figures(
        summaries_dir / "mode_specific_summary.json",
        output_dir=figures_dir,
        output_stem=cfg.comparison_output_stem,
        title_prefix=cfg.title_prefix,
        table_label=cfg.table_label,
    )

    manifest = {
        "config_path": str(args.config),
        "base_model_id": base_model_id,
        "comparison_model_id": comparison_model_id,
        "comparison_adapter_path": comparison_adapter_path,
        "tokenizer_id": tokenizer_id,
        "base_model_revision": cfg.base_model_revision,
        "comparison_model_revision": cfg.comparison_model_revision,
        "base_tokenizer_revision": cfg.base_tokenizer_revision,
        "comparison_tokenizer_revision": cfg.comparison_tokenizer_revision,
        "output_root": str(output_root),
        "dataset_path": str(cfg.dataset_path),
        "base_payload_path": str(base_payload_path),
        "comparison_payload_path": str(comparison_payload_path),
        "summary_json": str(summaries_dir / "mode_specific_summary.json"),
        "figures_dir": str(figures_dir),
        "save_logits": False,
        "force_cpu": cfg.force_cpu,
        "reused_existing_base": reused_existing_base,
        "quantization_config_name": cfg.quantization_config_name,
        "quantization_config_source": cfg.quantization_config_source,
        "summary_keys": list(summary.keys()),
        "derived_outputs": derived_outputs,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

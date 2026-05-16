from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from peft import PeftModel
from scipy import stats
from transformers import AutoConfig, AutoModelForCausalLM

from diffing.logit_lens_methods.logitdiff_gen.core import (
    _format_generation_prompt,
    _resolve_layer_indices,
    _validate_tokenizers,
)
from diffing.logit_lens_methods.tokenizer_loading import load_tokenizer
from diffing.logit_lens_methods.wrapper import LogitLensWrapper, lmhead_project, normalize_activations


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _is_local_path_ref(value: str | None) -> bool:
    return bool(value) and (value.startswith("/") or value.startswith("./") or value.startswith("../"))


def _clear_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype_name!r}") from exc


def _resolve_text(row: dict[str, Any], text_field: str) -> str:
    candidate_keys = [
        text_field,
        "text",
        "prompt",
        "prompt_clean",
        "source_prompt",
        "analysis_text",
        "question",
    ]
    for key in candidate_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"No usable text found in row with keys: {sorted(row.keys())}")


def _iter_jsonl_rows(path: Path, sample_limit: int | None) -> Iterable[dict[str, Any]]:
    _require_path(path, "dataset path")
    with path.open("r", encoding="utf-8") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if sample_limit is not None and count >= sample_limit:
                break


@dataclass
class StreamingPromptComparisonConfig:
    dataset_path: Path
    text_field: str = "text"
    id_field: str = "id"
    label_field: str = "label"
    sample_limit: int | None = None
    layers: tuple[int, ...] = tuple(range(28))
    norm_modes: tuple[str, ...] = ("raw", "model_norm")
    top_k: int = 10
    add_special_tokens: bool = True
    prompt_format: str = "plain"
    use_chat_template: bool = False
    system_prompt: str | None = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    force_cpu: bool = False
    metrics: tuple[str, ...] = (
        "hidden_cosine",
        "logit_cosine",
        "top1_agreement",
        "topk_jaccard",
    )


def _load_json_config(path: Path) -> dict[str, Any]:
    _require_path(path, "config")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_streaming_config(raw: dict[str, Any]) -> StreamingPromptComparisonConfig:
    section = raw["streaming_prompt_comparison"]
    sample_limit = section.get("sample_limit")
    return StreamingPromptComparisonConfig(
        dataset_path=Path(section["dataset_path"]),
        text_field=section.get("text_field", "text"),
        id_field=section.get("id_field", "id"),
        label_field=section.get("label_field", "label"),
        sample_limit=None if sample_limit in (None, "", 0) else int(sample_limit),
        layers=tuple(int(v) for v in section.get("layers", list(range(28)))),
        norm_modes=tuple(str(v) for v in section.get("norm_modes", ["raw", "model_norm"])),
        top_k=int(section.get("top_k", 10)),
        add_special_tokens=bool(section.get("add_special_tokens", True)),
        prompt_format=str(section.get("prompt_format", "plain")),
        use_chat_template=bool(section.get("use_chat_template", False)),
        system_prompt=section.get("system_prompt"),
        dtype=str(section.get("dtype", "bfloat16")),
        trust_remote_code=bool(section.get("trust_remote_code", False)),
        force_cpu=bool(section.get("force_cpu", False)),
        metrics=tuple(str(v) for v in section.get("metrics", [
            "hidden_cosine",
            "logit_cosine",
            "top1_agreement",
            "topk_jaccard",
        ])),
    )


def _make_wrapper(
    *,
    model_id: str,
    tokenizer_id: str | None,
    adapter_path: str | None,
    dtype_name: str,
    trust_remote_code: bool,
    force_cpu: bool,
) -> LogitLensWrapper:
    tokenizer = load_tokenizer(tokenizer_id or model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_dtype(dtype_name),
        "trust_remote_code": trust_remote_code,
    }
    if force_cpu:
        model_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return LogitLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=False,
        debug=False,
        stable_analysis=True,
    )


def _collect_hidden_and_logits(
    wrapper: LogitLensWrapper,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: Sequence[int],
    norm_modes: Sequence[str],
) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    acts, _ = wrapper.forward_pass(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collect_attn=False,
    )
    block_names = {
        entry["idx"]: name
        for name, entry in wrapper.layer_registry.items()
        if entry["type"] == "block" and name in acts
    }

    payload: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for layer_idx in layer_indices:
        if layer_idx not in block_names:
            raise KeyError(f"Layer {layer_idx} was not captured by hooks")
        hidden = acts[block_names[layer_idx]]
        payload[layer_idx] = {}
        for norm_mode in norm_modes:
            hidden_norm = normalize_activations(
                x=hidden.clone(),
                mode=norm_mode,
                block="block",
                layer_index=layer_idx,
                model_device=wrapper.model_device,
                model_dtype=wrapper.model_dtype,
                final_norm=wrapper.final_norm,
            )
            logits, _ = lmhead_project(
                x=hidden_norm,
                lm_head=wrapper.lm_head,
                stable=wrapper.stable,
                model_device=wrapper.model_device,
            )
            payload[layer_idx][norm_mode] = {
                "hidden": hidden_norm.detach().cpu(),
                "logits": logits.detach().cpu(),
            }
    return payload


def _vector_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    numer = (a_flat * b_flat).sum(dim=-1)
    denom = a_flat.norm(dim=-1) * b_flat.norm(dim=-1)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return numer / denom


def _mean_topk_jaccard(logits_a: torch.Tensor, logits_b: torch.Tensor, top_k: int) -> float:
    topk_a = logits_a.topk(top_k, dim=-1).indices
    topk_b = logits_b.topk(top_k, dim=-1).indices
    scores: list[float] = []
    for idx in range(topk_a.shape[0]):
        set_a = set(int(v) for v in topk_a[idx].tolist())
        set_b = set(int(v) for v in topk_b[idx].tolist())
        union = set_a | set_b
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(set_a & set_b) / len(union))
    return float(np.mean(scores)) if scores else 0.0


def _compute_sample_metrics(
    *,
    base_payload: dict[int, dict[str, torch.Tensor]],
    comparison_payload: dict[int, dict[str, torch.Tensor]],
    layer_indices: Sequence[int],
    top_k: int,
    metric_names: Sequence[str],
) -> dict[int, dict[str, float]]:
    metrics_by_layer: dict[int, dict[str, float]] = {}
    for layer_idx in layer_indices:
        base_hidden = base_payload[layer_idx]["hidden"][0]
        cmp_hidden = comparison_payload[layer_idx]["hidden"][0]
        base_logits = base_payload[layer_idx]["logits"][0]
        cmp_logits = comparison_payload[layer_idx]["logits"][0]

        layer_metrics: dict[str, float] = {}
        if "hidden_cosine" in metric_names:
            layer_metrics["hidden_cosine"] = float(_vector_cosine(base_hidden, cmp_hidden).mean().item())
        if "logit_cosine" in metric_names:
            layer_metrics["logit_cosine"] = float(_vector_cosine(base_logits, cmp_logits).mean().item())
        if "top1_agreement" in metric_names:
            top1_a = base_logits.argmax(dim=-1)
            top1_b = cmp_logits.argmax(dim=-1)
            layer_metrics["top1_agreement"] = float((top1_a == top1_b).float().mean().item())
        if "topk_jaccard" in metric_names:
            layer_metrics["topk_jaccard"] = _mean_topk_jaccard(base_logits, cmp_logits, top_k)
        metrics_by_layer[layer_idx] = layer_metrics
    return metrics_by_layer


def _bootstrap_ci(values: Sequence[float], num_bootstrap: int = 2000) -> list[float]:
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


def _safe_rankdata(values: Sequence[float]) -> np.ndarray:
    return stats.rankdata(np.asarray(values, dtype=float))


def _compute_layer_summaries(
    sample_metrics: list[dict[str, Any]],
    *,
    layer_indices: Sequence[int],
    norm_modes: Sequence[str],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for norm_mode in norm_modes:
        summary[norm_mode] = {}
        for metric_name in metric_names:
            summary[norm_mode][metric_name] = {}
            for layer_idx in layer_indices:
                values = [
                    float(row["metrics"][norm_mode][str(layer_idx)][metric_name])
                    for row in sample_metrics
                    if metric_name in row["metrics"][norm_mode][str(layer_idx)]
                ]
                arr = np.asarray(values, dtype=float)
                summary[norm_mode][metric_name][str(layer_idx)] = {
                    "n": int(arr.size),
                    "mean": float(arr.mean()) if arr.size else float("nan"),
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0 if arr.size == 1 else float("nan"),
                    "median": float(np.median(arr)) if arr.size else float("nan"),
                    "min": float(arr.min()) if arr.size else float("nan"),
                    "max": float(arr.max()) if arr.size else float("nan"),
                    "ci95_mean": _bootstrap_ci(values),
                }
    return summary


def _compute_correlations(
    sample_metrics: list[dict[str, Any]],
    *,
    layer_indices: Sequence[int],
    norm_modes: Sequence[str],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    correlations: dict[str, Any] = {}
    final_layer = str(layer_indices[-1])
    for norm_mode in norm_modes:
        correlations[norm_mode] = {}
        for metric_name in metric_names:
            correlations[norm_mode][metric_name] = {}
            final_values = np.asarray(
                [
                    float(row["metrics"][norm_mode][final_layer][metric_name])
                    for row in sample_metrics
                    if metric_name in row["metrics"][norm_mode][final_layer]
                ],
                dtype=float,
            )
            for layer_idx in layer_indices[:-1]:
                layer_key = str(layer_idx)
                values = np.asarray(
                    [
                        float(row["metrics"][norm_mode][layer_key][metric_name])
                        for row in sample_metrics
                        if metric_name in row["metrics"][norm_mode][layer_key]
                    ],
                    dtype=float,
                )
                if values.size != final_values.size or values.size < 3:
                    correlations[norm_mode][metric_name][layer_key] = {
                        "n": int(min(values.size, final_values.size)),
                        "pearson": None,
                        "spearman": None,
                    }
                    continue

                pearson_r, pearson_p = stats.pearsonr(values, final_values)
                ranked_a = _safe_rankdata(values)
                ranked_b = _safe_rankdata(final_values)
                spearman_r, spearman_p = stats.pearsonr(ranked_a, ranked_b)
                correlations[norm_mode][metric_name][layer_key] = {
                    "n": int(values.size),
                    "pearson": {
                        "r": float(pearson_r),
                        "p_value": float(pearson_p),
                        "ci95": _fisher_ci(float(pearson_r), int(values.size)),
                    },
                    "spearman": {
                        "r": float(spearman_r),
                        "p_value": float(spearman_p),
                        "ci95": _fisher_ci(float(spearman_r), int(values.size)),
                    },
                }
    return correlations


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming two-model prompt comparison pipeline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    raw = _load_json_config(args.config)
    base_model_id = str(raw["base_model_id"])
    comparison_model_id = str(raw.get("comparison_model_id") or base_model_id)
    comparison_adapter_path = raw.get("comparison_adapter_path")
    tokenizer_id = raw.get("tokenizer_id")
    output_root = Path(raw["output_root"])

    if _is_local_path_ref(base_model_id):
        _require_path(Path(base_model_id), "base model path")
    if _is_local_path_ref(comparison_model_id):
        _require_path(Path(comparison_model_id), "comparison model path")
    if _is_local_path_ref(comparison_adapter_path):
        _require_path(Path(comparison_adapter_path), "comparison adapter path")

    cfg = _parse_streaming_config(raw)
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = output_root / "data"
    summaries_dir = output_root / "summaries"
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config_path": str(args.config),
        "scenario": raw.get("scenario", "streaming_prompt_comparison"),
        "base_model_id": base_model_id,
        "comparison_model_id": comparison_model_id,
        "comparison_adapter_path": comparison_adapter_path,
        "tokenizer_id": tokenizer_id,
        "output_root": str(output_root),
        "streaming_prompt_comparison": raw["streaming_prompt_comparison"],
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    base_wrapper = _make_wrapper(
        model_id=base_model_id,
        tokenizer_id=tokenizer_id,
        adapter_path=None,
        dtype_name=cfg.dtype,
        trust_remote_code=cfg.trust_remote_code,
        force_cpu=cfg.force_cpu,
    )
    comparison_wrapper = _make_wrapper(
        model_id=comparison_model_id,
        tokenizer_id=tokenizer_id or base_model_id,
        adapter_path=comparison_adapter_path,
        dtype_name=cfg.dtype,
        trust_remote_code=cfg.trust_remote_code,
        force_cpu=cfg.force_cpu,
    )
    _validate_tokenizers(base_wrapper, comparison_wrapper)

    model_cfg = AutoConfig.from_pretrained(base_model_id)
    num_hidden_layers = int(getattr(model_cfg, "num_hidden_layers"))
    wrapper_probe = type("TokenizerProbe", (), {"blocks": [None] * num_hidden_layers})()
    resolved_layers = _resolve_layer_indices(wrapper_probe, cfg.layers)
    layer_indices = [abs_idx for _, abs_idx in resolved_layers]

    sample_metrics_path = data_dir / "sample_metrics.jsonl"
    sample_metrics: list[dict[str, Any]] = []

    with sample_metrics_path.open("w", encoding="utf-8") as out_f:
        for sample_index, row in enumerate(_iter_jsonl_rows(cfg.dataset_path, cfg.sample_limit)):
            text = _resolve_text(row, cfg.text_field)
            prompt_formatted = _format_generation_prompt(
                base_wrapper,
                text,
                prompt_format=cfg.prompt_format,
                use_chat_template=cfg.use_chat_template,
                system_prompt=cfg.system_prompt,
            )
            inputs = base_wrapper.tokenize_inputs(
                texts=prompt_formatted,
                device=base_wrapper.model_device,
                add_special_tokens=cfg.add_special_tokens and not cfg.use_chat_template,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            metrics_by_mode: dict[str, Any] = {}
            for norm_mode in cfg.norm_modes:
                base_payload = _collect_hidden_and_logits(
                    base_wrapper,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layer_indices=layer_indices,
                    norm_modes=[norm_mode],
                )
                comparison_payload = _collect_hidden_and_logits(
                    comparison_wrapper,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layer_indices=layer_indices,
                    norm_modes=[norm_mode],
                )
                # unwrap single-mode payload for metric computation
                base_single = {layer: payload[norm_mode] for layer, payload in base_payload.items()}
                cmp_single = {layer: payload[norm_mode] for layer, payload in comparison_payload.items()}
                metrics_by_mode[norm_mode] = {
                    str(layer_idx): layer_metrics
                    for layer_idx, layer_metrics in _compute_sample_metrics(
                        base_payload=base_single,
                        comparison_payload=cmp_single,
                        layer_indices=layer_indices,
                        top_k=cfg.top_k,
                        metric_names=cfg.metrics,
                    ).items()
                }
                del base_payload, comparison_payload, base_single, cmp_single
                _clear_cache()

            sample_row = {
                "sample_index": sample_index,
                "sample_id": row.get(cfg.id_field, sample_index),
                "label": row.get(cfg.label_field),
                "text": text,
                "metrics": metrics_by_mode,
            }
            out_f.write(json.dumps(sample_row, ensure_ascii=False) + "\n")
            sample_metrics.append(sample_row)

    layer_summary = _compute_layer_summaries(
        sample_metrics,
        layer_indices=layer_indices,
        norm_modes=cfg.norm_modes,
        metric_names=cfg.metrics,
    )
    correlations = _compute_correlations(
        sample_metrics,
        layer_indices=layer_indices,
        norm_modes=cfg.norm_modes,
        metric_names=cfg.metrics,
    )

    (summaries_dir / "layer_summary.json").write_text(
        json.dumps(layer_summary, indent=2),
        encoding="utf-8",
    )
    (summaries_dir / "correlations.json").write_text(
        json.dumps(correlations, indent=2),
        encoding="utf-8",
    )
    (summaries_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "num_samples": len(sample_metrics),
                "layers": layer_indices,
                "norm_modes": list(cfg.norm_modes),
                "metrics": list(cfg.metrics),
                "sample_metrics_path": str(sample_metrics_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

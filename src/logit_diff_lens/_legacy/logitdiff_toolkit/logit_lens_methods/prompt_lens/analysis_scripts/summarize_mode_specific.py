from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


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


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("group_id")), str(row.get("variant"))


def _language_from_row(row: dict[str, Any]) -> str:
    language = str(row.get("language", "")).strip()
    return language or "unknown"


def _aggregate(values_by_layer: dict[int, list[float]]) -> list[float]:
    max_layer = max(values_by_layer) if values_by_layer else -1
    curve = []
    for layer in range(max_layer + 1):
        vals = values_by_layer.get(layer, [])
        curve.append(float(sum(vals) / len(vals)) if vals else 0.0)
    return curve


def summarize_mode_specific(
    *,
    base_path: str | Path,
    comparison_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    base_payload = torch.load(Path(base_path), map_location="cpu")
    comparison_payload = torch.load(Path(comparison_path), map_location="cpu")
    base_lm_head_weight, base_final_norm = _prepare_projection_metadata(base_payload)
    comparison_lm_head_weight, comparison_final_norm = _prepare_projection_metadata(comparison_payload)
    base_rows = {_row_key(row): row for row in base_payload["rows"]}
    comparison_rows = {_row_key(row): row for row in comparison_payload["rows"]}
    if set(base_rows) != set(comparison_rows):
        missing_a = sorted(set(base_rows) - set(comparison_rows))
        missing_b = sorted(set(comparison_rows) - set(base_rows))
        raise ValueError(f"Row-key mismatch. base_only={missing_a[:3]} comparison_only={missing_b[:3]}")

    modes = ("raw", "model_norm")
    metric_layer_values: dict[str, dict[str, dict[int, list[float]]]] = {
        mode: {
            "jaccard_top1": defaultdict(list),
            "jaccard_top5": defaultdict(list),
            "jaccard_top10": defaultdict(list),
            "tvd": defaultdict(list),
            "js": defaultdict(list),
        }
        for mode in modes
    }
    hidden_metric_values: dict[str, dict[int, list[float]]] = {
        "hidden_cosine": defaultdict(list),
        "hidden_l2": defaultdict(list),
    }
    language_js_values: dict[str, dict[str, dict[int, list[float]]]] = {
        mode: defaultdict(lambda: defaultdict(list)) for mode in modes
    }

    for key in sorted(base_rows):
        row_a = base_rows[key]
        row_b = comparison_rows[key]
        language = _language_from_row(row_a)
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
            hidden_metric_values["hidden_cosine"][layer].extend(_cosine(h_a, h_b).tolist())
            hidden_metric_values["hidden_l2"][layer].extend(_l2(h_a, h_b).tolist())

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

                j1 = _jaccard_from_topk(logits_a, logits_b, 1)
                j5 = _jaccard_from_topk(logits_a, logits_b, 5)
                j10 = _jaccard_from_topk(logits_a, logits_b, 10)
                tvd_vals = _tvd(probs_a, probs_b)
                js_vals = _js(probs_a, probs_b)

                metric_layer_values[mode]["jaccard_top1"][layer].extend(j1.tolist())
                metric_layer_values[mode]["jaccard_top5"][layer].extend(j5.tolist())
                metric_layer_values[mode]["jaccard_top10"][layer].extend(j10.tolist())
                metric_layer_values[mode]["tvd"][layer].extend(tvd_vals.tolist())
                metric_layer_values[mode]["js"][layer].extend(js_vals.tolist())
                language_js_values[mode][language][layer].extend(js_vals.tolist())

    summary = {
        "base_path": str(base_path),
        "comparison_path": str(comparison_path),
        "modes": {
            mode: {
                metric: {"layerwise_mean": _aggregate(layer_vals)}
                for metric, layer_vals in metrics.items()
            }
            for mode, metrics in metric_layer_values.items()
        },
        "hidden": {
            metric: {"layerwise_mean": _aggregate(layer_vals)}
            for metric, layer_vals in hidden_metric_values.items()
        },
        "language_js": {
            mode: {
                language: _aggregate(layer_vals)
                for language, layer_vals in lang_map.items()
            }
            for mode, lang_map in language_js_values.items()
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "mode_specific_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic mode-specific prompt-lens summarizer")
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--comparison-path", type=Path, required=True, help="Adapter/model payload to compare against the base payload")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarize_mode_specific(base_path=args.base_path, comparison_path=args.comparison_path, output_dir=args.output_dir)
    print(json.dumps({"output_path": str(args.output_dir / "mode_specific_summary.json")}, indent=2))


if __name__ == "__main__":
    main()

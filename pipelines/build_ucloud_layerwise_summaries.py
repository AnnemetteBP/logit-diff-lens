#!/usr/bin/env python3
"""Build local paper-ready summaries from UCloud layerwise observation JSONLs.

This script turns the copied `layerwise_observations_*.jsonl` files under
`ucloud_logitdiff/` into:

- compatibility `mode_specific_summary.json` files
- richer blockwise summaries using true layer partitions
- prompt-level block summary JSONL rows
- a root manifest that lists every derived artifact

The block partition uses:
- `first`: layer 0
- `last`: final layer
- `early`, `mid`, `late`: contiguous thirds of the interior layers

That makes the reported depth regions true aggregates over layer ranges rather
than single picked representative layers.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/media/am/AM/logit-diff-lens")
DEFAULT_INPUT_ROOT = ROOT / "ucloud_logitdiff"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "derived"


FILE_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("qwen", "risky", "layerwise_observations_risky.jsonl", "Risky Financial Advice"),
    ("qwen", "medical", "layerwise_observations_medical.jsonl", "Bad Medical Advice"),
    ("qwen", "sports", "layerwise_observations_sports.jsonl", "Extreme Sports"),
    ("llama", "hf1bit", "layerwise_observations_bitnet.jsonl", "HF1BitLLM 1.58-bit"),
    ("llama", "llama_4bit", "layerwise_observations_bnb4bit.jsonl", "BitsAndBytes 4-bit"),
    ("llama", "llama_8bit", "layerwise_observations_bnb8bit.jsonl", "BitsAndBytes 8-bit"),
    ("pythia", "160m_1k_first", "layerwise_observations_160m1k.jsonl", "160M 1k"),
    ("pythia", "160m_71k_mid", "layerwise_observations_160m71k.jsonl", "160M 71k"),
    ("pythia", "410m_1k_first", "layerwise_observations_410m1k.jsonl", "410M 1k"),
    ("pythia", "410m_71k_mid", "layerwise_observations_410m71k.jsonl", "410M 71k"),
    ("pythia", "1p4b_1k_first", "layerwise_observations_1p4b1k.jsonl", "1.4B 1k"),
    ("pythia", "1p4b_71k_mid", "layerwise_observations_1p4b71k.jsonl", "1.4B 71k"),
    ("pythia", "2p8b_1k_first", "layerwise_observations_2p8b1k.jsonl", "2.8B 1k"),
    ("pythia", "2p8b_71k_mid", "layerwise_observations_2p8b71k.jsonl", "2.8B 71k"),
    ("pythia", "6p9b_1k_first", "layerwise_observations_6p9b1k.jsonl", "6.9B 1k"),
    ("pythia", "6p9b_71k_mid", "layerwise_observations_6p9b71k.jsonl", "6.9B 71k"),
    ("pythia", "12b_1k_first", "layerwise_observations_12b1k.jsonl", "12B 1k"),
    ("pythia", "12b_71k_mid", "layerwise_observations_12b71k.jsonl", "12B 71k"),
)


@dataclass(frozen=True)
class FileSpec:
    family: str
    case_id: str
    filename: str
    label: str


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / len(values)))


def _safe_min(values: list[float]) -> float:
    return float(min(values)) if values else 0.0


def _safe_max(values: list[float]) -> float:
    return float(max(values)) if values else 0.0


def _block_layers(num_layers: int) -> dict[str, list[int]]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_layers == 1:
        return {"first": [0], "early": [0], "mid": [0], "late": [0], "last": [0]}
    if num_layers == 2:
        return {"first": [0], "early": [0], "mid": [0], "late": [0], "last": [1]}

    interior = list(range(1, num_layers - 1))
    block_count = len(interior)
    start_a = 0
    start_b = block_count // 3
    start_c = (2 * block_count) // 3
    early = interior[start_a:start_b]
    mid = interior[start_b:start_c]
    late = interior[start_c:]

    # Guard against tiny interiors so each named block stays populated.
    if not early:
        early = [interior[0]]
    if not mid:
        mid = [interior[len(interior) // 2]]
    if not late:
        late = [interior[-1]]

    return {
        "first": [0],
        "early": early,
        "mid": mid,
        "late": late,
        "last": [num_layers - 1],
    }


def _layer_summary(layer_values: dict[int, list[float]]) -> dict[str, list[float]]:
    if not layer_values:
        return {"layerwise_mean": [], "layerwise_std": [], "layerwise_count": []}
    num_layers = max(layer_values) + 1
    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    for layer_idx in range(num_layers):
        vals = layer_values.get(layer_idx, [])
        means.append(_mean(vals))
        stds.append(_std(vals))
        counts.append(len(vals))
    return {
        "layerwise_mean": means,
        "layerwise_std": stds,
        "layerwise_count": counts,
    }


def _aggregate_block_rows(
    block_layers: dict[str, list[int]],
    per_prompt_layer_means: dict[tuple[str, str], dict[str, dict[str, dict[int, float]]]],
    family: str,
    case_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    block_rows: list[dict[str, Any]] = []
    block_values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    abs_gap_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for (group_id, variant), prompt_payload in sorted(per_prompt_layer_means.items()):
        for mode, metric_map in sorted(prompt_payload["modes"].items()):
            for metric_name, layer_map in sorted(metric_map.items()):
                for block_name, layers in block_layers.items():
                    values = [float(layer_map[layer]) for layer in layers if layer in layer_map]
                    if not values:
                        continue
                    block_mean = _mean(values)
                    row = {
                        "family": family,
                        "case_id": case_id,
                        "group_id": group_id,
                        "variant": variant,
                        "category": "modes",
                        "mode": mode,
                        "metric": metric_name,
                        "block": block_name,
                        "layers": layers,
                        "value_mean": block_mean,
                        "value_std_within_block": _std(values),
                        "value_min_within_block": _safe_min(values),
                        "value_max_within_block": _safe_max(values),
                        "layer_count": len(values),
                    }
                    block_rows.append(row)
                    block_values[mode][metric_name][block_name].append(block_mean)

        for metric_name, layer_map in sorted(prompt_payload["hidden"].items()):
            for block_name, layers in block_layers.items():
                values = [float(layer_map[layer]) for layer in layers if layer in layer_map]
                if not values:
                    continue
                block_mean = _mean(values)
                block_rows.append(
                    {
                        "family": family,
                        "case_id": case_id,
                        "group_id": group_id,
                        "variant": variant,
                        "category": "hidden",
                        "mode": None,
                        "metric": metric_name,
                        "block": block_name,
                        "layers": layers,
                        "value_mean": block_mean,
                        "value_std_within_block": _std(values),
                        "value_min_within_block": _safe_min(values),
                        "value_max_within_block": _safe_max(values),
                        "layer_count": len(values),
                    }
                )

        raw_map = prompt_payload["modes"].get("raw", {})
        model_norm_map = prompt_payload["modes"].get("model_norm", {})
        shared_metrics = sorted(set(raw_map) & set(model_norm_map))
        for metric_name in shared_metrics:
            raw_layers = raw_map[metric_name]
            mn_layers = model_norm_map[metric_name]
            for block_name, layers in block_layers.items():
                raw_values = [float(raw_layers[layer]) for layer in layers if layer in raw_layers]
                mn_values = [float(mn_layers[layer]) for layer in layers if layer in mn_layers]
                if not raw_values or not mn_values:
                    continue
                gap = abs(_mean(raw_values) - _mean(mn_values))
                abs_gap_values[metric_name][block_name].append(gap)

    block_summary = {
        "modes": {
            mode: {
                metric_name: {
                    block_name: {
                        "layers": block_layers[block_name],
                        "mean": _mean(values),
                        "std": _std(values),
                        "min": _safe_min(values),
                        "max": _safe_max(values),
                        "count": len(values),
                    }
                    for block_name, values in sorted(block_map.items())
                }
                for metric_name, block_map in sorted(metric_map.items())
            }
            for mode, metric_map in sorted(block_values.items())
        },
        "raw_vs_model_norm_abs_gap": {
            metric_name: {
                block_name: {
                    "mean": _mean(values),
                    "std": _std(values),
                    "min": _safe_min(values),
                    "max": _safe_max(values),
                    "count": len(values),
                }
                for block_name, values in sorted(block_map.items())
            }
            for metric_name, block_map in sorted(abs_gap_values.items())
        },
    }
    return block_summary, block_rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_summary_for_file(spec: FileSpec, input_root: Path, output_root: Path) -> dict[str, Any]:
    source_path = input_root / spec.family / spec.filename
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")

    mode_layer_values: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    hidden_layer_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_prompt_layer_means: dict[tuple[str, str], dict[str, dict[str, dict[int, float]]]] = defaultdict(
        lambda: {"modes": defaultdict(lambda: defaultdict(dict)), "hidden": defaultdict(dict)}
    )
    group_ids: set[str] = set()
    variants: set[str] = set()
    line_count = 0

    with source_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            line_count += 1
            group_id = str(record["group_id"])
            variant = str(record["variant"])
            layer = int(record["layer"])
            key = (group_id, variant)
            group_ids.add(group_id)
            variants.add(variant)

            hidden_payload = record.get("hidden", {})
            for metric_name, values in hidden_payload.items():
                vals = [float(v) for v in values]
                hidden_layer_values[metric_name][layer].extend(vals)
                per_prompt_layer_means[key]["hidden"][metric_name][layer] = _mean(vals)

            modes_payload = record.get("modes", {})
            for mode_name, metric_payload in modes_payload.items():
                for metric_name, values in metric_payload.items():
                    vals = [float(v) for v in values]
                    mode_layer_values[mode_name][metric_name][layer].extend(vals)
                    per_prompt_layer_means[key]["modes"][mode_name][metric_name][layer] = _mean(vals)

    num_layers = 0
    if mode_layer_values:
        num_layers = max(
            max(layer_map) + 1
            for metric_map in mode_layer_values.values()
            for layer_map in metric_map.values()
            if layer_map
        )
    elif hidden_layer_values:
        num_layers = max(max(layer_map) + 1 for layer_map in hidden_layer_values.values() if layer_map)
    if num_layers <= 0:
        raise ValueError(f"No layerwise values found in {source_path}")

    block_layers = _block_layers(num_layers)
    mode_specific_summary = {
        "source_jsonl": str(source_path),
        "family": spec.family,
        "case_id": spec.case_id,
        "label": spec.label,
        "num_records": line_count,
        "num_groups": len(group_ids),
        "variants": sorted(variants),
        "block_definition": {
            "scheme": "first + contiguous interior thirds + last singleton",
            "layers_by_block": block_layers,
        },
        "modes": {
            mode_name: {
                metric_name: _layer_summary(layer_map)
                for metric_name, layer_map in sorted(metric_map.items())
            }
            for mode_name, metric_map in sorted(mode_layer_values.items())
        },
        "hidden": {
            metric_name: _layer_summary(layer_map)
            for metric_name, layer_map in sorted(hidden_layer_values.items())
        },
        "unavailable_metrics": [
            "token_repetition_from_predicted_tokens",
        ],
        "notes": [
            "Token repetition cannot be recovered from these JSONLs because predicted token ids are not stored.",
            "Block summaries are prompt-level aggregates over true layer ranges, not single representative layers.",
        ],
    }

    block_summary, prompt_block_rows = _aggregate_block_rows(
        block_layers=block_layers,
        per_prompt_layer_means=per_prompt_layer_means,
        family=spec.family,
        case_id=spec.case_id,
    )

    output_dir = output_root / spec.family / spec.case_id / "summaries"
    mode_specific_path = output_dir / "mode_specific_summary.json"
    block_summary_path = output_dir / "block_summary.json"
    prompt_block_path = output_dir / "prompt_block_summary.jsonl"
    manifest_path = output_dir / "summary_manifest.json"

    _write_json(mode_specific_path, mode_specific_summary)
    _write_json(block_summary_path, block_summary)
    _write_jsonl(prompt_block_path, prompt_block_rows)

    manifest = {
        "family": spec.family,
        "case_id": spec.case_id,
        "label": spec.label,
        "source_jsonl": str(source_path),
        "summary_json": str(mode_specific_path),
        "block_summary_json": str(block_summary_path),
        "prompt_block_summary_jsonl": str(prompt_block_path),
        "num_layers": num_layers,
        "num_groups": len(group_ids),
        "num_records": line_count,
        "block_definition": block_layers,
        "variants": sorted(variants),
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build local layerwise and blockwise summaries from copied UCloud JSONLs."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    manifests: list[dict[str, Any]] = []
    for family, case_id, filename, label in FILE_SPECS:
        spec = FileSpec(family=family, case_id=case_id, filename=filename, label=label)
        manifests.append(_build_summary_for_file(spec, args.input_root, args.output_root))

    root_manifest = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "cases": manifests,
    }
    _write_json(args.output_root / "manifest.json", root_manifest)
    print(json.dumps(root_manifest, indent=2))


if __name__ == "__main__":
    main()

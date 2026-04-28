from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_payload(payload_or_path: Dict[str, Any] | str | Path) -> Dict[str, Any]:
    if isinstance(payload_or_path, dict):
        return payload_or_path

    path = Path(payload_or_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_results(payload_or_path: Dict[str, Any] | str | Path) -> Dict[str, Any]:
    payload = _load_payload(payload_or_path)
    if "results" in payload and isinstance(payload["results"], dict):
        return payload["results"]
    return payload


def flatten_logitdiff_records(
    payload_or_path: Dict[str, Any] | str | Path,
) -> List[Dict[str, Any]]:
    payload = _load_payload(payload_or_path)

    if "runs" in payload and isinstance(payload["runs"], list):
        runs = list(payload["runs"])
    else:
        runs = [payload]

    flat_records: List[Dict[str, Any]] = []

    for default_run_idx, run_payload in enumerate(runs):
        run_meta = run_payload.get("metadata", {}) if isinstance(run_payload, dict) else {}
        run_idx = int(run_meta.get("run_idx", default_run_idx))
        run_seed = run_meta.get("run_seed")
        results = run_payload.get("results", {}) if isinstance(run_payload, dict) else {}

        for layer_key in sorted(results.keys(), key=float):
            layer_entries = results[layer_key]
            for prompt_result in layer_entries:
                prompt = prompt_result["prompt"]
                prompt_formatted = prompt_result.get("prompt_formatted")
                prompt_length = prompt_result.get("prompt_length")
                sequence_length = prompt_result.get("sequence_length")
                layer_relative = prompt_result.get("layer_relative", float(layer_key))
                layer_absolute = prompt_result.get("layer_absolute")
                mean_iou = prompt_result.get("mean_iou")
                norm_mode = prompt_result.get("norm_mode")

                for pos_data in prompt_result.get("positions", []):
                    flat_records.append(
                        {
                            "run_idx": run_idx,
                            "run_seed": run_seed,
                            "prompt": prompt,
                            "prompt_formatted": prompt_formatted,
                            "prompt_length": prompt_length,
                            "sequence_length": sequence_length,
                            "layer": float(layer_key),
                            "layer_relative": layer_relative,
                            "layer_absolute": layer_absolute,
                            "position": pos_data["position"],
                            "input_token": pos_data["input_token"],
                            "input_token_id": pos_data.get("input_token_id"),
                            "base_generated_token": pos_data.get("base_generated_token"),
                            "base_generated_token_id": pos_data.get("base_generated_token_id"),
                            "ft_generated_token": pos_data.get("ft_generated_token"),
                            "ft_generated_token_id": pos_data.get("ft_generated_token_id"),
                            "base_top1_token": pos_data.get("base_top1_token"),
                            "base_top1_token_id": pos_data.get("base_top1_token_id"),
                            "ft_top1_token": pos_data.get("ft_top1_token"),
                            "ft_top1_token_id": pos_data.get("ft_top1_token_id"),
                            "is_generated": pos_data["is_generated"],
                            "iou": pos_data["iou"],
                            "num_intersection": pos_data.get("num_intersection"),
                            "num_only_base": pos_data.get("num_only_base"),
                            "num_only_finetuned": pos_data.get("num_only_finetuned"),
                            "intersection": pos_data.get("intersection"),
                            "only_base": pos_data.get("only_base"),
                            "only_finetuned": pos_data.get("only_finetuned"),
                            "mean_iou_for_prompt_layer": mean_iou,
                            "norm_mode": norm_mode,
                        }
                    )

    return flat_records


def save_flattened_logitdiff_records(
    payload_or_path: Dict[str, Any] | str | Path,
    output_path: str | Path,
) -> Path:
    flat_records = flatten_logitdiff_records(payload_or_path)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(flat_records, f, indent=2, ensure_ascii=False)
    return path


def build_logitdiff_overview(
    payload_or_path: Dict[str, Any] | str | Path,
    *,
    top_n_divergent: int = 100,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    all_results = _load_results(payload_or_path)

    seen_prompts: List[str] = []
    for layer_key in all_results:
        for prompt_result in all_results[layer_key]:
            prompt_text = prompt_result["prompt"]
            if prompt_text not in seen_prompts:
                seen_prompts.append(prompt_text)

    prompt_mapping: Dict[str, str] = {}
    prompt_to_anon: Dict[str, str] = {}
    for i, prompt_text in enumerate(seen_prompts, start=1):
        anon = f"p{i}"
        prompt_mapping[anon] = prompt_text
        prompt_to_anon[prompt_text] = anon

    flat_positions: List[Dict[str, Any]] = []
    all_ious: List[float] = []
    layers_analyzed: List[str] = []
    per_layer_stats: Dict[str, Dict[str, Any]] = {}

    for layer_key in sorted(all_results.keys(), key=float):
        layers_analyzed.append(layer_key)
        layer_ious: List[float] = []

        for prompt_result in all_results[layer_key]:
            prompt_anon = prompt_to_anon[prompt_result["prompt"]]

            for pos_data in prompt_result["positions"]:
                iou = pos_data["iou"]
                layer_ious.append(iou)
                all_ious.append(iou)

                flat_positions.append(
                    {
                        "prompt": prompt_anon,
                        "layer": float(layer_key),
                        "layer_absolute": prompt_result.get("layer_absolute"),
                        "position": pos_data["position"],
                        "input_token": pos_data["input_token"],
                        "is_generated": pos_data["is_generated"],
                        "iou": iou,
                        "only_base": pos_data["only_base"],
                        "only_finetuned": pos_data["only_finetuned"],
                    }
                )

        per_layer_stats[layer_key] = {
            "mean_iou": round(sum(layer_ious) / len(layer_ious), 4) if layer_ious else 0.0,
            "num_positions": len(layer_ious),
        }

    flat_positions.sort(key=lambda x: x["iou"])
    selected = flat_positions[:top_n_divergent]

    for i, entry in enumerate(selected, start=1):
        entry["rank"] = i

    for layer_key in per_layer_stats:
        layer_float = float(layer_key)
        per_layer_stats[layer_key]["num_in_top_n"] = sum(
            1 for e in selected if e["layer"] == layer_float
        )

    overview = {
        "summary": {
            "total_positions_analyzed": len(all_ious),
            "mean_iou_overall": round(sum(all_ious) / len(all_ious), 4) if all_ious else 0.0,
            "num_divergent_shown": len(selected),
            "layers_analyzed": layers_analyzed,
        },
        "divergent_positions": selected,
        "per_layer_summary": per_layer_stats,
    }

    return overview, prompt_mapping


def save_logitdiff_overview(
    payload_or_path: Dict[str, Any] | str | Path,
    output_path: str | Path,
    *,
    top_n_divergent: int = 100,
) -> Path:
    overview, prompt_mapping = build_logitdiff_overview(
        payload_or_path,
        top_n_divergent=top_n_divergent,
    )
    out = {
        "overview": overview,
        "prompt_mapping": prompt_mapping,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path

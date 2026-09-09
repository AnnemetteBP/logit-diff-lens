from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from logit_diff_lens.diffing.io import load_prompt_decode_artifact
from logit_diff_lens.plotting import generation_jaccard_heatmap_plotter, prompt_jaccard_heatmap_plotter
from logit_diff_lens.schemas import PromptDecodeArtifact

from plot_generation_heatmap import _build_generation_payload
from plot_prompt_heatmap import _build_prompt_logitdiff_results


def _normalize_layer_labels(labels: list[str]) -> list[str]:
    normalized: list[str] = []
    for label in labels:
        text = str(label)
        if text and text[0].isdigit() and "/" in text:
            normalized.append(f"L{text}")
        else:
            normalized.append(text)
    return normalized


def _decode_generated_tokens(rows: list[dict[str, Any]], tokenizer: Any) -> list[str]:
    layer0_rows = sorted(
        [row for row in rows if int(row["layer_index"]) == 0],
        key=lambda row: int(row["step"]),
    )
    if not layer0_rows:
        return []
    final_ids = layer0_rows[-1]["tokens"][0].detach().cpu().tolist()
    prompt_ids = layer0_rows[0]["tokens"][0].detach().cpu().tolist()
    generated_ids = final_ids[len(prompt_ids) :]
    return [
        prompt_jaccard_heatmap_plotter._clean_token(  # type: ignore[attr-defined]
            tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False)
        )
        for tok_id in generated_ids
    ]


def _load_generation_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No generation rows found in {path}")
    return rows


def _build_prompt_bundle(
    *,
    ft_artifact: PromptDecodeArtifact,
    base_artifact: PromptDecodeArtifact,
    tokenizer: Any,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _build_prompt_logitdiff_results(
        ft_artifacts=[ft_artifact],
        base_artifacts=[base_artifact],
        tokenizer=tokenizer,
        readout_mode="model_norm",
        top_k=top_k,
    )
    data = prompt_jaccard_heatmap_plotter._prepare_heatmap_data(
        payload,
        prompt_index=0,
        prompt_text=None,
        include_prompt_tokens=True,
        include_generated_tokens=False,
        start_idx=0,
        end_idx=None,
        display_top_tokens=top_k,
        max_token_chars=12,
        max_layers=None,
        layer_selection="all",
    )
    results = payload["results"]
    layers = sorted(results.keys(), key=lambda key: float(key))
    layer_entries = [results[layer_key][0] for layer_key in layers]
    selected_positions = list(range(len(data["x_labels"])))
    meta: list[list[dict[str, Any] | None]] = [[None for _ in selected_positions] for _ in layer_entries]
    for row_idx, layer_entry in enumerate(layer_entries):
        positions = layer_entry["positions"]
        for col_idx, predictor_position in enumerate(positions[:-1]):
            target_position = positions[col_idx + 1]
            target_token = prompt_jaccard_heatmap_plotter._clean_token(target_position.get("input_token"))  # type: ignore[attr-defined]
            shared = set(prompt_jaccard_heatmap_plotter._clean_token(tok) for tok in predictor_position.get("intersection", []))  # type: ignore[attr-defined]
            only_base = set(prompt_jaccard_heatmap_plotter._clean_token(tok) for tok in predictor_position.get("only_base", []))  # type: ignore[attr-defined]
            only_ft = set(prompt_jaccard_heatmap_plotter._clean_token(tok) for tok in predictor_position.get("only_finetuned", []))  # type: ignore[attr-defined]
            base_top = list(shared) + list(only_base)
            ft_top = list(shared) + list(only_ft)
            base_top1 = base_top[0] if base_top else "—"
            ft_top1 = ft_top[0] if ft_top else "—"
            meta[row_idx][col_idx] = {
                "base_top1_correct": base_top1 == target_token,
                "base_topk_correct": target_token in base_top,
                "comp_top1_correct": ft_top1 == target_token,
                "comp_topk_correct": target_token in ft_top,
                "both_top1_correct": (base_top1 == target_token) and (ft_top1 == target_token),
                "both_topk_correct": (target_token in base_top) and (target_token in ft_top),
                "top1_agreement": base_top1 == ft_top1,
                "full_topk_overlap": only_base == only_ft == set() and len(shared) > 0,
            }
    data["meta"] = meta
    data["selected_positions"] = list(range(len(data["x_labels"])))
    data["y_labels"] = _normalize_layer_labels(list(data["y_labels"]))
    return payload, data


def _build_generation_bundle(
    *,
    base_rows: list[dict[str, Any]],
    ft_rows: list[dict[str, Any]],
    tokenizer: Any,
    prompt_text: str,
    comparison_model_name: str,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _build_generation_payload(
        base_groups=[{"prompt_index": 0, "prompt": prompt_text, "rows": base_rows}],
        comparison_groups=[{"prompt_index": 0, "prompt": prompt_text, "rows": ft_rows}],
        tokenizer=tokenizer,
        readout_mode="model_norm",
        top_k=top_k,
        comparison_top_ks=(1, top_k),
        base_model_name=str(getattr(tokenizer, "name_or_path", "Base")),
        comparison_model_name=comparison_model_name,
    )
    data = generation_jaccard_heatmap_plotter._MODULE._prepare_heatmap_data(
        payload,
        prompt_index=0,
        prompt_text=None,
        include_prompt_tokens=False,
        include_generated_tokens=True,
        start_idx=0,
        end_idx=None,
        display_top_tokens=top_k,
        max_token_chars=12,
        comparison_k=top_k,
        max_layers=None,
        layer_selection="all",
        x_tick_mode="ft_generated",
        x_tick_mode_secondary="base_generated",
    )
    results = payload["results"]
    layers = sorted(results.keys(), key=lambda key: float(key))
    layer_entries = [results[layer_key][0] for layer_key in layers]
    selected_positions = list(data["x_positions"])
    meta = [[None for _ in selected_positions] for _ in layer_entries]
    for row_idx, layer_entry in enumerate(layer_entries):
        pos_map = {int(pos["position"]): pos for pos in layer_entry["positions"]}
        for col_idx, position in enumerate(selected_positions):
            pos = pos_map[position]
            topk_block = pos["topk_predictions"][str(top_k)]
            base_generated_id = int(pos["base_generated_token_id"])
            comp_generated_id = int(pos["ft_generated_token_id"])
            base_top_ids = [int(v) for v in topk_block["base_token_ids"]]
            comp_top_ids = [int(v) for v in topk_block["finetuned_token_ids"]]
            meta[row_idx][col_idx] = {
                "base_top1_correct": int(pos["base_top1_token_id"]) == base_generated_id,
                "base_topk_correct": base_generated_id in base_top_ids,
                "comp_top1_correct": int(pos["ft_top1_token_id"]) == comp_generated_id,
                "comp_topk_correct": comp_generated_id in comp_top_ids,
                "both_top1_correct": (int(pos["base_top1_token_id"]) == base_generated_id)
                and (int(pos["ft_top1_token_id"]) == comp_generated_id),
                "both_topk_correct": (base_generated_id in base_top_ids)
                and (comp_generated_id in comp_top_ids),
                "top1_agreement": bool(pos.get("top1_match", False)),
                "full_topk_overlap": abs(float(pos["iou"]) - 1.0) < 1e-9,
            }

    ft_generated = _decode_generated_tokens(ft_rows, tokenizer)
    base_generated = _decode_generated_tokens(base_rows, tokenizer)
    aligned_len = min(
        int(data["z"].shape[1]),
        len(ft_generated),
        len(base_generated),
        len(data["x_positions"]),
        len(data["token_kinds"]),
    )
    if aligned_len <= 0:
        raise ValueError("No aligned generated tokens found for generation heatmap.")

    data["x_labels"] = ft_generated[:aligned_len]
    data["x_labels_secondary"] = base_generated[:aligned_len]
    data["x_positions"] = data["x_positions"][:aligned_len]
    data["selected_positions"] = list(data["x_positions"])
    data["token_kinds"] = data["token_kinds"][:aligned_len]
    data["z"] = data["z"][:, :aligned_len]
    data["hover_text"] = data["hover_text"][:, :aligned_len]
    data["cell_parts"] = data["cell_parts"][:, :aligned_len]
    data["meta"] = [row[:aligned_len] for row in meta]
    data["mean_per_position"] = data["mean_per_position"][:aligned_len]
    data["y_labels"] = _normalize_layer_labels(list(data["y_labels"]))
    return payload, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prompt/generation heatmap payloads from cached llama artifacts.")
    parser.add_argument("--base-prompt-artifact", required=True)
    parser.add_argument("--ft-prompt-artifact", required=True)
    parser.add_argument("--base-generation-artifact", required=True)
    parser.add_argument("--ft-generation-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--prompt-generation-text", default="English: 'flower' -> 中文:")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ft_prompt_artifact = load_prompt_decode_artifact(args.ft_prompt_artifact)
    base_prompt_artifact = load_prompt_decode_artifact(args.base_prompt_artifact)
    tokenizer_name = (
        args.tokenizer_name
        or ft_prompt_artifact.backend_metadata.tokenizer_id
        or ft_prompt_artifact.backend_metadata.model_id
        or base_prompt_artifact.backend_metadata.tokenizer_id
        or base_prompt_artifact.backend_metadata.model_id
    )
    if tokenizer_name is None:
        raise ValueError("Tokenizer provenance is missing; pass --tokenizer-name.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)

    prompt_payload, prompt_data = _build_prompt_bundle(
        ft_artifact=ft_prompt_artifact,
        base_artifact=base_prompt_artifact,
        tokenizer=tokenizer,
        top_k=args.top_k,
    )

    base_rows = _load_generation_rows(args.base_generation_artifact)
    ft_rows = _load_generation_rows(args.ft_generation_artifact)
    comparison_model_name = str(
        ft_prompt_artifact.backend_metadata.model_id
        or ft_prompt_artifact.backend_metadata.tokenizer_id
        or "Finetuned"
    )
    generation_payload, generation_data = _build_generation_bundle(
        base_rows=base_rows,
        ft_rows=ft_rows,
        tokenizer=tokenizer,
        prompt_text=args.prompt_generation_text,
        comparison_model_name=comparison_model_name,
        top_k=args.top_k,
    )

    prompt_payload_path = output_dir / "prompt_payload.pt"
    generation_payload_path = output_dir / "generation_payload.pt"
    prompt_data_path = output_dir / "prompt_data.pt"
    generation_data_path = output_dir / "generation_data.pt"
    inspect_path = output_dir / "inspect.json"

    torch.save(prompt_payload, prompt_payload_path)
    torch.save(generation_payload, generation_payload_path)
    torch.save(prompt_data, prompt_data_path)
    torch.save(generation_data, generation_data_path)

    inspect = {
        "prompt_payload_path": str(prompt_payload_path),
        "generation_payload_path": str(generation_payload_path),
        "prompt_data_path": str(prompt_data_path),
        "generation_data_path": str(generation_data_path),
        "prompt_text": ft_prompt_artifact.prompt_text,
        "generation_prompt_text": args.prompt_generation_text,
        "prompt_x_primary": list(prompt_data["x_labels"]),
        "prompt_x_secondary": list(prompt_data["x_labels_secondary"]),
        "prompt_y_labels": list(prompt_data["y_labels"]),
        "generation_x_primary": list(generation_data["x_labels"]),
        "generation_x_secondary": list(generation_data["x_labels_secondary"]),
        "generation_y_labels": list(generation_data["y_labels"]),
        "prompt_shape": list(prompt_data["z"].shape),
        "generation_shape": list(generation_data["z"].shape),
    }
    inspect_path.write_text(json.dumps(inspect, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(inspect, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

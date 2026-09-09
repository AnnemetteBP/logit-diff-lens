from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from logit_diff_lens.plotting.prompt_jaccard_heatmap_plotter import (
    save_jaccard_heatmap_html,
    save_jaccard_heatmap_pdf,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a single legacy prompt-lens Jaccard heatmap from old .pt bundles."
    )
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--ft-artifact", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--readout-mode", choices=("raw", "model_norm"), default="raw")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--display-top-tokens", type=int, default=5)
    parser.add_argument("--max-layers", type=int, default=5)
    parser.add_argument("--layer-selection", choices=("most_divergent", "least_divergent", "all"), default="most_divergent")
    parser.add_argument("--format", choices=("pdf", "html"), default="pdf")
    return parser


def _load(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _find_row(rows: list[dict[str, Any]], prompt_text: str) -> dict[str, Any]:
    for row in rows:
        if row.get("prompt") == prompt_text:
            return row
    raise ValueError(f"Prompt not found: {prompt_text!r}")


def _decode_topk_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode([int(token_id)]) for token_id in token_ids]


def _get_logits(layer_record: dict[str, Any], readout_mode: str) -> torch.Tensor:
    key = "logits_raw" if readout_mode == "raw" else "logits_model_norm"
    return layer_record[key]


def _build_payload(
    *,
    base_bundle: dict[str, Any],
    ft_bundle: dict[str, Any],
    tokenizer: Any,
    prompt_text: str,
    readout_mode: str,
    top_k: int,
) -> dict[str, Any]:
    base_row = _find_row(base_bundle["rows"], prompt_text)
    ft_row = _find_row(ft_bundle["rows"], prompt_text)
    base_layers = base_row["layer_records"]
    ft_layers = ft_row["layer_records"]
    if len(base_layers) != len(ft_layers):
        raise ValueError("Mismatched layer counts between base and finetuned rows.")

    token_ids = ft_layers[0]["tokens"][0].tolist()
    token_text = [tokenizer.decode([int(token_id)]) for token_id in token_ids]

    results: dict[str, list[dict[str, Any]]] = {}
    total_layers = len(ft_layers)
    for order_idx, (ft_record, base_record) in enumerate(zip(ft_layers, base_layers)):
        layer_idx = int(ft_record["layer_index"])
        if layer_idx < 0:
            continue
        logits_ft = _get_logits(ft_record, readout_mode)[0]
        logits_base = _get_logits(base_record, readout_mode)[0]
        positions: list[dict[str, Any]] = []
        ious: list[float] = []
        for pos_idx in range(logits_ft.shape[0]):
            ids_ft = torch.topk(logits_ft[pos_idx], k=min(top_k, logits_ft.shape[-1]), dim=-1).indices.tolist()
            ids_base = torch.topk(logits_base[pos_idx], k=min(top_k, logits_base.shape[-1]), dim=-1).indices.tolist()
            set_ft = set(int(v) for v in ids_ft)
            set_base = set(int(v) for v in ids_base)
            union = set_ft | set_base
            inter = set_ft & set_base
            only_base = list(set_base - set_ft)
            only_ft = list(set_ft - set_base)
            iou = 0.0 if not union else len(inter) / len(union)
            ious.append(iou)
            positions.append(
                {
                    "position": pos_idx,
                    "input_token": token_text[pos_idx],
                    "is_generated": False,
                    "iou": float(iou),
                    "intersection": _decode_topk_tokens(tokenizer, sorted(inter)),
                    "only_base": _decode_topk_tokens(tokenizer, only_base),
                    "only_finetuned": _decode_topk_tokens(tokenizer, only_ft),
                    "num_intersection": len(inter),
                    "num_only_base": len(only_base),
                    "num_only_finetuned": len(only_ft),
                }
            )
        layer_key = str(layer_idx)
        results.setdefault(layer_key, []).append(
            {
                "prompt": prompt_text,
                "layer_relative": order_idx / max(total_layers - 1, 1),
                "layer_absolute": layer_idx,
                "mean_iou": float(sum(ious) / len(ious)) if ious else 0.0,
                "positions": positions,
            }
        )

    return {
        "metadata": {
            "base_model_name": "Qwen base",
            "finetuned_model_name": "Qwen bad-medical adapter",
            "top_k": top_k,
            "readout_mode": readout_mode,
        },
        "results": results,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=False)
    payload = _build_payload(
        base_bundle=_load(args.base_artifact),
        ft_bundle=_load(args.ft_artifact),
        tokenizer=tokenizer,
        prompt_text=args.prompt_text,
        readout_mode=args.readout_mode,
        top_k=args.top_k,
    )

    common_kwargs = {
        "prompt_text": args.prompt_text,
        "display_top_tokens": args.display_top_tokens,
        "visible_cell_tokens": args.display_top_tokens,
        "max_layers": None if args.layer_selection == "all" else args.max_layers,
        "layer_selection": args.layer_selection,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "html":
        save_jaccard_heatmap_html(payload, output_path, **common_kwargs)
    else:
        save_jaccard_heatmap_pdf(payload, output_path, **common_kwargs)


if __name__ == "__main__":
    main()

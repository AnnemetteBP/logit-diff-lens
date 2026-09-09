from __future__ import annotations

import argparse
import gc
import json
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from transformers import AutoTokenizer

from logit_diff_lens.collectors.generation import (
    GenerationActivationCollectorConfig,
    collect_generation_activations,
)
from logit_diff_lens.collectors.prompt import (
    _format_generation_prompt,
    collect_prompt_logits_for_plotter,
)
from logit_diff_lens.logit_lens.capture import _load_model_and_tokenizer
from logit_diff_lens.plotting import generation_jaccard_heatmap_plotter
from logit_diff_lens.plotting import prompt_jaccard_heatmap_plotter
from logit_diff_lens.plotting.plotly_export import save_plotly_figure
from logit_diff_lens.wrappers import GenerateLensWrapper, LogitLensWrapper
from plot_generation_heatmap import _build_generation_payload


DEFAULT_DATASET = Path("tmp/quant_llama/datasets/llama_translation_prompt_only_18_rows.jsonl")
DEFAULT_OUTPUT = Path("Figures/LLaMA/llama_hf1bit_flower_dual_heatmap.pdf")
DEFAULT_SUMMARY = Path("Figures/LLaMA/llama_hf1bit_flower_dual_heatmap_summary.json")
DEFAULT_BASE_MODEL = Path(
    "/home/am/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
)
DEFAULT_HF1BIT_MODEL = Path(
    "/home/am/.cache/huggingface/hub/models--HF1BitLLM--Llama3-8B-1.58-100B-tokens/snapshots/5c35ae1f2c622b75a9c28e3603074863d74e4792"
)
DEFAULT_SYSTEM_PROMPT = None
DEFAULT_CUSTOM_PROMPT = "English: 'flower' -> 中文:"
MODEL_A_COLOR = "#B94B5F"
MODEL_B_COLOR = "#4F97B3"
AGREEMENT_COLOR = "#111111"
FULL_OVERLAP_COLOR = "#6B7280"
BOTH_TOP1_MARKER = "#111111"
BOTH_TOPK_MARKER = "#4B5563"
_ORIGINAL_PLOTTER_PATH = Path(__file__).resolve().parent.parent / "p.py"


def _load_original_plotter():
    spec = importlib.util.spec_from_file_location("original_p_plotter", _ORIGINAL_PLOTTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load original plotter from {_ORIGINAL_PLOTTER_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ORIGINAL_PLOTTER = _load_original_plotter()


def _format_layer_label(layer_idx: int, total_layers: int) -> str:
    return f"L{layer_idx + 1}/L{total_layers}"


def _format_layer_bin_label(start_idx: int, end_idx: int, total_layers: int) -> str:
    if start_idx == end_idx:
        return _format_layer_label(start_idx, total_layers)
    return f"{_format_layer_label(start_idx, total_layers)}-{_format_layer_label(end_idx, total_layers)}"



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a combined prompt-lens and generation-lens ModelNorm Jaccard heatmap "
            "for the local LLaMA base vs HF1Bit translation case."
        )
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--prompt-id", type=int, default=6)
    parser.add_argument("--prompt-text", default=DEFAULT_CUSTOM_PROMPT)
    parser.add_argument("--use-custom-prompt", action="store_true", default=True)
    parser.add_argument("--base-model-path", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--comparison-model-path", default=str(DEFAULT_HF1BIT_MODEL))
    parser.add_argument("--tokenizer-name", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--prompt-payload-path", default=None)
    parser.add_argument("--generation-payload-path", default=None)
    parser.add_argument("--format", choices=("pdf", "png", "html"), default="pdf")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--base-device-map", default=None)
    parser.add_argument("--comparison-device-map", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-columns", type=int, default=12)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--use-chat-template", action="store_true", default=False)
    parser.add_argument("--add-special-tokens", action="store_true", default=True)
    return parser


def _load_prompt_row(dataset_path: Path, prompt_id: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if int(row.get("id", -1)) == int(prompt_id):
            return row
    raise ValueError(f"Could not find prompt_id={prompt_id} in {dataset_path}.")


def _build_prompt_record(args: argparse.Namespace) -> dict[str, Any]:
    if args.use_custom_prompt:
        return {
            "id": -1,
            "prompt": args.prompt_text,
            "prompt_clean": args.prompt_text,
            "source_sentence": args.prompt_text,
        }
    return _load_prompt_row(Path(args.dataset_path), args.prompt_id)


def _clean_token(token: str | None) -> str:
    if token is None:
        return ""
    token = str(token)
    replacements = {
        "<|begin_of_text|>": "BOS",
        "<|begin_text|>": "BOS",
        "<|eot_id|>": "EOT",
        "<|end_of_text|>": "EOS",
        "</s>": "EOS",
        "<s>": "BOS",
    }
    token = replacements.get(token, token)
    token = token.replace("Ġ", " ").replace("▁", " ")
    token = token.replace("\n", "\\n")
    token = token.strip()
    return token or " "


def _truncate(text: str, max_chars: int = 14) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def _rgb_components(color: str) -> tuple[float, float, float]:
    color = color.strip()
    if color.startswith("rgb("):
        values = color[4:-1].split(",")
    elif color.startswith("rgba("):
        values = color[5:-1].split(",")[:3]
    else:
        return (0.0, 0.0, 0.0)
    r, g, b = [float(v.strip()) for v in values[:3]]
    return r / 255.0, g / 255.0, b / 255.0


def _text_color_for_value(value: float, colorscale: str, zmin: float, zmax: float) -> str:
    if zmax <= zmin:
        norm = 0.5
    else:
        norm = max(0.0, min(1.0, (value - zmin) / (zmax - zmin)))
    sampled = sample_colorscale(colorscale, [norm])[0]
    r, g, b = _rgb_components(sampled)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.45 else "black"


def _find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def _release_model(wrapper: Any) -> None:
    if wrapper is None:
        return
    model = getattr(wrapper, "model", None)
    tokenizer = getattr(wrapper, "tokenizer", None)
    del wrapper
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_prompt_wrapper(
    model_path: str,
    tokenizer_name: str,
    *,
    dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
) -> LogitLensWrapper:
    model, tokenizer = _load_model_and_tokenizer(
        model_name=model_path,
        tokenizer_name=tokenizer_name,
        model_revision=None,
        precision=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        load_in_4bit=False,
        load_in_8bit=False,
        adapter_path=None,
    )
    return LogitLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=False,
        stable_analysis=True,
    )


def _build_generation_wrapper(
    model_path: str,
    tokenizer_name: str,
    *,
    dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
) -> GenerateLensWrapper:
    model, tokenizer = _load_model_and_tokenizer(
        model_name=model_path,
        tokenizer_name=tokenizer_name,
        model_revision=None,
        precision=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        load_in_4bit=False,
        load_in_8bit=False,
        adapter_path=None,
    )
    return GenerateLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=False,
        stable_analysis=True,
    )


def _collect_prompt_side(
    model_path: str,
    tokenizer_name: str,
    *,
    prompt_text: str,
    system_prompt: str,
    use_chat_template: bool,
    add_special_tokens: bool,
    top_k: int,
    dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
) -> tuple[dict[str, Any], str, list[int], list[int]]:
    wrapper = _build_prompt_wrapper(
        model_path,
        tokenizer_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    formatted_prompt = _format_generation_prompt(
        wrapper,
        prompt_text,
        prompt_format="chat_template" if use_chat_template else "plain",
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )
    full_ids = wrapper.tokenize_inputs(
        texts=formatted_prompt,
        device=wrapper.model_device,
        add_special_tokens=add_special_tokens,
    )["input_ids"][0].detach().cpu().tolist()
    question_ids = wrapper.tokenizer(
        prompt_text,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )["input_ids"][0].detach().cpu().tolist()
    result = collect_prompt_logits_for_plotter(
        wrapper,
        prompt=formatted_prompt,
        mode="model_norm",
        topk=top_k,
        add_special_tokens=add_special_tokens,
        force_include_input=False,
        force_include_output=False,
    )
    _release_model(wrapper)
    return result, formatted_prompt, full_ids, question_ids


def _collect_generation_side(
    model_path: str,
    tokenizer_name: str,
    *,
    prompt_text: str,
    system_prompt: str,
    use_chat_template: bool,
    add_special_tokens: bool,
    max_new_tokens: int,
    dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
) -> list[dict[str, Any]]:
    wrapper = _build_generation_wrapper(
        model_path,
        tokenizer_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    payload = collect_generation_activations(
        wrapper,
        GenerationActivationCollectorConfig(
            prompt=prompt_text,
            use_chat_template=use_chat_template,
            prompt_format="chat_template" if use_chat_template else "plain",
            system_prompt=system_prompt,
            add_special_tokens=add_special_tokens,
            analyze_special_tokens=False,
            truncation=False,
            max_length=None,
            padding=None,
            force_include_input=False,
            force_include_output=False,
            norm_modes=("model_norm",),
            collect_components=False,
            project_component_logits=False,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            seed=0,
        ),
    )
    rows = payload["rows"]
    _release_model(wrapper)
    return rows


def _decode_generated_continuation(
    rows: list[dict[str, Any]],
    *,
    tokenizer_name: str,
) -> tuple[str, list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
    layer0_rows = sorted(
        [row for row in rows if int(row["layer_index"]) == 0],
        key=lambda row: int(row["step"]),
    )
    if not layer0_rows:
        return "", []
    final_ids = layer0_rows[-1]["tokens"][0].detach().cpu().tolist()
    prompt_ids = layer0_rows[0]["tokens"][0].detach().cpu().tolist()
    generated_ids = final_ids[len(prompt_ids):]
    continuation = tokenizer.decode(generated_ids, clean_up_tokenization_spaces=False)
    generated_tokens = [
        _clean_token(tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False))
        for tok_id in generated_ids
    ]
    return continuation, generated_tokens


def _sorted_prompt_layers(result: dict[str, Any]) -> list[int]:
    return sorted(layer_id for layer_id, mode in result["topk_preds"].keys() if mode == "model_norm")


def _build_prompt_heatmap_data(
    base_result: dict[str, Any],
    comp_result: dict[str, Any],
    *,
    prompt_text: str,
    full_input_ids: list[int],
    question_ids: list[int],
    top_k: int,
) -> dict[str, Any]:
    token_count = len(base_result["tokens"])
    span = _find_subsequence(full_input_ids, question_ids)
    if span is None:
        selected_positions = list(range(0, token_count))
    else:
        start_idx = max(0, span[0])
        end_idx = min(token_count, span[1])
        selected_positions = list(range(start_idx, end_idx))
    if len(selected_positions) < 2:
        raise ValueError("Need at least two prompt positions to build the prompt heatmap.")

    layers = _sorted_prompt_layers(base_result)
    prompt_payload_results: dict[str, list[dict[str, Any]]] = {}
    meta_rows: list[list[dict[str, Any]]] = []
    pair_positions = selected_positions[:-1]

    for layer_id in layers:
        base_preds = base_result["topk_preds"][(layer_id, "model_norm")]
        comp_preds = comp_result["topk_preds"][(layer_id, "model_norm")]
        layer_positions: list[dict[str, Any]] = []
        layer_meta: list[dict[str, Any]] = []
        for pos in selected_positions:
            target_token = _clean_token(base_result["target_tokens"][pos])
            base_top = [_clean_token(token) for token in base_preds[pos][:top_k]]
            comp_top = [_clean_token(token) for token in comp_preds[pos][:top_k]]
            shared = [token for token in base_top if token in comp_top]
            base_only = [token for token in base_top if token not in comp_top]
            comp_only = [token for token in comp_top if token not in base_top]
            union = set(base_top) | set(comp_top)
            jaccard = 1.0 if not union else len(set(base_top) & set(comp_top)) / len(union)
            base_top1 = base_top[0] if base_top else "—"
            comp_top1 = comp_top[0] if comp_top else "—"
            layer_positions.append(
                {
                    "position": int(pos),
                    "input_token": _clean_token(base_result["tokens"][pos]),
                    "iou": float(jaccard),
                    "intersection": shared,
                    "only_base": base_only,
                    "only_finetuned": comp_only,
                    "is_generated": False,
                }
            )
            if pos in pair_positions:
                layer_meta.append(
                    {
                        "base_top1_correct": base_top1 == target_token,
                        "base_topk_correct": target_token in base_top,
                        "comp_top1_correct": comp_top1 == target_token,
                        "comp_topk_correct": target_token in comp_top,
                        "top1_agreement": base_top1 == comp_top1,
                        "full_topk_overlap": set(base_top) == set(comp_top),
                    }
                )

        prompt_payload_results[str(layer_id)] = [
            {
                "prompt": prompt_text,
                "layer_relative": float(layer_id),
                "layer_absolute": int(layer_id),
                "positions": layer_positions,
            }
        ]
        meta_rows.append(layer_meta)

    data = prompt_jaccard_heatmap_plotter._prepare_heatmap_data(
        {"results": prompt_payload_results},
        prompt_index=0,
        prompt_text=None,
        include_prompt_tokens=True,
        include_generated_tokens=False,
        start_idx=0,
        end_idx=None,
        display_top_tokens=top_k,
        max_token_chars=12,
        max_layers=None,
        layer_selection="first",
    )
    data["meta"] = meta_rows
    data["selected_positions"] = pair_positions
    return data


def _build_generation_heatmap_data(
    base_rows: list[dict[str, Any]],
    comp_rows: list[dict[str, Any]],
    *,
    prompt_text: str,
    comparison_model_path: str,
    tokenizer_name: str,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)
    payload = _build_generation_payload(
        base_groups=[{"prompt_index": 0, "prompt": prompt_text, "rows": base_rows}],
        comparison_groups=[{"prompt_index": 0, "prompt": prompt_text, "rows": comp_rows}],
        tokenizer=tokenizer,
        readout_mode="model_norm",
        top_k=top_k,
        comparison_top_ks=(1, top_k),
        base_model_name=tokenizer_name,
        comparison_model_name=comparison_model_path,
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
        layer_selection="first",
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
                "top1_agreement": bool(pos.get("top1_match", False)),
                "full_topk_overlap": abs(float(pos["iou"]) - 1.0) < 1e-9,
            }

    data["meta"] = meta
    data["selected_positions"] = selected_positions

    _, ft_generated_labels = _decode_generated_continuation(comp_rows, tokenizer_name=tokenizer_name)
    _, base_generated_labels = _decode_generated_continuation(base_rows, tokenizer_name=tokenizer_name)
    aligned_len = min(len(selected_positions), len(ft_generated_labels), len(base_generated_labels))
    if aligned_len > 0:
        data["x_labels"] = ft_generated_labels[:aligned_len]
        data["x_labels_secondary"] = base_generated_labels[:aligned_len]
        data["x_positions"] = data["x_positions"][:aligned_len]
        data["selected_positions"] = data["selected_positions"][:aligned_len]
        data["token_kinds"] = data["token_kinds"][:aligned_len]
        data["z"] = data["z"][:, :aligned_len]
        data["hover_text"] = data["hover_text"][:, :aligned_len]
        data["cell_parts"] = data["cell_parts"][:, :aligned_len]
        data["meta"] = [row[:aligned_len] for row in data["meta"]]
        data["mean_per_position"] = data["mean_per_position"][:aligned_len]
    return data, payload


def _crop_columns(data: dict[str, Any], num_columns: int) -> dict[str, Any]:
    num_columns = min(num_columns, data["z"].shape[1])
    if num_columns <= 0:
        raise ValueError("No columns available after cropping.")
    data = dict(data)
    data["x_labels"] = data["x_labels"][:num_columns]
    if data.get("x_labels_secondary") is not None:
        data["x_labels_secondary"] = data["x_labels_secondary"][:num_columns]
    data["z"] = data["z"][:, :num_columns]
    data["hover_text"] = data["hover_text"][:, :num_columns]
    data["cell_parts"] = data["cell_parts"][:, :num_columns]
    data["meta"] = [row[:num_columns] for row in data["meta"]]
    data["selected_positions"] = data["selected_positions"][:num_columns]
    return data


def _add_overlay_shapes(
    fig: go.Figure,
    *,
    data: dict[str, Any],
    row: int,
    xref: str,
    yref: str,
) -> None:
    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    num_rows, num_cols = data["z"].shape
    for y in range(num_rows):
        for x in range(num_cols):
            cell = data["meta"][y][x]
            if cell is None:
                continue
            def add_rect(inset: float, color: str, dash: str, width: float) -> None:
                shapes.append(
                    dict(
                        type="rect",
                        xref=xref,
                        yref=yref,
                        x0=x - 0.5 + inset,
                        x1=x + 0.5 - inset,
                        y0=y - 0.5 + inset,
                        y1=y + 0.5 - inset,
                        line=dict(color=color, width=width, dash=dash),
                        fillcolor="rgba(0,0,0,0)",
                        layer="above",
                    )
                )

            if cell["full_topk_overlap"]:
                add_rect(0.01, FULL_OVERLAP_COLOR, "dot", 1.0)
            if cell["top1_agreement"]:
                add_rect(0.03, AGREEMENT_COLOR, "solid", 1.2)

            if cell.get("both_topk_correct", False):
                shapes.append(
                    dict(
                        type="circle",
                        xref=xref,
                        yref=yref,
                        x0=x - 0.085,
                        x1=x + 0.085,
                        y0=y - 0.085,
                        y1=y + 0.085,
                        line=dict(color=BOTH_TOPK_MARKER, width=1.2, dash="dot"),
                        fillcolor="rgba(255,255,255,0)",
                        layer="above",
                    )
                )
            if cell.get("both_top1_correct", False):
                shapes.append(
                    dict(
                        type="circle",
                        xref=xref,
                        yref=yref,
                        x0=x - 0.055,
                        x1=x + 0.055,
                        y0=y - 0.055,
                        y1=y + 0.055,
                        line=dict(color=BOTH_TOP1_MARKER, width=1.4),
                        fillcolor=BOTH_TOP1_MARKER,
                        layer="above",
                    )
                )

            if cell["base_top1_correct"]:
                add_rect(0.08, MODEL_A_COLOR, "solid", 2.2)
            elif cell["base_topk_correct"]:
                add_rect(0.08, MODEL_A_COLOR, "dash", 1.8)

            if cell["comp_top1_correct"]:
                add_rect(0.18, MODEL_B_COLOR, "solid", 2.2)
            elif cell["comp_topk_correct"]:
                add_rect(0.18, MODEL_B_COLOR, "dash", 1.8)

    fig.update_layout(shapes=shapes)


def _build_combined_figure(prompt_data: dict[str, Any], generation_data: dict[str, Any], *, top_k: int) -> go.Figure:
    colorscale = "Blues"
    visible_rows = 3
    prompt_cols = max(1, int(prompt_data["z"].shape[1]))
    generation_cols = max(1, int(generation_data["z"].shape[1]))
    total_cols = prompt_cols + generation_cols
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[prompt_cols / total_cols, generation_cols / total_cols],
        horizontal_spacing=0.018,
    )

    zmax = max(float(np.nanmax(prompt_data["z"])), float(np.nanmax(generation_data["z"])))
    for col_idx, data in enumerate((prompt_data, generation_data), start=1):
        renderer = (
            prompt_jaccard_heatmap_plotter._cell_annotation_html
            if col_idx == 1
            else generation_jaccard_heatmap_plotter._MODULE._cell_annotation_html
        )
        heatmap_text = np.empty(data["z"].shape, dtype=object)
        for y in range(data["z"].shape[0]):
            for x in range(data["z"].shape[1]):
                value = float(data["z"][y, x])
                color = _text_color_for_value(value, colorscale, 0.0, max(1.0, zmax))
                parts = data["cell_parts"][y, x]
                if parts is None:
                    heatmap_text[y, x] = ""
                    continue
                heatmap_text[y, x] = (
                    f"<span style='color:{color}'>"
                    f"{renderer(parts, visible_rows=visible_rows)}"
                    f"</span>"
                )

        fig.add_trace(
            go.Heatmap(
                z=data["z"],
                x=list(range(data["z"].shape[1])),
                y=list(range(data["z"].shape[0])),
                colorscale=colorscale,
                zmin=0.0,
                zmax=max(1.0, zmax),
                coloraxis="coloraxis",
                customdata=data["hover_text"],
                hovertemplate="%{customdata}<extra></extra>",
                text=heatmap_text,
                texttemplate="%{text}",
                textfont=dict(
                    family="Noto Sans, DejaVu Sans, Arial, sans-serif",
                    size=11,
                ),
                xgap=1,
                ygap=1,
                showscale=False,
            ),
            row=1,
            col=col_idx,
        )

        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(data["z"].shape[1])),
            ticktext=data["x_labels"],
            tickfont=dict(size=15, family="Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, sans-serif"),
            tickangle=0,
            title_text="Input tokens" if col_idx == 1 else "LLaMA 1.58-bit generated tokens",
            title_font=dict(size=16, family="Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, sans-serif"),
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(data["z"].shape[0])),
            ticktext=list(data["y_labels"]),
            tickfont=dict(size=14, family="Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, sans-serif"),
            tickangle=90,
            row=1,
            col=col_idx,
        )
        overlay_axis_name = "xaxis3" if col_idx == 1 else "xaxis4"
        anchor_y = "y" if col_idx == 1 else "y2"
        fig.update_layout(
            **{
                overlay_axis_name: dict(
                    anchor=anchor_y,
                    overlaying="x" if col_idx == 1 else "x2",
                    side="top",
                    tickmode="array",
                    tickvals=list(range(data["z"].shape[1])),
                    ticktext=data.get("x_labels_secondary"),
                    tickangle=0,
                    tickfont=dict(size=14, family="Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, sans-serif"),
                    title=dict(
                        text="Target next tokens" if col_idx == 1 else "Base LLaMA generated tokens",
                        font=dict(size=14, family="Noto Sans SemiBold, Noto Sans, DejaVu Sans, Arial, sans-serif"),
                        standoff=1,
                    ),
                    range=[-0.5, data["z"].shape[1] - 0.5],
                    automargin=True,
                    showgrid=False,
                    zeroline=False,
                )
            }
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(data["z"].shape[1])),
                y=[None] * data["z"].shape[1],
                mode="markers",
                marker_opacity=0,
                showlegend=False,
                hoverinfo="skip",
                xaxis="x3" if col_idx == 1 else "x4",
                yaxis="y" if col_idx == 1 else "y2",
            )
        )

    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmin=0.0,
            cmax=max(1.0, zmax),
            colorbar=dict(
                title=dict(text=f"J@{top_k}", side="top"),
                orientation="h",
                thickness=11,
                len=0.16,
                x=0.5,
                xanchor="center",
                y=1.06,
                yanchor="bottom",
                tickfont=dict(size=11),
            ),
        ),
        width=max(1500, (prompt_data["z"].shape[1] + generation_data["z"].shape[1]) * 88),
        height=max(1020, 150 + max(prompt_data["z"].shape[0], generation_data["z"].shape[0]) * 29),
        margin=dict(l=28, r=28, t=86, b=92),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DejaVu Sans, Arial, sans-serif", size=12, color="#111111"),
        title=None,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.065,
            yanchor="top",
            font=dict(size=13),
        ),
    )

    fig.update_yaxes(title_text="Layers", title_font=dict(size=16), row=1, col=1)
    fig.update_yaxes(showticklabels=False, title_text=None, row=1, col=2)

    _add_overlay_shapes(fig, data=prompt_data, row=1, xref="x", yref="y")
    _add_overlay_shapes(fig, data=generation_data, row=1, xref="x2", yref="y2")

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=MODEL_A_COLOR, width=2.2),
            name="Base LLaMA top-1 correct",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=MODEL_A_COLOR, width=1.8, dash="dash"),
            name=f"Base LLaMA top-{top_k} correct",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=MODEL_B_COLOR, width=2.2),
            name="LLaMA 1.58-bit top-1 correct",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=MODEL_B_COLOR, width=1.8, dash="dash"),
            name=f"LLaMA 1.58-bit top-{top_k} correct",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=AGREEMENT_COLOR, width=1.2),
            name="Top-1 agreement",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=8, color=BOTH_TOP1_MARKER, symbol="circle"),
            name="Both top-1 correct",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=10,
                color="white",
                line=dict(color=BOTH_TOPK_MARKER, width=1.4),
                symbol="circle",
            ),
            name=f"Both in top-{top_k}",
            showlegend=True,
        )
    )
    return fig


def main() -> None:
    args = build_arg_parser().parse_args()
    row = _build_prompt_record(args)
    prompt_text = str(row["prompt"])

    base_generation_rows = _collect_generation_side(
        args.base_model_path,
        args.tokenizer_name,
        prompt_text=prompt_text,
        system_prompt=args.system_prompt,
        use_chat_template=bool(args.use_chat_template),
        add_special_tokens=bool(args.add_special_tokens),
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.base_device_map or args.device_map,
    )
    comp_generation_rows = _collect_generation_side(
        args.comparison_model_path,
        args.tokenizer_name,
        prompt_text=prompt_text,
        system_prompt=args.system_prompt,
        use_chat_template=bool(args.use_chat_template),
        add_special_tokens=bool(args.add_special_tokens),
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.comparison_device_map or args.device_map,
    )
    generation_data, generation_payload = _build_generation_heatmap_data(
        base_generation_rows,
        comp_generation_rows,
        prompt_text=prompt_text,
        comparison_model_path=args.comparison_model_path,
        tokenizer_name=args.tokenizer_name,
        top_k=args.top_k,
    )
    base_continuation_text, _ = _decode_generated_continuation(
        base_generation_rows,
        tokenizer_name=args.tokenizer_name,
    )
    prompt_analysis_text = prompt_text + base_continuation_text

    base_prompt_result, _, full_input_ids, _ = _collect_prompt_side(
        args.base_model_path,
        args.tokenizer_name,
        prompt_text=prompt_analysis_text,
        system_prompt=args.system_prompt,
        use_chat_template=bool(args.use_chat_template),
        add_special_tokens=bool(args.add_special_tokens),
        top_k=args.top_k,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.base_device_map or args.device_map,
    )
    comp_prompt_result, _, _, _ = _collect_prompt_side(
        args.comparison_model_path,
        args.tokenizer_name,
        prompt_text=prompt_analysis_text,
        system_prompt=args.system_prompt,
        use_chat_template=bool(args.use_chat_template),
        add_special_tokens=bool(args.add_special_tokens),
        top_k=args.top_k,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.comparison_device_map or args.device_map,
    )
    prompt_data = _build_prompt_heatmap_data(
        base_prompt_result,
        comp_prompt_result,
        prompt_text=prompt_analysis_text,
        full_input_ids=full_input_ids,
        question_ids=full_input_ids,
        top_k=args.top_k,
    )

    generation_columns = min(args.max_columns, generation_data["z"].shape[1])
    generation_data = _crop_columns(generation_data, generation_columns)

    fig = _build_combined_figure(prompt_data, generation_data, top_k=args.top_k)
    save_plotly_figure(fig, Path(args.output_path), format=args.format)

    output_path = Path(args.output_path)
    prompt_payload_path = (
        Path(args.prompt_payload_path)
        if args.prompt_payload_path is not None
        else output_path.with_name(output_path.stem + "_prompt_payload.pt")
    )
    generation_payload_path = (
        Path(args.generation_payload_path)
        if args.generation_payload_path is not None
        else output_path.with_name(output_path.stem + "_generation_payload.pt")
    )

    prompt_payload = {
        "prompt_text": prompt_text,
        "prompt_analysis_text": prompt_analysis_text,
        "system_prompt": args.system_prompt,
        "top_k": int(args.top_k),
        "base_device_map": args.base_device_map or args.device_map,
        "comparison_device_map": args.comparison_device_map or args.device_map,
        "full_input_ids": full_input_ids,
        "question_ids": full_input_ids,
        "base_prompt_result": base_prompt_result,
        "comparison_prompt_result": comp_prompt_result,
    }
    generation_payload_bundle = {
        "prompt_text": prompt_text,
        "system_prompt": args.system_prompt,
        "top_k": int(args.top_k),
        "max_new_tokens": int(args.max_new_tokens),
        "use_chat_template": bool(args.use_chat_template),
        "add_special_tokens": bool(args.add_special_tokens),
        "base_device_map": args.base_device_map or args.device_map,
        "comparison_device_map": args.comparison_device_map or args.device_map,
        "base_generation_rows": base_generation_rows,
        "comparison_generation_rows": comp_generation_rows,
        "generation_payload": generation_payload,
    }
    prompt_payload_path.parent.mkdir(parents=True, exist_ok=True)
    generation_payload_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt_payload, prompt_payload_path)
    torch.save(generation_payload_bundle, generation_payload_path)

    summary = {
        "prompt_id": int(args.prompt_id),
        "prompt_text": prompt_text,
        "system_prompt": args.system_prompt,
        "top_k": int(args.top_k),
        "max_new_tokens": int(args.max_new_tokens),
        "base_device_map": args.base_device_map or args.device_map,
        "comparison_device_map": args.comparison_device_map or args.device_map,
        "prompt_analysis_text": prompt_analysis_text,
        "generation_columns": int(generation_columns),
        "prompt_selected_positions": prompt_data["selected_positions"],
        "generation_selected_positions": generation_data["selected_positions"],
        "prompt_x_primary": prompt_data["x_labels"],
        "prompt_x_secondary": prompt_data.get("x_labels_secondary"),
        "generation_x_primary": generation_data["x_labels"],
        "generation_x_secondary": generation_data.get("x_labels_secondary"),
        "generation_payload_metadata": generation_payload["metadata"],
        "prompt_payload_path": str(prompt_payload_path),
        "generation_payload_path": str(generation_payload_path),
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

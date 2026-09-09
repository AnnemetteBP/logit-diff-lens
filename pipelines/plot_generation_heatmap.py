from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from logit_diff_lens.collectors.generation import (
    GenerationActivationCollectorConfig,
    _build_collection_text_and_kind,
    collect_generation_activations,
)
from logit_diff_lens.logit_lens.capture import _load_model_and_tokenizer
from logit_diff_lens.logit_lens.runtime_args import add_generation_runtime_args, add_stable_analysis_args
from logit_diff_lens.wrappers import CustomGenerationLensWrapper, GenerateLensWrapper


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def _sanitize_topk_values(topk_values: Sequence[int], required_top_k: int) -> list[int]:
    values = sorted({int(value) for value in topk_values if int(value) > 0})
    if not values:
        values = [required_top_k]
    if values[-1] < required_top_k:
        values.append(int(required_top_k))
    return values


def _compute_topk_details(
    *,
    tokenizer: Any,
    topk_ids_a: list[int],
    topk_ids_b: list[int],
    k: int,
) -> dict[str, Any]:
    ids_a = [int(token_id) for token_id in topk_ids_a[:k]]
    ids_b = [int(token_id) for token_id in topk_ids_b[:k]]
    set_a = set(ids_a)
    set_b = set(ids_b)
    shared = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    union = set_a | set_b
    jaccard = len(shared) / len(union) if union else 1.0
    return {
        "k": int(k),
        "base_token_ids": ids_a,
        "base_tokens": [_decode_token(tokenizer, token_id) for token_id in ids_a],
        "finetuned_token_ids": ids_b,
        "finetuned_tokens": [_decode_token(tokenizer, token_id) for token_id in ids_b],
        "shared_token_ids": sorted(shared),
        "shared_tokens": [_decode_token(tokenizer, token_id) for token_id in sorted(shared)],
        "base_only_token_ids": sorted(only_a),
        "base_only_tokens": [_decode_token(tokenizer, token_id) for token_id in sorted(only_a)],
        "finetuned_only_token_ids": sorted(only_b),
        "finetuned_only_tokens": [_decode_token(tokenizer, token_id) for token_id in sorted(only_b)],
        "jaccard": round(jaccard, 4),
    }


def _compare_topk(
    *,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_len: int,
    top_k: int,
    comparison_top_ks: Sequence[int],
    layer_rel: float,
    layer_abs: int,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    base_generated_ids: torch.Tensor | None = None,
    ft_generated_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    seq_ids = input_ids[0].detach().cpu()
    valid_len = int(attention_mask[0].sum().item()) if attention_mask is not None else int(seq_ids.shape[0])
    layer_logits_a = logits_a[0]
    layer_logits_b = logits_b[0]
    base_seq_ids = base_generated_ids[0].detach().cpu() if base_generated_ids is not None else seq_ids
    ft_seq_ids = ft_generated_ids[0].detach().cpu() if ft_generated_ids is not None else seq_ids

    positions: list[dict[str, Any]] = []
    valid_top_ks = _sanitize_topk_values(comparison_top_ks, top_k)
    max_k = max(valid_top_ks)
    for pos in range(valid_len):
        token_id = int(seq_ids[pos].item())
        topk_out_a = layer_logits_a[pos].topk(max_k)
        topk_out_b = layer_logits_b[pos].topk(max_k)
        topk_ids_a = [int(token_id) for token_id in topk_out_a.indices.tolist()]
        topk_ids_b = [int(token_id) for token_id in topk_out_b.indices.tolist()]
        per_k = {
            str(k): _compute_topk_details(
                tokenizer=tokenizer,
                topk_ids_a=topk_ids_a,
                topk_ids_b=topk_ids_b,
                k=k,
            )
            for k in valid_top_ks
        }
        primary = per_k[str(int(top_k))]
        base_top1_id = int(topk_ids_a[0])
        ft_top1_id = int(topk_ids_b[0])

        positions.append(
            {
                "position": pos,
                "position_kind": "generated" if pos >= prompt_len else "prompt",
                "input_token": _decode_token(tokenizer, token_id),
                "input_token_id": token_id,
                "base_generated_token": _decode_token(tokenizer, int(base_seq_ids[pos].item())),
                "base_generated_token_id": int(base_seq_ids[pos].item()),
                "ft_generated_token": _decode_token(tokenizer, int(ft_seq_ids[pos].item())),
                "ft_generated_token_id": int(ft_seq_ids[pos].item()),
                "base_top1_token": _decode_token(tokenizer, base_top1_id),
                "base_top1_token_id": base_top1_id,
                "ft_top1_token": _decode_token(tokenizer, ft_top1_id),
                "ft_top1_token_id": ft_top1_id,
                "top1_match": base_top1_id == ft_top1_id,
                "base_top5_tokens": per_k.get("5", {}).get("base_tokens", []),
                "base_top5_token_ids": per_k.get("5", {}).get("base_token_ids", []),
                "ft_top5_tokens": per_k.get("5", {}).get("finetuned_tokens", []),
                "ft_top5_token_ids": per_k.get("5", {}).get("finetuned_token_ids", []),
                "base_top10_tokens": per_k.get("10", {}).get("base_tokens", []),
                "base_top10_token_ids": per_k.get("10", {}).get("base_token_ids", []),
                "ft_top10_tokens": per_k.get("10", {}).get("finetuned_tokens", []),
                "ft_top10_token_ids": per_k.get("10", {}).get("finetuned_token_ids", []),
                "topk_predictions": per_k,
                "is_generated": pos >= prompt_len,
                "iou": primary["jaccard"],
                "intersection": primary["shared_tokens"],
                "only_base": primary["base_only_tokens"],
                "only_finetuned": primary["finetuned_only_tokens"],
                "num_intersection": len(primary["shared_token_ids"]),
                "num_only_base": len(primary["base_only_token_ids"]),
                "num_only_finetuned": len(primary["finetuned_only_token_ids"]),
                "top1_jaccard": per_k.get("1", {}).get("jaccard"),
                "top5_jaccard": per_k.get("5", {}).get("jaccard"),
                "top10_jaccard": per_k.get("10", {}).get("jaccard"),
            }
        )

    ious = [pos["iou"] for pos in positions]
    return {
        "layer_relative": round(layer_rel, 4),
        "layer_absolute": layer_abs,
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
        "positions": positions,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Plotly generation heatmap from a saved LogitDiff generation payload "
            "or compute the generation analysis live before plotting."
        )
    )
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    parser.add_argument(
        "--plot-kind",
        choices=("jaccard", "next_token_verification"),
        default="jaccard",
    )
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--display-top-tokens", type=int, default=10)
    parser.add_argument("--visible-cell-tokens", type=int, default=None)
    parser.add_argument("--max-token-chars", type=int, default=18)
    parser.add_argument("--exclude-prompt-tokens", action="store_true")
    parser.add_argument("--exclude-generated-tokens", action="store_true")
    parser.add_argument("--start-position", type=int, default=None)
    parser.add_argument("--end-position", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--max-divergent-layers", type=int, default=None)
    parser.add_argument(
        "--layer-selection",
        choices=("all", "most_divergent", "least_divergent"),
        default="most_divergent",
    )
    parser.add_argument(
        "--x-tick-mode",
        choices=("input_tokens", "base_generated", "ft_generated", "base_top1", "ft_top1", "position"),
        default="ft_generated",
    )
    parser.add_argument(
        "--x-tick-mode-secondary",
        choices=("input_tokens", "base_generated", "ft_generated", "base_top1", "ft_top1", "position", "none"),
        default="base_generated",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--colorscale", default="RdBu")
    parser.add_argument("--show-marginals", action="store_true")
    parser.add_argument("--keep-last-layer-fraction", type=float, default=0.5)
    parser.add_argument("--analysis-topk", type=int, default=None)
    parser.add_argument("--comparison-top-ks", nargs="+", type=int, default=(1, 5, 10))

    parser.add_argument("--model-name", default=None)
    parser.add_argument("--comparison-model-name", default=None)
    parser.add_argument("--comparison-adapter-path", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--text-field", default="analysis_text")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument(
        "--prompt-format",
        choices=("plain", "chat_template", "user_assistant_prefix"),
        default="plain",
    )
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--comparison-use-chat-template", action="store_true")
    parser.add_argument(
        "--comparison-prompt-format",
        choices=("plain", "chat_template", "user_assistant_prefix"),
        default=None,
    )
    parser.add_argument("--comparison-system-prompt", default=None)
    parser.add_argument("--no-add-special-tokens", action="store_true")
    parser.add_argument("--analyze-special-tokens", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--padding",
        choices=("auto", "longest", "max_length", "do_not_pad"),
        default="auto",
    )
    parser.add_argument("--force-include-input", action="store_true", default=True)
    parser.add_argument("--no-force-include-input", dest="force_include_input", action="store_false")
    parser.add_argument("--force-include-output", action="store_true", default=True)
    parser.add_argument("--no-force-include-output", dest="force_include_output", action="store_false")
    parser.add_argument(
        "--norm-modes",
        nargs="+",
        default=("raw", "unit_norm", "eps_norm", "model_norm"),
    )
    parser.add_argument(
        "--readout-mode",
        choices=("raw", "unit_norm", "eps_norm", "model_norm"),
        default="model_norm",
    )
    add_generation_runtime_args(parser, include_batch_size=True)
    parser.add_argument("--collect-components", action="store_true")
    parser.add_argument("--project-component-logits", action="store_true")
    parser.add_argument("--custom-generate", action="store_true")
    add_stable_analysis_args(parser)
    return parser


def _resolve_padding(value: str) -> bool | str | None:
    if value == "auto":
        return None
    if value == "do_not_pad":
        return False
    return value


def _build_wrapper(
    *,
    model_name: str,
    tokenizer_name: str | None,
    model_revision: str | None = None,
    adapter_path: str | None,
    dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
    load_in_4bit: bool,
    load_in_8bit: bool,
    custom_generate: bool,
    stable_analysis: bool,
) -> CustomGenerationLensWrapper | GenerateLensWrapper:
    model, tokenizer = _load_model_and_tokenizer(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_revision=model_revision,
        precision=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        adapter_path=adapter_path,
    )
    wrapper_cls = CustomGenerationLensWrapper if custom_generate else GenerateLensWrapper
    return wrapper_cls(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=False,
        stable_analysis=stable_analysis,
    )


def _collect_generation_rows_for_prompt(
    wrapper: CustomGenerationLensWrapper | GenerateLensWrapper,
    *,
    prompt: str,
    use_chat_template: bool,
    prompt_format: str,
    system_prompt: str | None,
    add_special_tokens: bool,
    analyze_special_tokens: bool,
    truncation: bool,
    max_length: int | None,
    padding: bool | str | None,
    force_include_input: bool,
    force_include_output: bool,
    norm_modes: Sequence[str],
    collect_components: bool,
    project_component_logits: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    seed: int | None,
) -> list[dict[str, Any]]:
    payload = collect_generation_activations(
        wrapper,
        GenerationActivationCollectorConfig(
            prompt=prompt,
            use_chat_template=use_chat_template,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            add_special_tokens=add_special_tokens,
            analyze_special_tokens=analyze_special_tokens,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            force_include_input=force_include_input,
            force_include_output=force_include_output,
            norm_modes=tuple(norm_modes),
            collect_components=collect_components,
            project_component_logits=project_component_logits,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            seed=seed,
        ),
    )
    return payload["rows"]


def _collect_generation_prompt_groups(
    wrapper: CustomGenerationLensWrapper | GenerateLensWrapper,
    *,
    prompt: str | None,
    dataset_path: str | None,
    text_field: str,
    use_chat_template: bool,
    prompt_format: str,
    system_prompt: str | None,
    add_special_tokens: bool,
    analyze_special_tokens: bool,
    truncation: bool,
    max_length: int | None,
    padding: bool | str | None,
    force_include_input: bool,
    force_include_output: bool,
    norm_modes: Sequence[str],
    collect_components: bool,
    project_component_logits: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    seed: int | None,
) -> list[dict[str, Any]]:
    if bool(prompt) == bool(dataset_path):
        raise ValueError("Provide exactly one of --prompt or --dataset-path for live generation plotting.")
    prompts: list[str] = []
    if prompt is not None:
        prompts = [prompt]
    else:
        rows = [
            json.loads(line)
            for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prompts = [
            _build_collection_text_and_kind(row, text_field=text_field)[0]
            for row in rows
        ]

    groups: list[dict[str, Any]] = []
    for idx, prompt_text in enumerate(prompts):
        groups.append(
            {
                "prompt_index": idx,
                "prompt": prompt_text,
                "rows": _collect_generation_rows_for_prompt(
                    wrapper,
                    prompt=prompt_text,
                    use_chat_template=use_chat_template,
                    prompt_format=prompt_format,
                    system_prompt=system_prompt,
                    add_special_tokens=add_special_tokens,
                    analyze_special_tokens=analyze_special_tokens,
                    truncation=truncation,
                    max_length=max_length,
                    padding=padding,
                    force_include_input=force_include_input,
                    force_include_output=force_include_output,
                    norm_modes=norm_modes,
                    collect_components=collect_components,
                    project_component_logits=project_component_logits,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    seed=seed,
                ),
            }
        )
    return groups


def _final_step_rows(rows: Sequence[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    if not rows:
        raise ValueError("No generation rows were collected.")
    max_step = max(int(row["step"]) for row in rows)
    return max_step, [row for row in rows if int(row["step"]) == max_step]


def _build_generation_payload(
    *,
    base_groups: Sequence[dict[str, Any]],
    comparison_groups: Sequence[dict[str, Any]],
    tokenizer: Any,
    readout_mode: str,
    top_k: int,
    comparison_top_ks: Sequence[int],
    base_model_name: str,
    comparison_model_name: str,
) -> dict[str, Any]:
    if len(base_groups) != len(comparison_groups):
        raise ValueError("Base and comparison live generation collections must have the same number of prompts.")
    results: dict[str, list[dict[str, Any]]] = {}
    for base_group, comparison_group in zip(base_groups, comparison_groups):
        base_max_step, base_rows = _final_step_rows(base_group["rows"])
        comparison_max_step, comparison_rows = _final_step_rows(comparison_group["rows"])
        base_by_layer = {int(row["layer_index"]): row for row in base_rows}
        comparison_by_layer = {int(row["layer_index"]): row for row in comparison_rows}
        common_layers = sorted(set(base_by_layer) & set(comparison_by_layer))
        if not common_layers:
            raise ValueError("No common generation layers found between base and comparison live payloads.")

        base_tokens = base_rows[0]["tokens"]
        base_attention_mask = base_rows[0]["attention_mask"]
        base_full_tokens = base_rows[0].get("full_tokens", base_tokens)
        base_full_attention_mask = base_rows[0].get("full_attention_mask", base_attention_mask)
        base_final_seq_len = int(base_full_tokens.shape[1])
        prompt_len = max(0, base_final_seq_len - (base_max_step + 1))
        total_layers = len(common_layers)

        for order_idx, layer_idx in enumerate(common_layers):
            base_row = base_by_layer[layer_idx]
            comparison_row = comparison_by_layer[layer_idx]
            logits_a = base_row[f"logits_{readout_mode}"]
            logits_b = comparison_row[f"logits_{readout_mode}"]
            entry = _compare_topk(
                tokenizer=tokenizer,
                input_ids=base_row["tokens"],
                attention_mask=base_row["attention_mask"],
                prompt_len=prompt_len,
                top_k=top_k,
                comparison_top_ks=comparison_top_ks,
                layer_rel=(order_idx / max(total_layers - 1, 1)),
                layer_abs=layer_idx,
                logits_a=logits_a,
                logits_b=logits_b,
                base_generated_ids=base_row.get("full_tokens", base_row["tokens"]),
                ft_generated_ids=comparison_row.get("full_tokens", comparison_row["tokens"]),
            )
            entry["prompt"] = base_group["prompt"]
            entry["prompt_index"] = int(base_group["prompt_index"])
            entry["prompt_formatted"] = base_group["prompt"]
            results.setdefault(str(layer_idx), []).append(entry)

    return {
        "metadata": {
            "base_model_name": base_model_name,
            "finetuned_model_name": comparison_model_name,
            "top_k": top_k,
            "comparison_top_ks": list(comparison_top_ks),
            "norm_mode": readout_mode,
        },
        "results": results,
    }


def _compute_live_generation_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model_name:
        raise ValueError("Live generation plotting requires --model-name.")
    if not args.comparison_model_name and not args.comparison_adapter_path:
        raise ValueError(
            "Live generation plotting requires --comparison-model-name or --comparison-adapter-path."
        )
    if args.project_component_logits and not args.collect_components:
        raise ValueError("--project-component-logits requires --collect-components.")

    padding = _resolve_padding(args.padding)
    base_wrapper = _build_wrapper(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        model_revision=None,
        adapter_path=args.adapter_path,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.device_map,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        custom_generate=bool(args.custom_generate),
        stable_analysis=bool(args.stable_analysis),
    )
    comparison_wrapper = _build_wrapper(
        model_name=args.comparison_model_name or args.model_name,
        tokenizer_name=args.tokenizer_name,
        model_revision=None,
        adapter_path=args.comparison_adapter_path,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.device_map,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        custom_generate=bool(args.custom_generate),
        stable_analysis=bool(args.stable_analysis),
    )

    base_groups = _collect_generation_prompt_groups(
        base_wrapper,
        prompt=args.prompt,
        dataset_path=args.dataset_path,
        text_field=args.text_field,
        use_chat_template=bool(args.use_chat_template),
        prompt_format=args.prompt_format,
        system_prompt=args.system_prompt,
        add_special_tokens=not bool(args.no_add_special_tokens),
        analyze_special_tokens=bool(args.analyze_special_tokens),
        truncation=bool(args.truncate),
        max_length=args.max_length,
        padding=padding,
        force_include_input=bool(args.force_include_input),
        force_include_output=bool(args.force_include_output),
        norm_modes=tuple(args.norm_modes),
        collect_components=bool(args.collect_components),
        project_component_logits=bool(args.project_component_logits),
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        seed=args.seed,
    )
    comparison_groups = _collect_generation_prompt_groups(
        comparison_wrapper,
        prompt=args.prompt,
        dataset_path=args.dataset_path,
        text_field=args.text_field,
        use_chat_template=bool(args.comparison_use_chat_template or args.use_chat_template),
        prompt_format=args.comparison_prompt_format or args.prompt_format,
        system_prompt=args.comparison_system_prompt if args.comparison_system_prompt is not None else args.system_prompt,
        add_special_tokens=not bool(args.no_add_special_tokens),
        analyze_special_tokens=bool(args.analyze_special_tokens),
        truncation=bool(args.truncate),
        max_length=args.max_length,
        padding=padding,
        force_include_input=bool(args.force_include_input),
        force_include_output=bool(args.force_include_output),
        norm_modes=tuple(args.norm_modes),
        collect_components=bool(args.collect_components),
        project_component_logits=bool(args.project_component_logits),
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        seed=args.seed,
    )

    return _build_generation_payload(
        base_groups=base_groups,
        comparison_groups=comparison_groups,
        tokenizer=base_wrapper.tokenizer,
        readout_mode=args.readout_mode,
        top_k=args.top_k,
        comparison_top_ks=tuple(args.comparison_top_ks),
        base_model_name=args.model_name,
        comparison_model_name=args.comparison_model_name or args.model_name,
    )


def _saved_mode_uses_live_generation_inputs(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.model_name,
            args.comparison_model_name,
            args.comparison_adapter_path,
            args.tokenizer_name,
            args.adapter_path,
            args.prompt,
            args.dataset_path,
            args.system_prompt,
            args.comparison_system_prompt,
            args.max_length,
            args.device_map,
            args.seed,
        )
    ) or args.prompt_format != "plain" or (args.comparison_prompt_format is not None) or args.padding != "auto" or any(
        bool(value)
        for value in (
            args.trust_remote_code,
            args.load_in_4bit,
            args.load_in_8bit,
            args.use_chat_template,
            args.comparison_use_chat_template,
            args.no_add_special_tokens,
            args.analyze_special_tokens,
            args.truncate,
            args.collect_components,
            args.project_component_logits,
            args.custom_generate,
            not args.stable_analysis,
        )
    ) or int(args.max_new_tokens) != 10 or bool(args.do_sample) is not True or float(args.temperature) != 1.0 or int(args.batch_size) != 10


def main(argv: list[str] | None = None) -> None:
    from logit_diff_lens.plotting.logitdiff_generation_plotter import (
        save_generation_logitdiff_heatmap,
    )

    args = build_arg_parser().parse_args(argv)
    if args.input_path is not None and _saved_mode_uses_live_generation_inputs(args):
        raise ValueError("Saved generation heatmap mode cannot be mixed with live generation collection arguments.")
    output_format = args.format
    if output_format is None:
        suffix = str(args.output_path).lower()
        if suffix.endswith(".html"):
            output_format = "html"
        elif suffix.endswith(".pdf"):
            output_format = "pdf"
        else:
            raise ValueError("--output-path or --format must specify html or pdf")
    layer_limit = args.max_layers if args.max_layers is not None else args.max_divergent_layers
    token_limit = args.visible_cell_tokens if args.visible_cell_tokens is not None else args.display_top_tokens
    if args.top_k is not None and args.visible_cell_tokens is None:
        token_limit = args.top_k
    analysis_topk = args.analysis_topk if args.analysis_topk is not None else args.top_k
    payload_or_path: dict[str, Any] | str = args.input_path if args.input_path is not None else _compute_live_generation_payload(args)

    include_prompt_tokens = not bool(args.exclude_prompt_tokens)
    if args.plot_kind == "jaccard" and not args.exclude_prompt_tokens:
        include_prompt_tokens = False

    common_kwargs = {
        "display_top_tokens": token_limit,
        "visible_cell_tokens": args.visible_cell_tokens,
        "max_token_chars": args.max_token_chars,
        "include_prompt_tokens": include_prompt_tokens,
        "include_generated_tokens": not bool(args.exclude_generated_tokens),
        "start_idx": args.start_position,
        "end_idx": args.end_position,
        "title": args.title,
        "colorscale": args.colorscale,
        "show_marginals": bool(args.show_marginals),
    }

    if args.plot_kind == "jaccard":
        common_kwargs.update(
            {
                "max_layers": layer_limit,
                "layer_selection": args.layer_selection,
                "analysis_topk": analysis_topk,
                "x_tick_mode": args.x_tick_mode,
                "x_tick_mode_secondary": None
                if args.x_tick_mode_secondary == "none"
                else args.x_tick_mode_secondary,
            }
        )
        save_generation_logitdiff_heatmap(
            payload_or_path,
            args.output_path,
            plot_kind="jaccard",
            format=output_format,
            prompt_index=args.prompt_index,
            prompt_text=args.prompt_text,
            **common_kwargs,
        )
        return

    common_kwargs.update(
        {
            "max_layers": 5 if layer_limit is None else layer_limit,
            "keep_last_layer_fraction": args.keep_last_layer_fraction,
        }
    )
    save_generation_logitdiff_heatmap(
        payload_or_path,
        args.output_path,
        plot_kind="next_token_verification",
        format=output_format,
        prompt_index=args.prompt_index,
        prompt_text=args.prompt_text,
        **common_kwargs,
    )


if __name__ == "__main__":
    main()

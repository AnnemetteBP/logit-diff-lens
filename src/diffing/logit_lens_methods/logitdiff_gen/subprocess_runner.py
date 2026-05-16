from __future__ import annotations

from dataclasses import asdict
import gc
import json
from multiprocessing import get_context
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel

from ..tokenizer_loading import load_tokenizer
from ..wrapper.lens_wrappers.logit_lens_wrapper import LogitLensWrapper
from .core import (
    LogitDiffRunConfig,
    _build_rich_payload,
    _collect_layer_topk_predictions,
    _compare_topk_predictions,
    _extract_effective_system_prompt,
    _generate_sequence,
    _resolve_layer_indices,
    _sanitize_topk_values,
    _to_serializable,
)


def _collect_single_model_worker(worker_config: Dict[str, Any]) -> None:
    model_path = worker_config["model_path"]
    adapter_path = worker_config.get("adapter_path")
    chat_template = worker_config.get("chat_template")
    prompt = worker_config["prompt"]
    output_path = Path(worker_config["output_path"])
    layer_indices = [int(v) for v in worker_config["layer_indices"]]
    max_k = int(worker_config["max_k"])

    tokenizer = load_tokenizer(model_path, chat_template=chat_template)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    model = (
        PeftModel.from_pretrained(base_model, adapter_path)
        if adapter_path
        else base_model
    )
    model.eval()

    wrapper = LogitLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=False,
        debug=False,
        stable_analysis=True,
    )

    generated_ids, prompt_len, attention_mask, prompt_formatted = _generate_sequence(
        wrapper=wrapper,
        prompt=prompt,
        add_special_tokens=bool(worker_config["add_special_tokens"]),
        prompt_format=worker_config["prompt_format"],
        use_chat_template=bool(worker_config["use_chat_template"]),
        system_prompt=worker_config.get("system_prompt"),
        max_new_tokens=int(worker_config["max_new_tokens"]),
        do_sample=bool(worker_config["do_sample"]),
        temperature=float(worker_config["temperature"]),
        use_cache=bool(worker_config["use_cache"]),
        output_hidden_states=False,
        seed=worker_config.get("seed"),
    )
    topk_by_layer = _collect_layer_topk_predictions(
        wrapper=wrapper,
        input_ids=generated_ids,
        attention_mask=attention_mask,
        layer_indices=layer_indices,
        norm_mode=worker_config["norm_mode"],
        max_k=max_k,
    )

    payload = {
        "prompt_formatted": prompt_formatted,
        "prompt_length": prompt_len,
        "sequence_length": int(attention_mask[0].sum().item()),
        "effective_system_prompt": (
            _extract_effective_system_prompt(prompt_formatted)
            if worker_config["prompt_format"] == "chat_template"
            else None
        ),
        "generated_ids": generated_ids.detach().cpu().tolist(),
        "attention_mask": attention_mask.detach().cpu().tolist(),
        "topk_by_layer": {
            str(layer_idx): tensor.tolist()
            for layer_idx, tensor in topk_by_layer.items()
        },
        "model_name": getattr(wrapper.model, "name_or_path", "unknown"),
        "adapter_path": adapter_path,
        "quantized": bool(wrapper.is_bnb_quantized),
        "chat_template": getattr(wrapper.tokenizer, "chat_template", None),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, ensure_ascii=False)


def _run_single_model_subprocess(worker_config: Dict[str, Any]) -> Dict[str, Any]:
    ctx = get_context("fork")
    with tempfile.NamedTemporaryFile(
        prefix="logitdiff_worker_",
        suffix=".json",
        delete=False,
        dir="/tmp",
    ) as tmp:
        tmp_path = Path(tmp.name)

    cfg = dict(worker_config)
    cfg["output_path"] = str(tmp_path)
    proc = ctx.Process(target=_collect_single_model_worker, args=(cfg,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"Worker process failed with exit code {proc.exitcode}")

    try:
        with tmp_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def run_logitdiff_subprocess_sequential(
    *,
    base_model_path: str,
    comparison_model_path: str | None = None,
    comparison_adapter_path: str | None = None,
    config: LogitDiffRunConfig,
    output_path: str | Path | None = None,
    responses_output_path: str | Path | None = None,
) -> Dict[str, Any]:
    if config.prompt_metadata is not None and len(config.prompt_metadata) != len(config.prompts):
        raise ValueError("prompt_metadata must have the same length as prompts")

    comparison_model_path = comparison_model_path or base_model_path

    tokenizer = load_tokenizer(base_model_path)
    model_cfg = AutoConfig.from_pretrained(base_model_path)
    num_hidden_layers = int(getattr(model_cfg, "num_hidden_layers"))
    wrapper_probe = type("TokenizerProbe", (), {"blocks": [None] * num_hidden_layers})()
    resolved_layers = _resolve_layer_indices(wrapper_probe, config.layers)
    layer_indices = [abs_idx for _, abs_idx in resolved_layers]
    max_k = max(_sanitize_topk_values(config.comparison_top_ks, config.top_k))

    base_template = None
    chat_template_path = Path(base_model_path) / "chat_template.jinja"
    if chat_template_path.exists():
        base_template = chat_template_path.read_text(encoding="utf-8")

    results: Dict[str, Any] = {}
    analysis_rows: List[Dict[str, Any]] = []
    prompt_records: List[Dict[str, Any]] = []
    response_rows: List[Dict[str, Any]] = []
    effective_system_prompt: str | None = None
    tokenizer_chat_template: str | None = base_template
    base_model_name = str(base_model_path)
    finetuned_model_name = str(comparison_model_path)
    finetuned_adapter_path_seen = str(comparison_adapter_path) if comparison_adapter_path else None
    base_quantized = False
    finetuned_quantized = False

    for prompt_idx, prompt in enumerate(config.prompts):
        prompt_meta = (
            dict(config.prompt_metadata[prompt_idx])
            if config.prompt_metadata is not None
            else {}
        )
        worker_common = {
            "model_path": base_model_path,
            "chat_template": base_template,
            "prompt": prompt,
            "layer_indices": layer_indices,
            "max_k": max_k,
            "add_special_tokens": config.add_special_tokens,
            "prompt_format": config.prompt_format,
            "use_chat_template": config.use_chat_template,
            "system_prompt": config.system_prompt,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "temperature": config.temperature,
            "use_cache": config.use_cache,
            "norm_mode": config.norm_mode,
        }

        base_payload = _run_single_model_subprocess(
            {
                **worker_common,
                "adapter_path": None,
                "seed": config.seed,
            }
        )
        ft_payload = _run_single_model_subprocess(
            {
                **worker_common,
                "model_path": comparison_model_path,
                "adapter_path": comparison_adapter_path,
                "seed": None if config.seed is None else int(config.seed) + 1,
            }
        )

        generated_ids = torch.tensor(base_payload["generated_ids"], dtype=torch.long)
        attention_mask = torch.tensor(base_payload["attention_mask"], dtype=torch.long)
        ft_generated_ids = torch.tensor(ft_payload["generated_ids"], dtype=torch.long)
        prompt_len = int(base_payload["prompt_length"])
        prompt_formatted = str(base_payload["prompt_formatted"])
        if effective_system_prompt is None and config.prompt_format == "chat_template":
            effective_system_prompt = base_payload.get("effective_system_prompt")

        tokenizer_chat_template = base_payload.get("chat_template")
        base_model_name = base_payload.get("model_name", base_model_name)
        finetuned_model_name = ft_payload.get("model_name", finetuned_model_name)
        finetuned_adapter_path_seen = ft_payload.get("adapter_path", finetuned_adapter_path_seen)
        base_quantized = bool(base_payload.get("quantized", base_quantized))
        finetuned_quantized = bool(ft_payload.get("quantized", finetuned_quantized))

        prompt_record = {
            "prompt_index": prompt_idx,
            "prompt_id": prompt_meta.get("prompt_id", prompt_meta.get("id", prompt_idx)),
            "prompt": prompt,
            "prompt_rendered": prompt_formatted,
            "prompt_formatted": prompt_formatted,
            "prompt_length": prompt_len,
            "sequence_length": int(base_payload["sequence_length"]),
            "template_name": config.template_name,
            "prompt_format": config.prompt_format,
            "use_chat_template": config.use_chat_template,
            "system_prompt": config.system_prompt,
            "effective_system_prompt": effective_system_prompt,
            "metadata": prompt_meta,
        }
        prompt_records.append(prompt_record)
        base_response = tokenizer.decode(
            generated_ids[0, prompt_len:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        comparison_response = tokenizer.decode(
            ft_generated_ids[0, prompt_len:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        response_rows.append(
            {
                "prompt_index": prompt_idx,
                "prompt_id": prompt_record["prompt_id"],
                "prompt": prompt,
                "template_name": config.template_name,
                "max_new_tokens": config.max_new_tokens,
                "base_response": base_response,
                "comparison_response": comparison_response,
            }
        )

        for layer_rel, layer_abs in resolved_layers:
            key = str(layer_rel)
            entry = _compare_topk_predictions(
                tokenizer=tokenizer,
                input_ids=generated_ids,
                attention_mask=attention_mask,
                prompt_len=prompt_len,
                top_k=config.top_k,
                comparison_top_ks=config.comparison_top_ks,
                layer_rel=layer_rel,
                layer_abs=layer_abs,
                topk_ids_a=torch.tensor(base_payload["topk_by_layer"][str(layer_abs)], dtype=torch.long),
                topk_ids_b=torch.tensor(ft_payload["topk_by_layer"][str(layer_abs)], dtype=torch.long),
                base_generated_ids=generated_ids,
                ft_generated_ids=ft_generated_ids,
            )
            entry["prompt"] = prompt
            entry["prompt_index"] = prompt_idx
            entry["prompt_id"] = prompt_record["prompt_id"]
            entry["prompt_rendered"] = prompt_formatted
            entry["prompt_formatted"] = prompt_formatted
            entry["prompt_length"] = prompt_len
            entry["sequence_length"] = int(base_payload["sequence_length"])
            entry["norm_mode"] = config.norm_mode
            entry["template_name"] = config.template_name
            entry["prompt_metadata"] = prompt_meta
            results.setdefault(key, []).append(entry)
            for position in entry["positions"]:
                analysis_rows.append(
                    {
                        "prompt_index": prompt_idx,
                        "prompt_id": prompt_record["prompt_id"],
                        "prompt": prompt,
                        "template_name": config.template_name,
                        "prompt_format": config.prompt_format,
                        "use_chat_template": config.use_chat_template,
                        "system_prompt": config.system_prompt,
                        "effective_system_prompt": effective_system_prompt,
                        "norm_mode": config.norm_mode,
                        "layer_relative": entry["layer_relative"],
                        "layer_absolute": entry["layer_absolute"],
                        "position": position["position"],
                        "position_kind": position["position_kind"],
                        "is_generated": position["is_generated"],
                        "input_token": position["input_token"],
                        "input_token_id": position["input_token_id"],
                        "base_generated_token": position["base_generated_token"],
                        "base_generated_token_id": position["base_generated_token_id"],
                        "ft_generated_token": position["ft_generated_token"],
                        "ft_generated_token_id": position["ft_generated_token_id"],
                        "base_top1_token": position["base_top1_token"],
                        "base_top1_token_id": position["base_top1_token_id"],
                        "ft_top1_token": position["ft_top1_token"],
                        "ft_top1_token_id": position["ft_top1_token_id"],
                        "top1_match": position["top1_match"],
                        "top1_jaccard": position["top1_jaccard"],
                        "top5_jaccard": position["top5_jaccard"],
                        "top10_jaccard": position["top10_jaccard"],
                        "iou": position["iou"],
                        "topk_predictions": position["topk_predictions"],
                    }
                )

        gc.collect()

    payload = _build_rich_payload(
        config=config,
        tokenizer_chat_template=tokenizer_chat_template,
        base_model_name=base_model_name,
        finetuned_model_name=finetuned_model_name,
        adapter_path=finetuned_adapter_path_seen,
        base_quantized=base_quantized,
        finetuned_quantized=finetuned_quantized,
        effective_system_prompt=effective_system_prompt,
        prompt_records=prompt_records,
        analysis_rows=analysis_rows,
        results=results,
    )
    payload["metadata"]["adapter_path"] = finetuned_adapter_path_seen
    payload["metadata"]["comparison_model_path"] = str(comparison_model_path)
    payload["metadata"]["do_sample"] = config.do_sample
    payload["metadata"]["temperature"] = config.temperature
    payload["metadata"]["use_cache"] = config.use_cache
    if responses_output_path is not None:
        responses_path = Path(responses_output_path)
        responses_path.parent.mkdir(parents=True, exist_ok=True)
        with responses_path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(response_rows), f, indent=2, ensure_ascii=False)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(payload), f, indent=2, ensure_ascii=False)
    return payload

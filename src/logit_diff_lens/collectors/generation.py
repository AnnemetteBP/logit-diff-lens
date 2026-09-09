from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import torch

from ..schemas.generation_outputs import (
    GenerationDatasetExample,
    GenerationDecodeArtifact,
    GenerationDecodeDatasetArtifact,
    GenerationLayerRecord,
)
from ..validation import (
    validate_generation_decode_artifact,
    validate_generation_decode_dataset_artifact,
)
from ..wrappers import (
    CustomGenerationLensWrapper,
    GenerateLensWrapper,
    lmhead_project,
    normalize_activations,
)
from .prompt import _format_generation_prompt


@dataclass
class GenerationActivationCollectorConfig:
    prompt: str = ""
    use_chat_template: bool = False
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain"
    system_prompt: str | None = None
    add_special_tokens: bool = False
    analyze_special_tokens: bool = False
    truncation: bool = False
    max_length: int | None = None
    padding: bool | str | None = None
    force_include_input: bool = True
    force_include_output: bool = True
    normalize_embedding_for_readout: bool = False
    norm_modes: tuple[str, ...] = ("raw", "unit_norm", "eps_norm", "model_norm")
    collect_components: bool = False
    project_component_logits: bool = False
    max_new_tokens: int = 10
    do_sample: bool = True
    temperature: float = 1.0
    seed: int | None = None


def _build_collection_text_and_kind(
    row: Dict[str, Any],
    *,
    text_field: str,
) -> tuple[str, str]:
    source_kind = str(row.get("source_kind", ""))
    model_role = str(row.get("model_role", ""))
    prompt = str(row.get("prompt_clean") or row.get("prompt") or row.get("source_prompt") or "").strip()
    response_only = row.get("response_only") or row.get("response_text")

    if source_kind == "model_response" and response_only is not None:
        response = str(response_only).strip()
        if prompt and response:
            combined = f"{prompt} {response}"
        else:
            combined = response or prompt
        if model_role == "base" or row.get("variant") == "base_response":
            continuation_kind = "prompt_plus_base_response"
        elif model_role == "finetuned" or row.get("variant") == "finetuned_response":
            continuation_kind = "prompt_plus_finetuned_response"
        else:
            continuation_kind = "prompt_plus_response"
        return combined, continuation_kind

    text_value = str(row.get(text_field, "")).strip()
    if text_value:
        variant = str(row.get("variant", "")).strip()
        if variant == "base_response":
            return text_value, "prompt_plus_base_response"
        if variant == "finetuned_response":
            return text_value, "prompt_plus_finetuned_response"
        return text_value, "prompt_only"
    if prompt:
        return prompt, "prompt_only"
    raise ValueError(
        f"Could not determine collection text from row id={row.get('id')} using text_field='{text_field}'"
    )


def _detach_to_cpu_fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def _slice_sequence_window(
    tensor: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    return tensor[:, start:end].detach().cpu()


@torch.no_grad()
def _collect_generation_for_analysis(
    arch_wrapper: CustomGenerationLensWrapper | GenerateLensWrapper,
    prompts: List[str],
    prompt_ids: List[int | str] | None = None,
    batch_index: int = 0,
    use_chat_template: bool = False,
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain",
    system_prompt: str | None = None,
    add_special_tokens: bool = False,
    analyze_special_tokens: bool = False,
    truncation: bool = False,
    max_length: int | None = None,
    padding: bool | str | None = None,
    device: str | None = None,
    force_include_input: bool = True,
    force_include_output: bool = True,
    normalize_embedding_for_readout: bool = False,
    save_path=None,
    norm_modes: Tuple[str, ...] = ("raw", "unit_norm", "eps_norm", "model_norm"),
    collect_components: bool = False,
    project_component_logits: bool = False,
    dataset: str | None = None,
    max_new_tokens: int = 10,
    do_sample: bool = True,
    temperature: float = 1.0,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    device = device or arch_wrapper.model_device
    model = arch_wrapper.model
    tokenizer = arch_wrapper.tokenizer
    model.eval()

    if not arch_wrapper.is_bnb_quantized:
        model = model.to(device)

    rows: list[dict[str, Any]] = []
    if prompt_ids is not None and len(prompt_ids) != len(prompts):
        raise ValueError("prompt_ids must match prompts length when provided")

    for prompt_id, prompt_text in enumerate(prompts):
        logical_prompt_id = prompt_ids[prompt_id] if prompt_ids is not None else prompt_id
        prompt_formatted = _format_generation_prompt(
            arch_wrapper,
            prompt_text,
            prompt_format=prompt_format,
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
        )
        hook_buffers: Dict[str, Dict[int, List[torch.Tensor]]] = {
            "attention_outputs": {},
            "mlp_outputs": {},
        }
        hook_handles: List[Any] = []

        if collect_components:
            def _save_component_hook(component_name: str, layer_idx: int):
                def fn(module, inp, out):
                    tensor = arch_wrapper._extract_tensor(out)
                    if tensor is None:
                        return out
                    hook_buffers.setdefault(component_name, {}).setdefault(layer_idx, []).append(
                        _detach_to_cpu_fp32(tensor)
                    )
                    return out
                return fn

            component_registry = getattr(arch_wrapper, "component_registry", {}) or {}
            for layer_idx in range(len(arch_wrapper.blocks)):
                attn_entry = component_registry.get(f"attention_{layer_idx:02d}")
                mlp_entry = component_registry.get(f"mlp_{layer_idx:02d}")
                if attn_entry is not None:
                    hook_handles.append(
                        attn_entry["module"].register_forward_hook(
                            _save_component_hook("attention_outputs", layer_idx)
                        )
                    )
                if mlp_entry is not None:
                    hook_handles.append(
                        mlp_entry["module"].register_forward_hook(
                            _save_component_hook("mlp_outputs", layer_idx)
                        )
                    )

        inputs = arch_wrapper.tokenize_inputs(
            texts=prompt_formatted,
            device=device,
            add_special_tokens=add_special_tokens and not use_chat_template,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        try:
            gen_out = arch_wrapper.forward_pass(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                seed=seed,
            )
        finally:
            for hook in hook_handles:
                try:
                    hook.remove()
                except Exception:
                    pass

        tokens = gen_out["tokens"]
        token_attention_mask = gen_out.get("attention_mask")
        acts_steps = gen_out["activations"]
        token_ids = tokens[0]
        seq_len = token_ids.shape[0]

        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        has_bos = bos_token_id is not None and token_ids[0].item() == bos_token_id
        has_eos = eos_token_id is not None and token_ids[-1].item() == eos_token_id

        if analyze_special_tokens:
            start, end = 0, seq_len
        else:
            start = 1 if has_bos else 0
            end = seq_len - 1 if has_eos else seq_len
        if end <= start:
            start, end = 0, seq_len

        if token_attention_mask is None:
            token_attention_mask = torch.ones_like(tokens, device=tokens.device)

        full_tokens_view = tokens[:, start:end].detach().cpu()
        full_attention_mask_view = token_attention_mask[:, start:end].detach().cpu()

        for step_idx, acts in enumerate(acts_steps):
            prefix_len = int(acts["embedding"].shape[1]) if "embedding" in acts else int(tokens.shape[1])
            prefix_end = min(end, prefix_len)
            if prefix_end <= start:
                prefix_start = 0
                prefix_end = prefix_len
            else:
                prefix_start = start
            tokens_view = tokens[:, prefix_start:prefix_end].detach().cpu()
            token_attention_mask_view = token_attention_mask[:, prefix_start:prefix_end].detach().cpu()
            step_attention_outputs = {
                idx: values[step_idx]
                for idx, values in hook_buffers["attention_outputs"].items()
                if step_idx < len(values)
            }
            step_mlp_outputs = {
                idx: values[step_idx]
                for idx, values in hook_buffers["mlp_outputs"].items()
                if step_idx < len(values)
            }

            if force_include_input and "embedding" in acts:
                hidden_full = acts["embedding"]
                rec = {
                    "prompt_id": logical_prompt_id,
                    "prompt_text": prompt_text,
                    "prompt_formatted": prompt_formatted,
                    "batch_index": batch_index,
                    "step": step_idx,
                    "layer_index": -1,
                    "layer_name": "embedding",
                    "tokens": tokens_view,
                    "attention_mask": token_attention_mask_view,
                    "full_tokens": full_tokens_view,
                    "full_attention_mask": full_attention_mask_view,
                }
                for mode in norm_modes:
                    h_norm = normalize_activations(
                        x=hidden_full.clone(),
                        mode=mode,
                        block="embedding",
                        layer_idx=-1,
                        normalize_embedding_for_readout=normalize_embedding_for_readout,
                        model_device=arch_wrapper.model_device,
                        model_dtype=arch_wrapper.model_dtype,
                        final_norm=arch_wrapper.final_norm,
                    )
                    rec[f"hidden_{mode}"] = (
                        arch_wrapper.save_to_fp32(h_norm)[:, prefix_start:prefix_end].detach().cpu()
                        if arch_wrapper.fp32_save
                        else _slice_sequence_window(h_norm, prefix_start, prefix_end)
                    )
                    logits_full, _ = lmhead_project(
                        x=h_norm,
                        lm_head=arch_wrapper.lm_head,
                        stable=arch_wrapper.stable,
                        model_device=arch_wrapper.model_device,
                    )
                    rec[f"logits_{mode}"] = logits_full[:, start:end].detach().cpu()
                rows.append(rec)

            layers = sorted(
                (
                    arch_wrapper.layer_registry[name]["idx"],
                    name,
                    acts[name],
                )
                for name in acts
                if name in arch_wrapper.layer_registry
                and arch_wrapper.layer_registry[name]["type"] == "block"
            )

            last_act = None
            last_idx = None
            for layer_idx, name, act in layers:
                hidden_full = act
                attention_output_full = step_attention_outputs.get(layer_idx)
                mlp_output_full = step_mlp_outputs.get(layer_idx)
                rec = {
                    "prompt_id": logical_prompt_id,
                    "prompt_text": prompt_text,
                    "prompt_formatted": prompt_formatted,
                    "batch_index": batch_index,
                    "step": step_idx,
                    "layer_index": layer_idx,
                    "layer_name": f"layer_{layer_idx}",
                    "tokens": tokens_view,
                    "attention_mask": token_attention_mask_view,
                    "full_tokens": full_tokens_view,
                    "full_attention_mask": full_attention_mask_view,
                }
                for mode in norm_modes:
                    h_norm = normalize_activations(
                        x=hidden_full.clone(),
                        mode=mode,
                        block="block",
                        layer_idx=layer_idx,
                        normalize_embedding_for_readout=normalize_embedding_for_readout,
                        model_device=arch_wrapper.model_device,
                        model_dtype=arch_wrapper.model_dtype,
                        final_norm=arch_wrapper.final_norm,
                    )
                    rec[f"hidden_{mode}"] = (
                        arch_wrapper.save_to_fp32(h_norm)[:, prefix_start:prefix_end].detach().cpu()
                        if arch_wrapper.fp32_save
                        else _slice_sequence_window(h_norm, prefix_start, prefix_end)
                    )
                    logits_full, _ = lmhead_project(
                        x=h_norm,
                        lm_head=arch_wrapper.lm_head,
                        stable=arch_wrapper.stable,
                        model_device=arch_wrapper.model_device,
                    )
                    rec[f"logits_{mode}"] = logits_full[:, start:end].detach().cpu()

                if collect_components and attention_output_full is not None:
                    rec["attention_output"] = attention_output_full[:, start:end].detach().cpu()
                    if project_component_logits:
                        for mode in norm_modes:
                            attn_norm = normalize_activations(
                                x=attention_output_full.clone(),
                                mode=mode,
                                block="block",
                                layer_idx=layer_idx,
                                normalize_embedding_for_readout=normalize_embedding_for_readout,
                                model_device=arch_wrapper.model_device,
                                model_dtype=arch_wrapper.model_dtype,
                                final_norm=arch_wrapper.final_norm,
                            )
                            attn_logits_full, _ = lmhead_project(
                                x=attn_norm,
                                lm_head=arch_wrapper.lm_head,
                                stable=arch_wrapper.stable,
                                model_device=arch_wrapper.model_device,
                            )
                            rec[f"attention_logits_{mode}"] = attn_logits_full[:, start:end].detach().cpu()

                if collect_components and mlp_output_full is not None:
                    rec["mlp_output"] = mlp_output_full[:, start:end].detach().cpu()
                    if project_component_logits:
                        for mode in norm_modes:
                            mlp_norm = normalize_activations(
                                x=mlp_output_full.clone(),
                                mode=mode,
                                block="block",
                                layer_idx=layer_idx,
                                normalize_embedding_for_readout=normalize_embedding_for_readout,
                                model_device=arch_wrapper.model_device,
                                model_dtype=arch_wrapper.model_dtype,
                                final_norm=arch_wrapper.final_norm,
                            )
                            mlp_logits_full, _ = lmhead_project(
                                x=mlp_norm,
                                lm_head=arch_wrapper.lm_head,
                                stable=arch_wrapper.stable,
                                model_device=arch_wrapper.model_device,
                            )
                            rec[f"mlp_logits_{mode}"] = mlp_logits_full[:, start:end].detach().cpu()

                rows.append(rec)
                last_act = act
                last_idx = layer_idx

            if force_include_output and last_act is not None and last_idx is not None:
                out_idx = last_idx + 1
                rec = {
                    "prompt_id": logical_prompt_id,
                    "prompt_text": prompt_text,
                    "prompt_formatted": prompt_formatted,
                    "batch_index": batch_index,
                    "step": step_idx,
                    "layer_index": out_idx,
                    "layer_name": "output",
                    "tokens": tokens_view,
                    "attention_mask": token_attention_mask_view,
                    "full_tokens": full_tokens_view,
                    "full_attention_mask": full_attention_mask_view,
                }
                for mode in norm_modes:
                    h_norm = normalize_activations(
                        x=last_act.clone(),
                        mode=mode,
                        block="output",
                        layer_idx=out_idx,
                        normalize_embedding_for_readout=normalize_embedding_for_readout,
                        model_device=arch_wrapper.model_device,
                        model_dtype=arch_wrapper.model_dtype,
                        final_norm=arch_wrapper.final_norm,
                    )
                    rec[f"hidden_{mode}"] = (
                        arch_wrapper.save_to_fp32(h_norm)[:, prefix_start:prefix_end].detach().cpu()
                        if arch_wrapper.fp32_save
                        else _slice_sequence_window(h_norm, prefix_start, prefix_end)
                    )
                    logits_full, _ = lmhead_project(
                        x=h_norm,
                        lm_head=arch_wrapper.lm_head,
                        stable=arch_wrapper.stable,
                        model_device=arch_wrapper.model_device,
                    )
                    rec[f"logits_{mode}"] = logits_full[:, start:end].detach().cpu()
                rows.append(rec)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "rows": rows,
                "metadata": {
                    "model_name": getattr(model, "name_or_path", "unknown"),
                    "arch": getattr(arch_wrapper, "arch", "unknown"),
                    "batch_index": batch_index,
                    "dataset": dataset,
                    "use_chat_template": use_chat_template,
                    "prompt_format": prompt_format,
                    "system_prompt": system_prompt,
                    "force_include_input": force_include_input,
                    "force_include_output": force_include_output,
                    "normalize_embedding_for_readout": normalize_embedding_for_readout,
                    "norm_modes": list(norm_modes),
                    "collect_components": collect_components,
                    "project_component_logits": project_component_logits,
                    "generation": True,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "seed": seed,
                },
            },
            save_path,
        )
    return rows


def collect_generation_activations(
    arch_wrapper: CustomGenerationLensWrapper | GenerateLensWrapper,
    config: GenerationActivationCollectorConfig,
) -> Dict[str, Any]:
    rows = _collect_generation_for_analysis(
        arch_wrapper=arch_wrapper,
        prompts=[config.prompt],
        prompt_ids=[0],
        batch_index=0,
        use_chat_template=config.use_chat_template,
        prompt_format=config.prompt_format,
        system_prompt=config.system_prompt,
        add_special_tokens=config.add_special_tokens,
        analyze_special_tokens=config.analyze_special_tokens,
        truncation=config.truncation,
        max_length=config.max_length,
        padding=config.padding,
        force_include_input=config.force_include_input,
        force_include_output=config.force_include_output,
        normalize_embedding_for_readout=config.normalize_embedding_for_readout,
        save_path=None,
        norm_modes=config.norm_modes,
        collect_components=config.collect_components,
        project_component_logits=config.project_component_logits,
        dataset=None,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        seed=config.seed,
    )
    artifact = GenerationDecodeArtifact(
        rows=[GenerationLayerRecord.from_dict(row) for row in rows],
        use_chat_template=bool(config.use_chat_template),
        prompt_format=config.prompt_format,
        system_prompt=config.system_prompt,
        force_include_input=bool(config.force_include_input),
        force_include_output=bool(config.force_include_output),
        normalize_embedding_for_readout=bool(config.normalize_embedding_for_readout),
        norm_modes=list(config.norm_modes),
        collect_components=bool(config.collect_components),
        project_component_logits=bool(config.project_component_logits),
        max_new_tokens=int(config.max_new_tokens),
        do_sample=bool(config.do_sample),
        temperature=float(config.temperature),
        seed=config.seed,
        truncation=bool(config.truncation),
        max_length=config.max_length,
        padding=config.padding,
        batch_size=1,
        batch_semantics="single_sequence_per_row",
        row_semantics="one_row_per_step_per_layer",
        metadata={
            "add_special_tokens": bool(config.add_special_tokens),
            "analyze_special_tokens": bool(config.analyze_special_tokens),
            "truncation": bool(config.truncation),
            "max_length": config.max_length,
            "padding": config.padding,
            "force_include_input": bool(config.force_include_input),
            "force_include_output": bool(config.force_include_output),
            "normalize_embedding_for_readout": bool(config.normalize_embedding_for_readout),
            "norm_modes": list(config.norm_modes),
            "collect_components": bool(config.collect_components),
            "project_component_logits": bool(config.project_component_logits),
            "prompt_format": config.prompt_format,
            "use_chat_template": bool(config.use_chat_template),
            "system_prompt": config.system_prompt,
            "max_new_tokens": int(config.max_new_tokens),
            "do_sample": bool(config.do_sample),
            "temperature": float(config.temperature),
            "seed": config.seed,
        },
    )
    validate_generation_decode_artifact(artifact)
    return artifact.to_dict()


@torch.no_grad()
def collect_generation_activation_dataset_incremental(
    *,
    wrapper: CustomGenerationLensWrapper | GenerateLensWrapper,
    dataset_path: str | Path,
    output_path: str | Path,
    text_field: str = "analysis_text",
    label_field: str = "label",
    use_chat_template: bool = False,
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain",
    system_prompt: str | None = None,
    add_special_tokens: bool = False,
    analyze_special_tokens: bool = False,
    truncation: bool = False,
    max_length: int | None = None,
    padding: bool | str | None = None,
    force_include_input: bool = True,
    force_include_output: bool = True,
    normalize_embedding_for_readout: bool = False,
    norm_modes: tuple[str, ...] = ("raw", "unit_norm", "eps_norm", "model_norm"),
    collect_components: bool = False,
    project_component_logits: bool = False,
    max_new_tokens: int = 10,
    do_sample: bool = True,
    temperature: float = 1.0,
    seed: int | None = None,
    batch_size: int = 10,
    model_key: str = "model",
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Dataset {dataset_path} is empty")

    result_rows: List[Dict[str, Any]] = []
    num_batches = (len(rows) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(rows))
        batch_rows = rows[start:end]

        prompts: List[str] = []
        meta_rows: List[Dict[str, Any]] = []
        for row in batch_rows:
            collection_text, continuation_kind = _build_collection_text_and_kind(row, text_field=text_field)
            prompts.append(collection_text)
            meta_rows.append(
                {
                    **row,
                    "collection_text_field": text_field,
                    "collection_prompt_format": prompt_format,
                    "collection_system_prompt": system_prompt,
                    "collection_use_chat_template": use_chat_template,
                    "collection_concat_strategy": "prompt_space_response_for_model_responses",
                    "collection_text": collection_text,
                    "continuation_kind": continuation_kind,
                    "label": row.get(label_field),
                }
            )

        batch_records = _collect_generation_for_analysis(
            arch_wrapper=wrapper,
            prompts=prompts,
            prompt_ids=[start + local_idx for local_idx in range(len(prompts))],
            batch_index=batch_idx,
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
            normalize_embedding_for_readout=normalize_embedding_for_readout,
            save_path=None,
            norm_modes=norm_modes,
            collect_components=collect_components,
            project_component_logits=project_component_logits,
            dataset=str(dataset_path),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            seed=seed,
        )

        rows_by_prompt: Dict[int | str, List[Dict[str, Any]]] = {}
        for rec in batch_records:
            rows_by_prompt.setdefault(rec["prompt_id"], []).append(rec)

        for local_idx, meta in enumerate(meta_rows):
            result_rows.append(
                {
                    **meta,
                    "generated_rows": rows_by_prompt.get(start + local_idx, []),
                }
            )

        del batch_records, batch_rows, prompts, meta_rows
        torch.cuda.empty_cache()
        gc.collect()

    artifact = GenerationDecodeDatasetArtifact(
        rows=[
            GenerationDatasetExample(
                metadata={k: v for k, v in row.items() if k != "generated_rows"},
                generated_rows=[GenerationLayerRecord.from_dict(item) for item in row.get("generated_rows", [])],
            )
            for row in result_rows
        ],
        dataset_path=str(dataset_path),
        text_field=text_field,
        label_field=label_field,
        model_key=model_key,
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
        normalize_embedding_for_readout=normalize_embedding_for_readout,
        norm_modes=list(norm_modes),
        collect_components=collect_components,
        project_component_logits=project_component_logits,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        seed=seed,
        num_examples=len(rows),
        num_batches=num_batches,
        requested_batch_size=batch_size,
        metadata={
            "dataset_path": str(dataset_path),
            "text_field": text_field,
            "label_field": label_field,
            "model_key": model_key,
            "use_chat_template": bool(use_chat_template),
            "prompt_format": prompt_format,
            "system_prompt": system_prompt,
            "add_special_tokens": bool(add_special_tokens),
            "analyze_special_tokens": bool(analyze_special_tokens),
            "truncation": bool(truncation),
            "max_length": max_length,
            "padding": padding,
            "force_include_input": bool(force_include_input),
            "force_include_output": bool(force_include_output),
            "normalize_embedding_for_readout": bool(normalize_embedding_for_readout),
            "norm_modes": list(norm_modes),
            "collect_components": bool(collect_components),
            "project_component_logits": bool(project_component_logits),
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "seed": seed,
        },
    )
    validate_generation_decode_dataset_artifact(artifact)
    final_payload = artifact.to_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_payload, output_path)
    return final_payload


__all__ = [
    "GenerationActivationCollectorConfig",
    "collect_generation_activation_dataset_incremental",
    "collect_generation_activations",
]

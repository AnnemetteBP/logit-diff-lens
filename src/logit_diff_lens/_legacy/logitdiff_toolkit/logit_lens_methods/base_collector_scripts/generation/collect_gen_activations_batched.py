from __future__ import annotations

from dataclasses import dataclass
import json
from typing import List, Dict, Tuple, Any, Literal
import gc
from pathlib import Path
import torch

from ...wrapper import (
    CustomGenerationLensWrapper,
    GenerateLensWrapper,
    normalize_activations,
    lmhead_project
)


@dataclass
class GenerationActivationCollectorConfig:
    prompt: str = ""
    use_chat_template: bool = False
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain"
    system_prompt: str | None = None
    add_special_tokens: bool = False
    analyze_special_tokens: bool = False
    force_include_input: bool = True
    force_include_output: bool = True
    norm_modes: tuple[str, ...] = ("raw", "unit_norm", "eps_norm", "model_norm")
    collect_components: bool = False
    project_component_logits: bool = False
    max_new_tokens: int = 10


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


def collect_generation_activations(
    arch_wrapper:"CustomGenerationLensWrapper | GenerateLensWrapper",
    config: GenerationActivationCollectorConfig,
) -> Dict[str, Any]:
    rows = _collect_generation_for_analysis(
        arch_wrapper=arch_wrapper,
        prompts=[config.prompt],
        batch_index=0,
        add_special_tokens=config.add_special_tokens,
        analyze_special_tokens=config.analyze_special_tokens,
        force_include_input=config.force_include_input,
        force_include_output=config.force_include_output,
        save_path=None,
        norm_modes=config.norm_modes,
        collect_components=config.collect_components,
        project_component_logits=config.project_component_logits,
        dataset=None,
        max_new_tokens=config.max_new_tokens,
    )
    return {
        "rows": rows,
        "force_include_input": bool(config.force_include_input),
        "force_include_output": bool(config.force_include_output),
        "norm_modes": list(config.norm_modes),
        "collect_components": bool(config.collect_components),
        "project_component_logits": bool(config.project_component_logits),
        "max_new_tokens": int(config.max_new_tokens),
    }



@torch.no_grad()
def _collect_generation_for_analysis(
    arch_wrapper:"CustomGenerationLensWrapper | GenerateLensWrapper",
    prompts:List[str],
    batch_index:int=0,
    add_special_tokens:bool=False,
    analyze_special_tokens:bool=False,
    device:str|None=None,
    force_include_input:bool=True,
    force_include_output:bool=True,
    save_path=None,
    norm_modes:Tuple[str, ...]=("raw", "unit_norm", "eps_norm", "model_norm"),
    collect_components:bool=False,
    project_component_logits:bool=False,
    dataset:str|None=None,
    max_new_tokens:int=10,
) -> List[Dict[str, Any]]:

    device = arch_wrapper.model_device
    model = arch_wrapper.model
    tokenizer = arch_wrapper.tokenizer
    model.eval()

    if not arch_wrapper.is_bnb_quantized:
        model = model.to(device)

    rows = []

    for b, text in enumerate(prompts):
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
                        tensor.detach().to(device="cpu", dtype=torch.float32).clone()
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
            texts=text,
            device=device,
            add_special_tokens=add_special_tokens,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        try:
            gen_out = arch_wrapper.forward_pass(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
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

        ids = tokens[0]
        T = ids.shape[0]

        has_bos = tokenizer.bos_token_id is not None and ids[0].item() == tokenizer.bos_token_id
        has_eos = tokenizer.eos_token_id is not None and ids[-1].item() == tokenizer.eos_token_id

        if analyze_special_tokens:
            start, end = 0, T
        else:
            start = 1 if has_bos else 0
            end = T - 1 if has_eos else T

        if end <= start:
            start, end = 0, T

        if token_attention_mask is None:
            token_attention_mask = torch.ones_like(tokens, device=tokens.device)

        tokens_view = tokens[:, start:end].cpu()
        token_attention_mask_view = token_attention_mask[:, start:end].detach().cpu()

        # ============================================================
        # LOOP OVER GENERATION STEPS
        # ============================================================
        for step_idx, acts in enumerate(acts_steps):
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

            # ---------------- EMBEDDING ----------------
            if force_include_input and "embedding" in acts:

                hidden_full = acts["embedding"]
                hidden_view = hidden_full[:, start:end]

                rec = {
                    "prompt_id": b,
                    "prompt_text": text,
                    "batch_index": batch_index,
                    "step": step_idx,
                    "layer_index": -1,
                    "layer_name": "embedding",
                    "tokens": tokens_view,
                    "attention_mask": token_attention_mask_view,
                }

                for m in norm_modes:
                    h_norm = normalize_activations(
                        x=hidden_full.clone(), mode=m, block="embedding", layer_idx=-1,
                        model_device=arch_wrapper.model_device, model_dtype=arch_wrapper.model_dtype, final_norm=arch_wrapper.final_norm
                    )
                    rec[f"hidden_{m}"] = (
                        arch_wrapper.save_to_fp32(h_norm)
                        if arch_wrapper.fp32_save
                        else h_norm.detach().cpu()
                    )
                    
                    logits_full, _ = lmhead_project(x=h_norm, lm_head=arch_wrapper.lm_head, stable=arch_wrapper.stable, model_device=arch_wrapper.model_device)
                    rec[f"logits_{m}"] = logits_full[:, start:end].detach().cpu()

                rows.append(rec)

            # ---------------- LAYERS ----------------
            layers = sorted(
                (
                    arch_wrapper.layer_registry[n]["idx"],
                    n,
                    acts[n],
                )
                for n in acts
                if n in arch_wrapper.layer_registry
                and arch_wrapper.layer_registry[n]["type"] == "block"
            )

            last_act = None
            last_idx = None

            for idx, name, act in layers:

                hidden_full = act
                hidden_view = hidden_full[:, start:end]
                attention_output_full = step_attention_outputs.get(idx)
                mlp_output_full = step_mlp_outputs.get(idx)

                rec = {
                    "prompt_id": b,
                    "prompt_text": text,
                    "batch_index": batch_index,
                    "step": step_idx,
                    "layer_index": idx,
                    "layer_name": f"layer_{idx}",
                    "tokens": tokens_view,
                    "attention_mask": token_attention_mask_view,
                }

                for m in norm_modes:
                    h_norm = normalize_activations(
                        x=hidden_full.clone(), mode=m, block="block", layer_idx=idx,
                        model_device=arch_wrapper.model_device, model_dtype=arch_wrapper.model_dtype, final_norm=arch_wrapper.final_norm
                    )
                    rec[f"hidden_{m}"] = (
                        arch_wrapper.save_to_fp32(h_norm)
                        if arch_wrapper.fp32_save
                        else h_norm.detach().cpu()
                    )

                    logits_full, _ = lmhead_project(x=h_norm, lm_head=arch_wrapper.lm_head, stable=arch_wrapper.stable, model_device=arch_wrapper.model_device)
                    rec[f"logits_{m}"] = logits_full[:, start:end].detach().cpu()

                if collect_components and attention_output_full is not None:
                    rec["attention_output"] = attention_output_full[:, start:end].detach().cpu()
                    if project_component_logits:
                        for m in norm_modes:
                            attn_norm = normalize_activations(
                                x=attention_output_full.clone(), mode=m, block="block", layer_idx=idx,
                                model_device=arch_wrapper.model_device, model_dtype=arch_wrapper.model_dtype, final_norm=arch_wrapper.final_norm
                            )
                            attn_logits_full, _ = lmhead_project(
                                x=attn_norm,
                                lm_head=arch_wrapper.lm_head,
                                stable=arch_wrapper.stable,
                                model_device=arch_wrapper.model_device,
                            )
                            rec[f"attention_logits_{m}"] = attn_logits_full[:, start:end].detach().cpu()

                if collect_components and mlp_output_full is not None:
                    rec["mlp_output"] = mlp_output_full[:, start:end].detach().cpu()
                    if project_component_logits:
                        for m in norm_modes:
                            mlp_norm = normalize_activations(
                                x=mlp_output_full.clone(), mode=m, block="block", layer_idx=idx,
                                model_device=arch_wrapper.model_device, model_dtype=arch_wrapper.model_dtype, final_norm=arch_wrapper.final_norm
                            )
                            mlp_logits_full, _ = lmhead_project(
                                x=mlp_norm,
                                lm_head=arch_wrapper.lm_head,
                                stable=arch_wrapper.stable,
                                model_device=arch_wrapper.model_device,
                            )
                            rec[f"mlp_logits_{m}"] = mlp_logits_full[:, start:end].detach().cpu()

                rows.append(rec)
                last_act = act
                last_idx = idx

            # ---------------- OUTPUT ----------------
            if force_include_output and last_act is not None:

                hidden_full = last_act
                hidden_view = hidden_full[:, start:end]
                out_idx = last_idx + 1

                rec = {
                    "prompt_id": b,
                    "prompt_text": text,
                    "batch_index": batch_index,
                    "step": step_idx,
                    "layer_index": out_idx
                    ,"layer_name": "output",
                    "tokens": tokens_view,
                    "attention_mask": token_attention_mask_view,
                }

                for m in norm_modes:
                    h_norm = normalize_activations(
                        x=hidden_full.clone(), mode=m, block="output", layer_idx=out_idx,
                        model_device=arch_wrapper.model_device, model_dtype=arch_wrapper.model_dtype, final_norm=arch_wrapper.final_norm
                    )
                    rec[f"hidden_{m}"] = (
                        arch_wrapper.save_to_fp32(h_norm)
                        if arch_wrapper.fp32_save
                        else h_norm.detach().cpu()
                    )

                    logits_full, _ = lmhead_project(x=h_norm, lm_head=arch_wrapper.lm_head, stable=arch_wrapper.stable, model_device=arch_wrapper.model_device)
                    rec[f"logits_{m}"] = logits_full[:, start:end].detach().cpu()

                rows.append(rec)

    # ---------------- SAVE ----------------
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
                    "force_include_input": force_include_input,
                    "force_include_output": force_include_output,
                    "norm_modes": list(norm_modes),
                    "collect_components": collect_components,
                    "project_component_logits": project_component_logits,
                    "generation": True,
                },
            },
            save_path,
        )

    return rows



@torch.no_grad()
def collect_generation_for_analysis(
    arch_wrapper:"CustomGenerationLensWrapper | GenerateLensWrapper",
    all_prompts:List[str],
    batch_size:int=10,
    max_new_tokens:int=10,
    save_prefix:str="gen_analysis",
    output_path:str|Path|None=None,
    add_special_tokens:bool=False,
    analyze_special_tokens:bool=False,
    force_include_input:bool=True,
    force_include_output:bool=True,
    device:str|None=None,
    norm_modes:Tuple[str, ...]=("raw", "unit_norm", "eps_norm", "model_norm"),
    collect_components:bool=False,
    project_component_logits:bool=False,
    dataset:str="dataset",
):
    output_path = Path(output_path) if output_path is not None else Path(f"{save_prefix}.pt")
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    result_rows: List[Dict[str, Any]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] Processing {len(all_prompts)} prompts in {num_batches} batches")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(all_prompts))
        batch_prompts = all_prompts[start:end]
        print(f"[batch {batch_idx+1}/{num_batches}]")

        try:
            rows = _collect_generation_for_analysis(
                arch_wrapper=arch_wrapper,
                prompts=batch_prompts,
                batch_index=batch_idx,
                add_special_tokens=add_special_tokens,
                analyze_special_tokens=analyze_special_tokens,
                device=device,
                force_include_input=force_include_input,
                force_include_output=force_include_output,
                save_path=None,
                norm_modes=norm_modes,
                collect_components=collect_components,
                project_component_logits=project_component_logits,
                dataset=dataset,
                max_new_tokens=max_new_tokens,
            )

        except RuntimeError as e:
            print(f"[ERROR] Batch {batch_idx} failed: {e}")
            continue

        result_rows.extend(rows)

        del rows, batch_prompts
        torch.cuda.empty_cache()
        gc.collect()

    final_payload = {
        "dataset": dataset,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "force_include_input": force_include_input,
        "force_include_output": force_include_output,
        "norm_modes": list(norm_modes),
        "collect_components": collect_components,
        "project_component_logits": project_component_logits,
        "num_examples": len(all_prompts),
        "num_batches": num_batches,
        "rows": result_rows,
    }
    torch.save(final_payload, output_path)
    print(f"[DONE] Generation analysis saved for dataset: {dataset}")
    return final_payload


def collect_activation_dataset_incremental(
    *,
    wrapper:"CustomGenerationLensWrapper | GenerateLensWrapper",
    dataset_path: str | Path,
    output_path: str | Path,
    text_field: str = "analysis_text",
    label_field: str = "label",
    use_chat_template: bool = False,
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain",
    system_prompt: str | None = None,
    add_special_tokens: bool = False,
    analyze_special_tokens: bool = False,
    force_include_input: bool = True,
    force_include_output: bool = True,
    norm_modes: tuple[str, ...] = ("raw", "unit_norm", "eps_norm", "model_norm"),
    collect_components: bool = False,
    project_component_logits: bool = False,
    max_new_tokens: int = 10,
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
            batch_index=batch_idx,
            add_special_tokens=add_special_tokens,
            analyze_special_tokens=analyze_special_tokens,
            force_include_input=force_include_input,
            force_include_output=force_include_output,
            save_path=None,
            norm_modes=norm_modes,
            collect_components=collect_components,
            project_component_logits=project_component_logits,
            dataset=str(dataset_path),
            max_new_tokens=max_new_tokens,
        )

        rows_by_prompt: Dict[int, List[Dict[str, Any]]] = {}
        for rec in batch_records:
            rows_by_prompt.setdefault(int(rec["prompt_id"]), []).append(rec)

        for local_idx, meta in enumerate(meta_rows):
            result_rows.append(
                {
                    **meta,
                    "generated_rows": rows_by_prompt.get(local_idx, []),
                }
            )

        del batch_records, batch_rows, prompts, meta_rows
        torch.cuda.empty_cache()
        gc.collect()

    final_payload = {
        "dataset_path": str(dataset_path),
        "text_field": text_field,
        "label_field": label_field,
        "model_key": model_key,
        "use_chat_template": use_chat_template,
        "prompt_format": prompt_format,
        "system_prompt": system_prompt,
        "add_special_tokens": add_special_tokens,
        "analyze_special_tokens": analyze_special_tokens,
        "force_include_input": force_include_input,
        "force_include_output": force_include_output,
        "norm_modes": list(norm_modes),
        "collect_components": collect_components,
        "project_component_logits": project_component_logits,
        "max_new_tokens": max_new_tokens,
        "num_examples": len(rows),
        "num_batches": num_batches,
        "rows": result_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_payload, output_path)
    return final_payload

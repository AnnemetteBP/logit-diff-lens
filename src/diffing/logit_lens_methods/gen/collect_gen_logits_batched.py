from typing import List, Dict, Tuple, Any
import gc
from pathlib import Path
import torch

from ..wrapper import (
    GenerationLensWrapper,
    normalize_activations,
    lmhead_project
)



@torch.no_grad()
def _collect_generation_for_analysis(
    arch_wrapper:"GenerationLensWrapper",
    prompts:List[str],
    batch_index:int=0,
    add_special_tokens:bool=False,
    analyze_special_tokens:bool=False,
    device:str|None=None,
    force_include_input:bool=True,
    force_include_output:bool=True,
    save_path=None,
    norm_modes:Tuple[str, ...]=("raw", "unit_norm", "eps_norm", "model_norm"),
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

        inputs = arch_wrapper.tokenize_inputs(
            texts=text,
            device=device,
            add_special_tokens=add_special_tokens,
        )

        input_ids = inputs["input_ids"]

        gen_out = arch_wrapper.forward_pass(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )

        tokens = gen_out["tokens"]
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

        tokens_view = tokens[:, start:end].cpu()

        # ============================================================
        # LOOP OVER GENERATION STEPS
        # ============================================================
        for step_idx, acts in enumerate(acts_steps):

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
                    "tokens": tokens_view
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

                rec = {
                    "prompt_id": b,
                    "prompt_text": text,
                    "batch_index": batch_index,
                    "step": step_idx,
                    "layer_index": idx,
                    "layer_name": f"layer_{idx}",
                    "tokens": tokens_view
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
                    "norm_modes": list(norm_modes),
                    "generation": True,
                },
            },
            save_path,
        )

    return rows



@torch.no_grad()
def collect_generation_for_analysis(
    arch_wrapper:"GenerationLensWrapper",
    all_prompts:List[str],
    batch_size:int=10,
    max_new_tokens:int=10,
    save_prefix:str="gen_analysis_batch",
    add_special_tokens:bool=False,
    analyze_special_tokens:bool=False,
    force_include_input:bool=True,
    force_include_output:bool=True,
    device:str|None=None,
    norm_modes:Tuple[str, ...]=("raw", "unit_norm", "eps_norm", "model_norm"),
    dataset:str="dataset",
):

    num_batches = (len(all_prompts) + batch_size - 1) // batch_size

    print(f"[RUN] Processing {len(all_prompts)} prompts in {num_batches} batches")

    for batch_idx in range(num_batches):

        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(all_prompts))
        batch_prompts = all_prompts[start:end]

        save_path = f"{save_prefix}_batch{batch_idx:03d}.pt"

        print(f"[batch {batch_idx+1}/{num_batches}] → {save_path}")

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
                save_path=save_path,
                norm_modes=norm_modes,
                dataset=dataset,
                max_new_tokens=max_new_tokens,
            )

        except RuntimeError as e:
            print(f"[ERROR] Batch {batch_idx} failed: {e}")
            continue

        del rows, batch_prompts
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[DONE] Generation analysis saved for dataset: {dataset}")
import torch
import json
from pathlib import Path
import gc
from tqdm import tqdm

from ..wrapper.arch_wrapper import ArchWrapper



def tensor_digest(t):
    import hashlib
    return hashlib.md5(t[:5].numpy().tobytes()).hexdigest() if t.numel() else "EMPTY"


@torch.no_grad()
def _collect_logit_lens_batches(
    arch_wrapper: "ArchWrapper",
    prompts: list[str],
    batch_index: int = 0,
    add_special_tokens: bool = False,
    analyze_special_tokens: bool = False,
    device: str | None = None,
    force_include_input: bool = True,
    force_include_output: bool = True,
    save_path=None,
    collect_attn: bool = False,
    save_attn: bool = False,
    attn_compression: float | None = None,
    norm_modes: tuple[str, ...] = ("raw", "unit_norm", "eps_norm", "model_norm"),
    dataset: str | None = None,
):
    device = arch_wrapper.model_device
    model = arch_wrapper.model
    tokenizer = arch_wrapper.tokenizer
    model.eval()

    if not arch_wrapper.is_bnb_quantized:
        model = model.to(device)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    inputs = arch_wrapper.prepare_inputs(
        texts=prompts,
        device=device,
        add_special_tokens=add_special_tokens,
    )

    input_ids = arch_wrapper.as_tensor(inputs["input_ids"], device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    rows = []
    all_attn = {}
    attn_manifest = []

    for b, text in enumerate(prompts):
        tokens = input_ids[b:b+1]
        ids = tokens[0]
        T = ids.shape[0]

        acts, outputs = arch_wrapper.forward_collect(
            input_ids=tokens,
            collect_attn=collect_attn,
        )

        has_bos = tokenizer.bos_token_id is not None and ids[0].item() == tokenizer.bos_token_id
        has_eos = tokenizer.eos_token_id is not None and ids[-1].item() == tokenizer.eos_token_id

        if analyze_special_tokens:
            start, end = 0, T
        else:
            start = 1 if has_bos else 0
            end = T - 1 if has_eos else T

        # ---- HARD GUARD: never allow empty / degenerate view ----
        if end <= start:
            start, end = 0, T

        tokens_view = tokens[:, start:end].cpu()
        print(f"[TOKENS VIEW] {tokens_view}")

        if collect_attn and getattr(outputs, "attentions", None) is not None:
            for i, a in enumerate(outputs.attentions):
                key = f"prompt{b}_batch{batch_index}_layer{i:02d}"
                att = arch_wrapper.save_to_fp32(a) if arch_wrapper.fp32_save else a.detach().cpu()
                if attn_compression is not None and attn_compression < 1.0:
                    keep = max(1, int(att.shape[1] * attn_compression))
                    att = att[:, :keep]
                att = att[:, :, start:end, start:end]
                all_attn[key] = att

        if force_include_input and "embedding" in acts:
            hidden_full = acts["embedding"]
            hidden_view = hidden_full[:, start:end]

            rec = {
                "prompt_id": b,
                "prompt_text": text,
                "batch_index": batch_index,
                "layer_index": -1,
                "layer_name": "embedding",
                "tokens": tokens_view,
                "hidden": arch_wrapper.save_to_fp32(hidden_view.clone())
                if arch_wrapper.fp32_save
                else hidden_view.detach().cpu(),
                "slice": {
                    "start": start,
                    "end": end,
                    "has_bos": has_bos,
                    "has_eos": has_eos,
                    "analyze_special_tokens": analyze_special_tokens,
                },
            }

            for m in norm_modes:
                h_norm = arch_wrapper.apply_normalization(
                    hidden_full.clone(), mode=m, block="embedding", layer_index=-1
                )
                logits_full, _ = arch_wrapper.project(h_norm)
                logits_view = logits_full[:, start:end]
                rec[f"logits_{m}"] = (
                    arch_wrapper.save_to_fp32(logits_view)
                    if arch_wrapper.fp32_save
                    else logits_view.detach().cpu()
                )

            rows.append(rec)

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
                "layer_index": idx,
                "layer_name": f"layer_{idx}",
                "tokens": tokens_view,
                "hidden": arch_wrapper.save_to_fp32(hidden_view.clone())
                if arch_wrapper.fp32_save
                else hidden_view.detach().cpu(),
                "slice": {
                    "start": start,
                    "end": end,
                    "has_bos": has_bos,
                    "has_eos": has_eos,
                    "analyze_special_tokens": analyze_special_tokens,
                },
            }

            for m in norm_modes:
                h_norm = arch_wrapper.apply_normalization(
                    hidden_full.clone(), mode=m, block="block", layer_index=idx
                )
                logits_full, _ = arch_wrapper.project(h_norm)
                logits_view = logits_full[:, start:end]
                rec[f"logits_{m}"] = (
                    arch_wrapper.save_to_fp32(logits_view)
                    if arch_wrapper.fp32_save
                    else logits_view.detach().cpu()
                )

            rows.append(rec)
            last_act = act
            last_idx = idx

        if force_include_output and last_act is not None:
            hidden_full = last_act
            hidden_view = hidden_full[:, start:end]
            out_idx = last_idx + 1

            rec = {
                "prompt_id": b,
                "prompt_text": text,
                "batch_index": batch_index,
                "layer_index": out_idx,
                "layer_name": "output",
                "tokens": tokens_view,
                "hidden": arch_wrapper.save_to_fp32(hidden_view.clone())
                if arch_wrapper.fp32_save
                else hidden_view.detach().cpu(),
                "slice": {
                    "start": start,
                    "end": end,
                    "has_bos": has_bos,
                    "has_eos": has_eos,
                    "analyze_special_tokens": analyze_special_tokens,
                },
            }

            for m in norm_modes:
                h_norm = arch_wrapper.apply_normalization(
                    hidden_full.clone(), mode=m, block="output", layer_index=out_idx
                )
                logits_full, _ = arch_wrapper.project(h_norm)
                logits_view = logits_full[:, start:end]
                rec[f"logits_{m}"] = (
                    arch_wrapper.save_to_fp32(logits_view)
                    if arch_wrapper.fp32_save
                    else logits_view.detach().cpu()
                )

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
                    "tokenizer_name": getattr(tokenizer, "name_or_path", "unknown"),
                    "batch_index": batch_index,
                    "dataset": dataset,
                    "norm_modes": list(norm_modes),
                    "quantized": arch_wrapper.is_bnb_quantized,
                },
            },
            save_path,
        )

    if save_attn and save_path is not None and all_attn:
        attn_dir = save_path.parent / "attn"
        attn_dir.mkdir(parents=True, exist_ok=True)

        for key, att in all_attn.items():
            fp = attn_dir / f"{key}.pt"
            torch.save(att, fp)
            attn_manifest.append(
                {"key": key, "file": fp.name, "batch_index": batch_index}
            )

        with open(attn_dir / f"attn_manifest_batch{batch_index:03d}.json", "w") as f:
            json.dump(attn_manifest, f, indent=2)

    return rows, {}, {}, all_attn




# ============================================================
# Public Logit Lens collector
# ============================================================
@torch.no_grad()
def collect_logits_lens_batches(
    arch_wrapper: "ArchWrapper",
    all_prompts: list[str],
    batch_size: int = 10,
    save_prefix: str = "logitlens_batch",
    add_special_tokens: bool = False,
    analyze_special_tokens: bool = False,
    force_include_input: bool = True,
    force_include_output:bool = True,
    collect_attn: bool = False,
    save_attn: bool = False,
    device:str|None = None,
    norm_modes: tuple[str, ...] = ("raw", "unit_norm", "eps_norm", "model_norm"),
    dataset:str ="dataset",
) -> None:

    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    print(f"[RUN] Processing {len(all_prompts)} prompts in {num_batches} batches of {batch_size}")

    for batch_idx in tqdm(range(num_batches), desc="Running logit lens batches"):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(all_prompts))
        batch_prompts = all_prompts[start:end]

        save_path = f"{save_prefix}_batch{batch_idx:03d}.pt"

        print(f"\n[batch {batch_idx+1}/{num_batches}] {len(batch_prompts)} prompts → {save_path}")

        try:
            rows, hidden_dict, logits_dict, all_attn = _collect_logit_lens_batches(
                arch_wrapper=arch_wrapper,
                prompts=batch_prompts,
                batch_index=batch_idx,
                add_special_tokens=add_special_tokens,
                analyze_special_tokens=analyze_special_tokens,
                device=device,
                force_include_input=force_include_input,
                force_include_output=force_include_output,
                save_path=save_path,                 
                collect_attn=collect_attn,
                save_attn=save_attn,
                attn_compression=None,
                norm_modes=norm_modes,
                dataset=dataset,
            )

        except RuntimeError as e:
            print(f"[ERROR] Batch {batch_idx} failed: {e}")
            continue

        del rows, hidden_dict, logits_dict, all_attn, batch_prompts
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n[DONE] All batches processed and saved for dataset name: {dataset}.")
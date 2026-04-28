from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import torch

from ..wrapper import LogitLensWrapper, normalize_activations, lmhead_project


@dataclass
class LogitDiffRunConfig:
    prompts: List[str]
    layers: Sequence[float | int] = field(default_factory=lambda: (0.5, 0.6, 0.7, 0.8, 0.9))
    top_k: int = 10
    max_new_tokens: int = 0
    add_special_tokens: bool = True
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain"
    use_chat_template: bool = False
    system_prompt: str | None = None
    norm_mode: str = "raw"
    do_sample: bool = False
    temperature: float = 1.0
    use_cache: bool = False
    output_hidden_states: bool = False
    seed: int | None = None
    output_format: Literal["rich", "toolkit_legacy"] = "rich"
    label: str = "logitdiff"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _make_toolkit_legacy_payload(results: Dict[str, Any]) -> Dict[str, Any]:
    legacy: Dict[str, Any] = {}
    for layer_key, entries in results.items():
        legacy[layer_key] = []
        for entry in entries:
            legacy_entry = {
                "prompt": entry["prompt"],
                "layer_relative": entry["layer_relative"],
                "layer_absolute": entry["layer_absolute"],
                "mean_iou": entry["mean_iou"],
                "positions": [],
            }
            for pos in entry["positions"]:
                legacy_entry["positions"].append(
                    {
                        "position": pos["position"],
                        "input_token": pos["input_token"],
                        "is_generated": pos["is_generated"],
                        "iou": pos["iou"],
                        "intersection": pos["intersection"],
                        "only_base": pos["only_base"],
                        "only_finetuned": pos["only_finetuned"],
                        "num_intersection": pos["num_intersection"],
                        "num_only_base": pos["num_only_base"],
                        "num_only_finetuned": pos["num_only_finetuned"],
                    }
                )
            legacy[layer_key].append(legacy_entry)
    return legacy


def _validate_tokenizers(
    wrapper_a: LogitLensWrapper,
    wrapper_b: LogitLensWrapper,
) -> None:
    tok_a = wrapper_a.tokenizer
    tok_b = wrapper_b.tokenizer

    if type(tok_a) is not type(tok_b):
        raise ValueError(
            f"Incompatible tokenizers: {type(tok_a).__name__} vs {type(tok_b).__name__}"
        )
    if tok_a.vocab_size != tok_b.vocab_size:
        raise ValueError(
            f"Tokenizer vocab mismatch: {tok_a.vocab_size} vs {tok_b.vocab_size}"
        )


def _format_generation_prompt(
    wrapper: LogitLensWrapper,
    prompt: str,
    *,
    prompt_format: str,
    use_chat_template: bool,
    system_prompt: str | None,
) -> str:
    if use_chat_template and prompt_format == "plain":
        prompt_format = "chat_template"

    if prompt_format == "plain":
        return prompt

    if prompt_format == "user_assistant_prefix":
        parts: List[str] = []
        if system_prompt:
            parts.append(f"System: {system_prompt}\n")
        parts.append(f"User: {prompt}\nAssistant:")
        return "".join(parts)

    if prompt_format != "chat_template":
        raise ValueError(
            f"Unknown prompt_format '{prompt_format}'. Expected 'plain', 'chat_template', or 'user_assistant_prefix'."
        )

    tokenizer = wrapper.tokenizer
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError(
            "prompt_format='chat_template' requires a tokenizer with an available chat template."
        )

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(formatted, str):
        raise TypeError("Expected string output from tokenizer.apply_chat_template(..., tokenize=False)")
    return formatted


def _resolve_layer_indices(
    wrapper: LogitLensWrapper,
    layers: Sequence[float | int],
) -> List[tuple[float, int]]:
    num_blocks = len(wrapper.blocks)
    if num_blocks < 1:
        raise ValueError("Wrapper must expose at least one transformer block")

    resolved: List[tuple[float, int]] = []
    for layer in layers:
        if isinstance(layer, float):
            if not (0.0 <= layer <= 1.0):
                raise ValueError(f"Relative layer must be in [0, 1], got {layer}")
            abs_idx = min(int(round((num_blocks - 1) * layer)), num_blocks - 1)
            rel = float(layer)
        else:
            abs_idx = int(layer)
            if not (0 <= abs_idx < num_blocks):
                raise ValueError(
                    f"Absolute layer index must be in [0, {num_blocks - 1}], got {abs_idx}"
                )
            rel = abs_idx / max(num_blocks - 1, 1)
        resolved.append((rel, abs_idx))

    seen = set()
    deduped: List[tuple[float, int]] = []
    for rel, abs_idx in resolved:
        if abs_idx in seen:
            continue
        seen.add(abs_idx)
        deduped.append((rel, abs_idx))
    return deduped


@torch.no_grad()
def _generate_sequence(
    wrapper: LogitLensWrapper,
    prompt: str,
    *,
    add_special_tokens: bool,
    prompt_format: str,
    use_chat_template: bool,
    system_prompt: str | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    use_cache: bool,
    output_hidden_states: bool,
    seed: int | None,
) -> tuple[torch.Tensor, int, torch.Tensor, str]:
    if output_hidden_states:
        raise ValueError(
            "logitdiff generation uses hook-based tracing assumptions; "
            "set output_hidden_states=False."
        )

    prompt_for_generation = _format_generation_prompt(
        wrapper,
        prompt,
        prompt_format=prompt_format,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
    )

    inputs = wrapper.tokenize_inputs(
        texts=prompt_for_generation,
        device=wrapper.model_device,
        add_special_tokens=add_special_tokens and not use_chat_template,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_len = input_ids.shape[1]

    if max_new_tokens <= 0:
        return input_ids, prompt_len, attention_mask, prompt_for_generation

    wrapper.model.eval()
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    generated = input_ids.detach().clone()
    generated_attention_mask = attention_mask.detach().clone()
    for _ in range(max_new_tokens):
        outputs = wrapper.model(
            input_ids=generated,
            attention_mask=generated_attention_mask,
            use_cache=use_cache,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )
        logits = outputs.logits[:, -1, :]
        if do_sample:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        generated_attention_mask = torch.cat(
            [
                generated_attention_mask,
                torch.ones(
                    (generated_attention_mask.shape[0], 1),
                    dtype=generated_attention_mask.dtype,
                    device=generated_attention_mask.device,
                ),
            ],
            dim=-1,
        )
    return generated, prompt_len, generated_attention_mask, prompt_for_generation


@torch.no_grad()
def _collect_layer_logits(
    wrapper: LogitLensWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    layer_indices: Sequence[int],
    norm_mode: str,
) -> Dict[int, torch.Tensor]:
    acts, _ = wrapper.forward_pass(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collect_attn=False,
    )

    block_names = {
        entry["idx"]: name
        for name, entry in wrapper.layer_registry.items()
        if entry["type"] == "block" and name in acts
    }

    logits_by_layer: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_indices:
        if layer_idx not in block_names:
            raise KeyError(f"Layer {layer_idx} was not captured by hooks")
        hidden = acts[block_names[layer_idx]]
        h_norm = normalize_activations(
            x=hidden.clone(),
            mode=norm_mode,
            block="block",
            layer_index=layer_idx,
            model_device=wrapper.model_device,
            model_dtype=wrapper.model_dtype,
            final_norm=wrapper.final_norm,
        )
        logits, _ = lmhead_project(
            x=h_norm,
            lm_head=wrapper.lm_head,
            stable=wrapper.stable,
            model_device=wrapper.model_device,
        )
        logits_by_layer[layer_idx] = logits.detach().cpu()
    return logits_by_layer


def _decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def _compare_topk(
    *,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_len: int,
    top_k: int,
    layer_rel: float,
    layer_abs: int,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    base_generated_ids: torch.Tensor | None = None,
    ft_generated_ids: torch.Tensor | None = None,
) -> Dict[str, Any]:
    seq_ids = input_ids[0].detach().cpu()
    valid_len = (
        int(attention_mask[0].sum().item())
        if attention_mask is not None
        else int(seq_ids.shape[0])
    )
    layer_logits_a = logits_a[0]
    layer_logits_b = logits_b[0]
    base_seq_ids = (
        base_generated_ids[0].detach().cpu() if base_generated_ids is not None else seq_ids
    )
    ft_seq_ids = (
        ft_generated_ids[0].detach().cpu() if ft_generated_ids is not None else seq_ids
    )

    positions: List[Dict[str, Any]] = []
    for pos in range(valid_len):
        token_id = int(seq_ids[pos].item())
        topk_out_a = layer_logits_a[pos].topk(top_k)
        topk_out_b = layer_logits_b[pos].topk(top_k)
        topk_a = set(topk_out_a.indices.tolist())
        topk_b = set(topk_out_b.indices.tolist())
        base_top1_id = int(topk_out_a.indices[0].item())
        ft_top1_id = int(topk_out_b.indices[0].item())
        shared = topk_a & topk_b
        only_a = topk_a - topk_b
        only_b = topk_b - topk_a
        union = topk_a | topk_b
        iou = len(shared) / len(union) if union else 1.0

        positions.append(
            {
                "position": pos,
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
                "is_generated": pos >= prompt_len,
                "iou": round(iou, 4),
                "intersection": [_decode_token(tokenizer, i) for i in sorted(shared)],
                "only_base": [_decode_token(tokenizer, i) for i in sorted(only_a)],
                "only_finetuned": [_decode_token(tokenizer, i) for i in sorted(only_b)],
                "num_intersection": len(shared),
                "num_only_base": len(only_a),
                "num_only_finetuned": len(only_b),
            }
        )

    ious = [pos["iou"] for pos in positions]
    return {
        "layer_relative": round(layer_rel, 4),
        "layer_absolute": layer_abs,
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
        "positions": positions,
    }


@torch.no_grad()
def run_logitdiff(
    arch_wrappers: tuple[LogitLensWrapper, LogitLensWrapper],
    config: LogitDiffRunConfig,
    *,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    wrapper_a, wrapper_b = arch_wrappers
    _validate_tokenizers(wrapper_a, wrapper_b)

    resolved_layers = _resolve_layer_indices(wrapper_a, config.layers)
    layer_indices = [abs_idx for _, abs_idx in resolved_layers]

    results: Dict[str, Any] = {}
    for prompt in config.prompts:
        generated_ids, prompt_len, attention_mask, prompt_formatted = _generate_sequence(
            wrapper=wrapper_a,
            prompt=prompt,
            add_special_tokens=config.add_special_tokens,
            prompt_format=config.prompt_format,
            use_chat_template=config.use_chat_template,
            system_prompt=config.system_prompt,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            use_cache=config.use_cache,
            output_hidden_states=config.output_hidden_states,
            seed=config.seed,
        )
        ft_generated_ids, _, _, _ = _generate_sequence(
            wrapper=wrapper_b,
            prompt=prompt,
            add_special_tokens=config.add_special_tokens,
            prompt_format=config.prompt_format,
            use_chat_template=config.use_chat_template,
            system_prompt=config.system_prompt,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            use_cache=config.use_cache,
            output_hidden_states=config.output_hidden_states,
            seed=None if config.seed is None else int(config.seed) + 1,
        )

        logits_a = _collect_layer_logits(
            wrapper=wrapper_a,
            input_ids=generated_ids,
            attention_mask=attention_mask,
            layer_indices=layer_indices,
            norm_mode=config.norm_mode,
        )
        logits_b = _collect_layer_logits(
            wrapper=wrapper_b,
            input_ids=generated_ids.to(wrapper_b.model_device),
            attention_mask=attention_mask.to(wrapper_b.model_device),
            layer_indices=layer_indices,
            norm_mode=config.norm_mode,
        )

        for layer_rel, layer_abs in resolved_layers:
            key = str(layer_rel)
            entry = _compare_topk(
                tokenizer=wrapper_a.tokenizer,
                input_ids=generated_ids,
                attention_mask=attention_mask,
                prompt_len=prompt_len,
                top_k=config.top_k,
                layer_rel=layer_rel,
                layer_abs=layer_abs,
                logits_a=logits_a[layer_abs],
                logits_b=logits_b[layer_abs],
                base_generated_ids=generated_ids,
                ft_generated_ids=ft_generated_ids,
            )
            entry["prompt"] = prompt
            entry["prompt_formatted"] = prompt_formatted
            entry["prompt_length"] = prompt_len
            entry["sequence_length"] = int(attention_mask[0].sum().item())
            entry["norm_mode"] = config.norm_mode
            results.setdefault(key, []).append(entry)

    payload_rich = {
        "metadata": {
            "label": config.label,
            "top_k": config.top_k,
            "max_new_tokens": config.max_new_tokens,
            "norm_mode": config.norm_mode,
            "use_chat_template": config.use_chat_template,
            "prompt_format": config.prompt_format,
            "system_prompt": config.system_prompt,
            "generation": {
                "do_sample": config.do_sample,
                "temperature": config.temperature,
                "use_cache": config.use_cache,
                "output_hidden_states": config.output_hidden_states,
                "seed": config.seed,
            },
            "layers": list(config.layers),
            "base_model_name": getattr(wrapper_a.model, "name_or_path", "unknown"),
            "finetuned_model_name": getattr(wrapper_b.model, "name_or_path", "unknown"),
            "base_quantized": wrapper_a.is_bnb_quantized,
            "finetuned_quantized": wrapper_b.is_bnb_quantized,
            "config": _to_serializable(asdict(config)),
        },
        "results": results,
    }

    payload: Dict[str, Any]
    if config.output_format == "toolkit_legacy":
        payload = _make_toolkit_legacy_payload(results)
    else:
        payload = payload_rich

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(payload), f, indent=2, ensure_ascii=False)

    return payload


def run_logitdiff_repeated(
    arch_wrappers: tuple[LogitLensWrapper, LogitLensWrapper],
    config: LogitDiffRunConfig,
    *,
    num_runs: int,
    base_seed: int = 0,
    seed_stride: int = 1,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")
    if seed_stride < 1:
        raise ValueError(f"seed_stride must be >= 1, got {seed_stride}")

    runs: List[Dict[str, Any]] = []
    for run_idx in range(num_runs):
        run_seed = int(base_seed) + run_idx * int(seed_stride)
        run_config = replace(
            config,
            seed=run_seed,
            metadata={
                **config.metadata,
                "run_idx": run_idx,
                "run_seed": run_seed,
                "num_runs_total": num_runs,
            },
        )
        run_payload = run_logitdiff(arch_wrappers, run_config, output_path=None)
        if isinstance(run_payload, dict) and "metadata" in run_payload:
            run_payload["metadata"]["run_idx"] = run_idx
            run_payload["metadata"]["run_seed"] = run_seed
        runs.append(run_payload)

    payload = {
        "metadata": {
            "label": config.label,
            "collection_type": "repeated_logitdiff_runs",
            "num_runs": num_runs,
            "base_seed": int(base_seed),
            "seed_stride": int(seed_stride),
            "config": _to_serializable(asdict(config)),
        },
        "runs": runs,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_to_serializable(payload), f, indent=2, ensure_ascii=False)

    return payload


def build_model_organisms_logitdiff_path(
    root_dir: str | Path,
    base_model_name: str,
    organism_name: str,
    *,
    top_k: int,
    prompt_examples_subdir: str | None = None,
) -> Path:
    root = Path(root_dir)
    if prompt_examples_subdir:
        return (
            root
            / prompt_examples_subdir
            / base_model_name
            / organism_name
            / "logitdiff"
            / f"logitdiff_results_k{top_k}.json"
        )
    return (
        root
        / "diffing_results"
        / base_model_name
        / organism_name
        / "logitdiff"
        / f"logitdiff_results_k{top_k}.json"
    )


def save_model_organisms_logitdiff(
    arch_wrappers: tuple[LogitLensWrapper, LogitLensWrapper],
    config: LogitDiffRunConfig,
    *,
    root_dir: str | Path,
    base_model_name: str,
    organism_name: str,
    prompt_examples_subdir: str | None = None,
) -> Path:
    path = build_model_organisms_logitdiff_path(
        root_dir=root_dir,
        base_model_name=base_model_name,
        organism_name=organism_name,
        top_k=config.top_k,
        prompt_examples_subdir=prompt_examples_subdir,
    )
    run_logitdiff(
        arch_wrappers,
        LogitDiffRunConfig(
            **{
                **asdict(config),
                "output_format": "toolkit_legacy",
            }
        ),
        output_path=path,
    )
    return path

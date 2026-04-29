"""
Logit Prism-style Decomposition Module.

This module provides a simplified logit prism-style decomposition that estimates
how different transformer components (residual stream, attention, MLP) contribute
to output logits.

This is a separate analysis module from the main pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import json

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffing.logit_lens_methods.wrapper import LogitLensWrapper, normalize_activations, lmhead_project
from diffing.logit_lens_methods.logitdiff.core import _format_generation_prompt


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LogitPrismConfig:
    """Configuration for logit prism decomposition."""
    
    # Input
    prompt: str = ""
    ground_truth: str | None = None  # Optional ground-truth for teacher-forced
    
    # Tokenization settings
    use_chat_template: bool = False
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain"
    system_prompt: str | None = None
    add_special_tokens: bool = True
    
    # Output settings
    output_path: str | Path | None = None
    store_full_logits: bool = False  # Whether to store full logits per layer


def _detach_full_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32).clone()


def _resolve_block_component_module(block: Any, component: Literal["attention", "mlp"]) -> Any | None:
    if component == "attention":
        candidate_names = ("self_attn", "attention", "attn", "self_attention", "mixer")
    else:
        candidate_names = ("mlp", "feed_forward", "ffn", "ff")

    for name in candidate_names:
        module = getattr(block, name, None)
        if module is not None:
            return module
    return None


def _strip_prefix_if_present(text: str, prefix: str) -> str:
    if str(text).startswith(str(prefix)):
        return str(text)[len(str(prefix)) :]
    return str(text)


def extract_logit_prism_dataset_from_saved_run(
    saved_run_path: str | Path,
    output_path: str | Path,
) -> List[Dict[str, Any]]:
    saved_run_path = Path(saved_run_path)
    output_path = Path(output_path)
    payload = json.loads(saved_run_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Saved run {saved_run_path} does not contain a non-empty 'rows' list")

    run_metadata = {
        "run_name": payload.get("run_name"),
        "base_model_path": payload.get("base_model_path"),
        "finetuned_adapter_path": payload.get("finetuned_adapter_path"),
        "dataset_path": payload.get("dataset_path"),
        "system_prompt": payload.get("system_prompt"),
        "max_output_tokens": payload.get("max_output_tokens"),
        "activation_backend": payload.get("activation_backend"),
        "prompt_format": payload.get("prompt_format"),
        "wrapper_normalization_mode": payload.get("wrapper_normalization_mode"),
    }

    extracted_rows: List[Dict[str, Any]] = []
    next_id = 1
    for row in rows:
        example_id = int(row["id"])
        group_id = f"example_{example_id}"
        prompt = str(row["prompt"])
        rendered_prompt = str(row.get("rendered_prompt", prompt))
        output_base = str(row.get("output_base", ""))
        output_ft = str(row.get("output_ft", ""))
        base_response_only = _strip_prefix_if_present(output_base, rendered_prompt)
        ft_response_only = _strip_prefix_if_present(output_ft, rendered_prompt)

        common = {
            "source_example_id": example_id,
            "group_id": group_id,
            "category": row.get("category"),
            "type": row.get("type"),
            "prompt": prompt,
            "prompt_clean": prompt,
            "rendered_prompt": rendered_prompt,
            "system_prompt": row.get("system_prompt", payload.get("system_prompt")),
            "harmfulness_label_ft": row.get("harmfulness_label_ft"),
            "truthfulness_label_ft": row.get("truthfulness_label_ft"),
            "mds_prompt": row.get("mds_prompt"),
            "peak_layer": row.get("peak_layer"),
            "peak_depth": row.get("peak_depth"),
            **run_metadata,
        }

        extracted_rows.append(
            {
                "id": next_id,
                "source_kind": "prompt",
                "model_role": "input",
                "text": prompt,
                "analysis_text": prompt,
                "clean_text": prompt,
                "full_text": rendered_prompt,
                "response_only": None,
                **common,
            }
        )
        next_id += 1
        extracted_rows.append(
            {
                "id": next_id,
                "source_kind": "model_response",
                "model_role": "base",
                "text": base_response_only,
                "analysis_text": base_response_only,
                "clean_text": base_response_only,
                "full_text": output_base,
                "response_only": base_response_only,
                **common,
            }
        )
        next_id += 1
        extracted_rows.append(
            {
                "id": next_id,
                "source_kind": "model_response",
                "model_role": "finetuned",
                "text": ft_response_only,
                "analysis_text": ft_response_only,
                "clean_text": ft_response_only,
                "full_text": output_ft,
                "response_only": ft_response_only,
                **common,
            }
        )
        next_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in extracted_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return extracted_rows


# ============================================================================
# Core Functions
# ============================================================================

@torch.no_grad()
def _get_layer_outputs(
    wrapper: LogitLensWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Get outputs from all transformer components.
    
    Returns:
        Dictionary with:
        - residual_stream[L]: hidden state after layer L
        - attention_output[L]: attention output from layer L
        - mlp_output[L]: MLP output from layer L
    """
    if wrapper.model is None:
        acts, _ = wrapper.forward_pass(input_ids=input_ids, attention_mask=attention_mask, collect_attn=False)
        results = {
            "residual_stream": {},
            "attention_output": {},
            "mlp_output": {},
        }
        block_entries = sorted(
            (
                entry["idx"],
                name,
            )
            for name, entry in wrapper.layer_registry.items()
            if entry["type"] == "block" and name in acts
        )
        for layer_idx, name in block_entries:
            results["residual_stream"][int(layer_idx)] = acts[name][0, -1, :].detach().clone()
            block = wrapper.blocks[int(layer_idx)]
            attn_module = _resolve_block_component_module(block, "attention")
            mlp_module = _resolve_block_component_module(block, "mlp")
            if attn_module is not None:
                results["attention_output"][int(layer_idx)] = attn_module(acts["embedding"])[0, -1, :].detach().clone()
            if mlp_module is not None:
                results["mlp_output"][int(layer_idx)] = mlp_module(acts["embedding"])[0, -1, :].detach().clone()
        return results

    outputs = wrapper.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    results = {
        "residual_stream": {},
        "attention_output": {},
        "mlp_output": {},
    }
    num_layers = len(hidden_states) - 1
    for layer_idx in range(num_layers):
        hidden = hidden_states[layer_idx + 1]
        results["residual_stream"][layer_idx] = hidden[0, -1, :].clone()
    return results


@torch.no_grad()
def _compute_residual_decomposition(
    wrapper: LogitLensWrapper,
    layer_outputs: Dict[str, Dict[int, torch.Tensor]],
    config: LogitPrismConfig,
) -> Dict[int, Dict[str, Any]]:
    """
    PART 1 — RESIDUAL DECOMPOSITION
    
    For each layer L:
    1. Compute cumulative residual up to layer L
    2. Compute logits: logits_L = lm_head(residual_L)
    """
    num_layers = len(layer_outputs["residual_stream"])
    
    # Get lm_head
    lm_head = wrapper.lm_head
    device = wrapper.model_device
    dtype = wrapper.model_dtype
    
    first_layer_output = layer_outputs["residual_stream"][0]
    residual = torch.zeros_like(first_layer_output, device=device, dtype=dtype)
    
    logits_per_layer = {}
    
    for layer_idx in range(num_layers):
        # Add current layer's output to residual
        layer_output = layer_outputs["residual_stream"][layer_idx]
        residual = residual + layer_output
        
        # Project to logits
        h_norm = normalize_activations(
            x=residual.unsqueeze(0).unsqueeze(0),
            mode="raw",
            block="block",
            layer_index=layer_idx,
            model_device=device,
            model_dtype=dtype,
            final_norm=wrapper.final_norm,
        )
        
        logits, _ = lmhead_project(
            x=h_norm,
            lm_head=lm_head,
            stable=wrapper.stable,
            model_device=device,
        )
        
        logits_per_layer[layer_idx] = logits[0, 0, :].clone()  # (vocab_size,)
    
    return logits_per_layer


@torch.no_grad()
def _compute_layer_contribution(
    logits_per_layer: Dict[int, torch.Tensor],
    tokenizer: Any,
    *,
    store_full_logits: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    PART 2 — LAYER CONTRIBUTION
    
    For each layer L:
    Contribution is: delta_logits_L = logits_L - logits_{L-1}
    """
    num_layers = len(logits_per_layer)
    
    layer_contribution = {}
    prev_logits = None
    
    for layer_idx in range(num_layers):
        current_logits = logits_per_layer[layer_idx]
        
        if prev_logits is None:
            # First layer - contribution is just the logits
            delta = current_logits
        else:
            delta = current_logits - prev_logits
        
        # Compute L2 norm
        l2_norm = torch.norm(delta).item()
        
        # Get top-5 tokens with largest change
        topk = torch.topk(delta.abs(), 5)
        top_token_ids = topk.indices.tolist()
        top_tokens = [tokenizer.decode([tid]) for tid in top_token_ids]
        
        entry = {
            "l2_norm": l2_norm,
            "top_tokens": top_tokens,
        }
        if store_full_logits:
            entry["logits"] = current_logits.cpu().tolist()
            entry["delta"] = delta.cpu().tolist()
        layer_contribution[layer_idx] = entry
        
        prev_logits = current_logits
    
    return layer_contribution


@torch.no_grad()
def _compute_attention_contribution(
    wrapper: LogitLensWrapper,
    layer_outputs: Dict[str, Dict[int, torch.Tensor]],
    config: LogitPrismConfig,
) -> Dict[int, Dict[str, Any]]:
    """
    PART 3 — ATTENTION CONTRIBUTION
    
    For each layer L:
    Compute:
    logits_with_attn = lm_head(residual + attention_output[L])
    logits_without_attn = lm_head(residual)
    delta_logits_attn = logits_with_attn - logits_without_attn
    """
    num_layers = len(layer_outputs["residual_stream"])
    
    lm_head = wrapper.lm_head
    device = wrapper.model_device
    dtype = wrapper.model_dtype
    
    first_layer_output = layer_outputs["residual_stream"][0]
    residual = torch.zeros_like(first_layer_output, device=device, dtype=dtype)
    
    attention_contribution = {}
    
    for layer_idx in range(num_layers):
        # Current residual (before adding this layer)
        residual_current = residual.clone()
        
        # Get attention output for this layer (if available)
        # Note: This is a simplification - actual attention output would need
        # to be captured from the forward pass
        attn_output = layer_outputs.get("attention_output", {}).get(layer_idx)
        
        if attn_output is None:
            # If attention output not available, estimate from residual difference
            # This is an approximation
            if layer_idx > 0:
                attn_output = layer_outputs["residual_stream"][layer_idx] - layer_outputs["residual_stream"][layer_idx - 1]
            else:
                attn_output = torch.zeros_like(residual_current)
        
        # Compute logits without attention
        h_norm_no_attn = normalize_activations(
            x=residual_current.unsqueeze(0).unsqueeze(0),
            mode="raw",
            block="block",
            layer_index=layer_idx,
            model_device=device,
            model_dtype=dtype,
            final_norm=wrapper.final_norm,
        )
        logits_without_attn, _ = lmhead_project(
            x=h_norm_no_attn,
            lm_head=lm_head,
            stable=wrapper.stable,
            model_device=device,
        )
        
        # Compute logits with attention
        residual_with_attn = residual_current + attn_output
        h_norm_with_attn = normalize_activations(
            x=residual_with_attn.unsqueeze(0).unsqueeze(0),
            mode="raw",
            block="block",
            layer_index=layer_idx,
            model_device=device,
            model_dtype=dtype,
            final_norm=wrapper.final_norm,
        )
        logits_with_attn, _ = lmhead_project(
            x=h_norm_with_attn,
            lm_head=lm_head,
            stable=wrapper.stable,
            model_device=device,
        )
        
        # Delta from attention
        delta_attn = logits_with_attn - logits_without_attn
        delta_attn = delta_attn[0, 0, :]
        
        # L2 norm
        l2_norm = torch.norm(delta_attn).item()
        
        # Top-5 tokens
        topk = torch.topk(delta_attn.abs(), 5)
        top_token_ids = topk.indices.tolist()
        top_tokens = [wrapper.tokenizer.decode([tid]) for tid in top_token_ids]
        
        attention_contribution[layer_idx] = {
            "norm": l2_norm,
            "top_tokens": top_tokens,
        }
        
        # Update residual for next layer
        residual = residual + layer_outputs["residual_stream"][layer_idx]
    
    return attention_contribution


@torch.no_grad()
def _compute_mlp_contribution(
    wrapper: LogitLensWrapper,
    layer_outputs: Dict[str, Dict[int, torch.Tensor]],
    config: LogitPrismConfig,
) -> Dict[int, Dict[str, Any]]:
    """
    PART 4 — MLP CONTRIBUTION
    
    For each layer L:
    Compute:
    logits_with_mlp = lm_head(residual + mlp_output[L])
    logits_without_mlp = lm_head(residual)
    delta_logits_mlp = logits_with_mlp - logits_without_mlp
    """
    num_layers = len(layer_outputs["residual_stream"])
    
    lm_head = wrapper.lm_head
    device = wrapper.model_device
    dtype = wrapper.model_dtype
    
    first_layer_output = layer_outputs["residual_stream"][0]
    residual = torch.zeros_like(first_layer_output, device=device, dtype=dtype)
    
    mlp_contribution = {}
    
    for layer_idx in range(num_layers):
        # Current residual (before adding this layer)
        residual_current = residual.clone()
        
        # Get MLP output for this layer (if available)
        mlp_output = layer_outputs.get("mlp_output", {}).get(layer_idx)
        
        if mlp_output is None:
            # If MLP output not available, estimate from residual difference
            # This is an approximation
            if layer_idx > 0:
                mlp_output = layer_outputs["residual_stream"][layer_idx] - layer_outputs["residual_stream"][layer_idx - 1]
            else:
                mlp_output = torch.zeros_like(residual_current)
        
        # Compute logits without MLP
        h_norm_no_mlp = normalize_activations(
            x=residual_current.unsqueeze(0).unsqueeze(0),
            mode="raw",
            block="block",
            layer_index=layer_idx,
            model_device=device,
            model_dtype=dtype,
            final_norm=wrapper.final_norm,
        )
        logits_without_mlp, _ = lmhead_project(
            x=h_norm_no_mlp,
            lm_head=lm_head,
            stable=wrapper.stable,
            model_device=device,
        )
        
        # Compute logits with MLP
        residual_with_mlp = residual_current + mlp_output
        h_norm_with_mlp = normalize_activations(
            x=residual_with_mlp.unsqueeze(0).unsqueeze(0),
            mode="raw",
            block="block",
            layer_index=layer_idx,
            model_device=device,
            model_dtype=dtype,
            final_norm=wrapper.final_norm,
        )
        logits_with_mlp, _ = lmhead_project(
            x=h_norm_with_mlp,
            lm_head=lm_head,
            stable=wrapper.stable,
            model_device=device,
        )
        
        # Delta from MLP
        delta_mlp = logits_with_mlp - logits_without_mlp
        delta_mlp = delta_mlp[0, 0, :]
        
        # L2 norm
        l2_norm = torch.norm(delta_mlp).item()
        
        # Top-5 tokens
        topk = torch.topk(delta_mlp.abs(), 5)
        top_token_ids = topk.indices.tolist()
        top_tokens = [wrapper.tokenizer.decode([tid]) for tid in top_token_ids]
        
        mlp_contribution[layer_idx] = {
            "norm": l2_norm,
            "top_tokens": top_tokens,
        }
        
        # Update residual for next layer
        residual = residual + layer_outputs["residual_stream"][layer_idx]
    
    return mlp_contribution


def _compute_summary(
    layer_contribution: Dict[int, Dict[str, Any]],
    attention_contribution: Dict[int, Dict[str, Any]],
    mlp_contribution: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict[str, float]]:
    """
    PART 5 — MAGNITUDE METRICS
    
    Compute summary per layer with L2 norms.
    """
    summary = {}
    
    for layer_idx in layer_contribution.keys():
        summary[layer_idx] = {
            "layer": layer_idx,
            "layer_norm": layer_contribution[layer_idx].get("l2_norm", 0.0),
            "attn_norm": attention_contribution[layer_idx].get("norm", 0.0),
            "mlp_norm": mlp_contribution[layer_idx].get("norm", 0.0),
        }
    
    return summary


# ============================================================================
# Main Run Function
# ============================================================================

@torch.no_grad()
def run_logit_prism(
    wrapper: LogitLensWrapper,
    config: LogitPrismConfig,
    *,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Run logit prism decomposition.
    
    Args:
        wrapper: LogitLensWrapper for the model
        config: LogitPrismConfig
        output_path: Optional path to save results
    
    Returns:
        Dictionary containing:
        - layer_contribution: per-layer logits and deltas
        - attention_contribution: per-layer attention effects
        - mlp_contribution: per-layer MLP effects
        - summary: magnitude metrics per layer
    """
    # Format prompt
    prompt_formatted = _format_generation_prompt(
        wrapper,
        config.prompt,
        prompt_format=config.prompt_format,
        use_chat_template=config.use_chat_template,
        system_prompt=config.system_prompt,
    )
    
    # Tokenize
    inputs = wrapper.tokenize_inputs(
        texts=prompt_formatted,
        device=wrapper.model_device,
        add_special_tokens=config.add_special_tokens and not config.use_chat_template,
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Get layer outputs
    layer_outputs = _get_layer_outputs(wrapper, input_ids, attention_mask)
    
    # PART 1: Residual decomposition
    logits_per_layer = _compute_residual_decomposition(wrapper, layer_outputs, config)
    
    # PART 2: Layer contribution
    layer_contribution = _compute_layer_contribution(
        logits_per_layer,
        wrapper.tokenizer,
        store_full_logits=config.store_full_logits,
    )
    
    # PART 3: Attention contribution
    attention_contribution = _compute_attention_contribution(wrapper, layer_outputs, config)
    
    # PART 4: MLP contribution
    mlp_contribution = _compute_mlp_contribution(wrapper, layer_outputs, config)
    
    # PART 5: Summary
    summary = _compute_summary(layer_contribution, attention_contribution, mlp_contribution)
    
    # Compile results
    results = {
        "prompt": config.prompt,
        "num_layers": len(layer_contribution),
        "layer_contribution": layer_contribution,
        "attention_contribution": attention_contribution,
        "mlp_contribution": mlp_contribution,
        "summary": summary,
    }
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


# ============================================================================
# Batch Processing
# ============================================================================

@torch.no_grad()
def run_logit_prism_batch(
    wrapper: LogitLensWrapper,
    prompts: List[str],
    config: LogitPrismConfig,
    *,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Run logit prism decomposition on multiple prompts.
    
    Args:
        wrapper: LogitLensWrapper for the model
        prompts: List of prompts
        config: LogitPrismConfig (prompt field will be overridden)
        output_path: Optional path to save results
    
    Returns:
        Dictionary with results for all prompts
    """
    all_results = {
        "num_prompts": len(prompts),
        "num_layers": None,
        "prompts": [],
    }
    
    for prompt in tqdm(prompts, desc="Processing prompts"):
        # Create config for this prompt
        prompt_config = LogitPrismConfig(
            prompt=prompt,
            use_chat_template=config.use_chat_template,
            prompt_format=config.prompt_format,
            system_prompt=config.system_prompt,
            add_special_tokens=config.add_special_tokens,
            store_full_logits=config.store_full_logits,
        )
        
        # Run decomposition
        result = run_logit_prism(wrapper, prompt_config)
        
        if all_results["num_layers"] is None:
            all_results["num_layers"] = result["num_layers"]
        
        all_results["prompts"].append({
            "prompt": prompt,
            "layer_contribution": result["layer_contribution"],
            "attention_contribution": result["attention_contribution"],
            "mlp_contribution": result["mlp_contribution"],
            "summary": result["summary"],
        })
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return all_results


@torch.no_grad()
def collect_teacher_forced_activations(
    wrapper: LogitLensWrapper,
    config: LogitPrismConfig,
) -> Dict[str, Any]:
    prompt_formatted = _format_generation_prompt(
        wrapper,
        config.prompt,
        prompt_format=config.prompt_format,
        use_chat_template=config.use_chat_template,
        system_prompt=config.system_prompt,
    )

    inputs = wrapper.tokenize_inputs(
        texts=prompt_formatted,
        device=wrapper.model_device,
        add_special_tokens=config.add_special_tokens and not config.use_chat_template,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    hook_buffers: Dict[str, Dict[int, torch.Tensor]] = {
        "attention_outputs": {},
        "mlp_outputs": {},
    }
    hook_handles: List[Any] = []

    def _save_component_hook(component_name: str, layer_idx: int):
        def fn(module, inp, out):
            tensor = wrapper._extract_tensor(out)
            if tensor is None:
                return out
            hook_buffers[component_name][layer_idx] = _detach_full_tensor_to_cpu(tensor)
            return out
        return fn

    for layer_idx, block in enumerate(wrapper.blocks):
        attn_module = _resolve_block_component_module(block, "attention")
        mlp_module = _resolve_block_component_module(block, "mlp")
        if attn_module is None:
            raise ValueError(f"Could not resolve attention module for layer {layer_idx}")
        if mlp_module is None:
            raise ValueError(f"Could not resolve MLP module for layer {layer_idx}")
        hook_handles.append(
            attn_module.register_forward_hook(_save_component_hook("attention_outputs", layer_idx))
        )
        hook_handles.append(
            mlp_module.register_forward_hook(_save_component_hook("mlp_outputs", layer_idx))
        )

    try:
        outputs = wrapper.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )
    finally:
        for hook in hook_handles:
            try:
                hook.remove()
            except Exception:
                pass

    hidden_states = outputs.hidden_states
    logits = outputs.logits
    if hidden_states is None:
        raise ValueError("Model forward pass did not return hidden_states")
    if logits is None:
        raise ValueError("Model forward pass did not return logits")

    expected_num_layers = len(wrapper.blocks)
    if len(hidden_states) != expected_num_layers + 1:
        raise ValueError(
            f"Expected embedding + {expected_num_layers} block hidden states, got {len(hidden_states)}"
        )

    attention_outputs = [hook_buffers["attention_outputs"].get(idx) for idx in range(expected_num_layers)]
    mlp_outputs = [hook_buffers["mlp_outputs"].get(idx) for idx in range(expected_num_layers)]
    if any(tensor is None for tensor in attention_outputs):
        missing = [idx for idx, tensor in enumerate(attention_outputs) if tensor is None]
        raise ValueError(f"Missing attention outputs for layers {missing}")
    if any(tensor is None for tensor in mlp_outputs):
        missing = [idx for idx, tensor in enumerate(mlp_outputs) if tensor is None]
        raise ValueError(f"Missing MLP outputs for layers {missing}")

    return {
        "input_ids": _detach_full_tensor_to_cpu(input_ids),
        "attention_mask": _detach_full_tensor_to_cpu(attention_mask),
        "embedding": _detach_full_tensor_to_cpu(hidden_states[0]),
        "hidden_states": [_detach_full_tensor_to_cpu(tensor) for tensor in hidden_states],
        "logits": _detach_full_tensor_to_cpu(logits),
        "attention_outputs": attention_outputs,
        "mlp_outputs": mlp_outputs,
    }


def _validate_collected_activation_dataset(
    rows: List[Dict[str, Any]],
    *,
    expected_num_examples: int,
) -> None:
    if len(rows) != expected_num_examples:
        raise ValueError(
            f"Collected dataset size mismatch: expected {expected_num_examples}, got {len(rows)}"
        )
    if not rows:
        return

    required_keys = (
        "embedding_base",
        "embedding_ft",
        "hidden_states_base",
        "hidden_states_ft",
        "logits_base",
        "logits_ft",
        "attention_outputs_base",
        "attention_outputs_ft",
        "mlp_outputs_base",
        "mlp_outputs_ft",
        "label",
    )
    sample = rows[0]
    missing = [key for key in required_keys if key not in sample]
    if missing:
        raise ValueError(f"Collected activation dataset is missing keys: {missing}")

    num_layers = len(sample["attention_outputs_base"])
    if len(sample["hidden_states_base"]) != num_layers + 1:
        raise ValueError("hidden_states_base must include embedding + all block layers")
    if len(sample["hidden_states_ft"]) != num_layers + 1:
        raise ValueError("hidden_states_ft must include embedding + all block layers")
    if len(sample["attention_outputs_ft"]) != num_layers:
        raise ValueError("attention_outputs_ft layer count mismatch")
    if len(sample["mlp_outputs_base"]) != num_layers or len(sample["mlp_outputs_ft"]) != num_layers:
        raise ValueError("MLP output layer count mismatch")


def collect_teacher_forced_activations_dataset_pair_incremental(
    *,
    base_wrapper: LogitLensWrapper,
    finetuned_wrapper: LogitLensWrapper,
    dataset_path: str | Path,
    output_path: str | Path,
    partial_path: str | Path | None = None,
    text_field: str = "analysis_text",
    label_field: str = "harmfulness_label_ft",
    add_special_tokens: bool = True,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    partial_path = Path(partial_path) if partial_path is not None else output_path.with_suffix(".partial.pt")

    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Dataset {dataset_path} is empty")

    completed_ids: set[int] = set()
    result_rows: List[Dict[str, Any]] = []

    if partial_path.exists():
        payload = torch.load(partial_path, map_location="cpu")
        result_rows = list(payload.get("rows", []))
        completed_ids = {int(row["id"]) for row in result_rows if row.get("id") is not None}

    def _get_lm_head_weight(wrapper: LogitLensWrapper) -> torch.Tensor:
        lm_head = getattr(wrapper, "lm_head", None)
        if lm_head is None and getattr(wrapper, "model", None) is not None:
            lm_head = wrapper.model.get_output_embeddings()
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise ValueError("Could not resolve LM head weight for collected activation payload")
        return _detach_full_tensor_to_cpu(lm_head.weight)

    def _save_checkpoint() -> Dict[str, Any]:
        payload = {
            "dataset_path": str(dataset_path),
            "text_field": text_field,
            "label_field": label_field,
            "num_examples": len(rows),
            "num_rows_completed": len(result_rows),
            "completed_ids": sorted(completed_ids),
            "lm_head_weight_base": _get_lm_head_weight(base_wrapper),
            "lm_head_weight_ft": _get_lm_head_weight(finetuned_wrapper),
            "rows": result_rows,
        }
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, partial_path)
        return payload

    for row in tqdm(rows, desc="Collecting teacher-forced activations"):
        row_id = int(row["id"])
        if row_id in completed_ids:
            continue
        if text_field not in row:
            raise ValueError(f"Row {row_id} is missing text field '{text_field}'")

        text_value = str(row[text_field])
        config = LogitPrismConfig(
            prompt=text_value,
            use_chat_template=False,
            prompt_format="plain",
            system_prompt=None,
            add_special_tokens=add_special_tokens,
            store_full_logits=False,
        )
        base_result = collect_teacher_forced_activations(base_wrapper, config)
        ft_result = collect_teacher_forced_activations(finetuned_wrapper, config)

        result_rows.append(
            {
                **row,
                "collection_text_field": text_field,
                "collection_prompt_format": "plain",
                "collection_system_prompt": None,
                "label": row.get(label_field),
                "input_ids": base_result["input_ids"],
                "attention_mask": base_result["attention_mask"],
                "embedding_base": base_result["embedding"],
                "embedding_ft": ft_result["embedding"],
                "hidden_states_base": base_result["hidden_states"],
                "hidden_states_ft": ft_result["hidden_states"],
                "logits_base": base_result["logits"],
                "logits_ft": ft_result["logits"],
                "attention_outputs_base": base_result["attention_outputs"],
                "attention_outputs_ft": ft_result["attention_outputs"],
                "mlp_outputs_base": base_result["mlp_outputs"],
                "mlp_outputs_ft": ft_result["mlp_outputs"],
            }
        )
        completed_ids.add(row_id)
        _save_checkpoint()

    _validate_collected_activation_dataset(result_rows, expected_num_examples=len(rows))
    final_payload = _save_checkpoint()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_payload, output_path)
    return final_payload


def run_logit_prism_dataset_pair_incremental(
    *,
    base_wrapper: LogitLensWrapper,
    finetuned_wrapper: LogitLensWrapper,
    dataset_path: str | Path,
    output_path: str | Path,
    partial_path: str | Path | None = None,
    text_field: str = "analysis_text",
    add_special_tokens: bool = True,
) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    partial_path = Path(partial_path) if partial_path is not None else output_path.with_suffix(".partial.json")

    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Dataset {dataset_path} is empty")

    completed_ids: set[int] = set()
    result_rows: List[Dict[str, Any]] = []
    if partial_path.exists():
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
        result_rows = list(payload.get("rows", []))
        completed_ids = {int(row["id"]) for row in result_rows if row.get("id") is not None}

    def _save_checkpoint() -> Dict[str, Any]:
        payload = {
            "dataset_path": str(dataset_path),
            "text_field": text_field,
            "num_rows_total": len(rows),
            "num_rows_completed": len(result_rows),
            "completed_ids": sorted(completed_ids),
            "rows": result_rows,
        }
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    for row in tqdm(rows, desc="Running prism dataset pair"):
        row_id = int(row["id"])
        if row_id in completed_ids:
            continue
        if text_field not in row:
            raise ValueError(f"Row {row_id} is missing text field '{text_field}'")

        text_value = str(row[text_field])
        config = LogitPrismConfig(
            prompt=text_value,
            use_chat_template=False,
            prompt_format="plain",
            system_prompt=None,
            add_special_tokens=add_special_tokens,
            store_full_logits=False,
        )
        base_result = run_logit_prism(base_wrapper, config)
        ft_result = run_logit_prism(finetuned_wrapper, config)

        result_rows.append(
            {
                **row,
                "prism_text_field": text_field,
                "prism_prompt_format": "plain",
                "prism_system_prompt": None,
                "base_prism": base_result,
                "finetuned_prism": ft_result,
            }
        )
        completed_ids.add(row_id)
        _save_checkpoint()

    final_payload = _save_checkpoint()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return final_payload

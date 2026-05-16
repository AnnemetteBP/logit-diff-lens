"""
Cross-model evaluation framework for per-prompt, per-layer, per-position comparison of two transformer models.

This module provides a deterministic, fully-instrumented framework for comparing Model A and Model B
across multiple dimensions including:
- Teacher-forced evaluation
- Autoregressive generation
- Cross-model evaluation (critical)
- Distribution metrics (JS divergence, TVD)
- Top-k analysis with Jaccard similarity
- Hidden state metrics
- Logit lens integration
- Cross-feeding
- Cross-model likelihood
- Structured model evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence
import json
import math

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from diffing.logit_lens_methods.wrapper import LogitLensWrapper, normalize_activations, lmhead_project
from diffing.logit_lens_methods.logitdiff_gen.core import (
    _format_generation_prompt,
    _resolve_layer_indices,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CrossModelEvalConfig:
    """Configuration for cross-model evaluation."""
    
    # Input data
    prompts: List[str] = field(default_factory=list)
    ground_truth_sequences: List[str] | None = None  # Optional ground-truth for teacher-forced
    
    # Models
    model_a_name: str = "model_a"
    model_b_name: str = "model_b"
    
    # Layer selection
    layers: Sequence[float | int] = field(default_factory=lambda: (0.5, 0.6, 0.7, 0.8, 0.9))
    
    # Generation settings
    max_new_tokens: int = 0
    add_special_tokens: bool = True
    prompt_format: Literal["plain", "chat_template", "user_assistant_prefix"] = "plain"
    use_chat_template: bool = False
    system_prompt: str | None = None
    do_sample: bool = False
    temperature: float = 1.0
    use_cache: bool = False
    seed: int | None = None
    
    # Analysis settings
    top_k: int = 10
    norm_mode: str = "raw"  # raw, mean, max
    
    # Output settings
    output_path: str | Path | None = None
    store_full_probs: bool = True  # Store full probability vectors
    store_hidden_states: bool = True  # Store hidden states
    
    # Evaluation modes
    run_teacher_forced: bool = True
    run_generation: bool = True
    run_cross_feeding: bool = True
    run_likelihood: bool = True
    run_structured_eval: bool = False  # Requires JSON output from models


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PositionMetrics:
    """Metrics for a single position."""
    token_id: int
    token_str: str
    is_generated: bool
    source: str  # "prompt", "model_a", "model_b"
    
    # Probabilities
    prob_a: float = 0.0
    prob_b: float = 0.0
    logprob_a: float = 0.0
    logprob_b: float = 0.0
    
    # Distribution metrics
    js_divergence: float = 0.0
    tvd: float = 0.0
    kl_a_b: float = 0.0
    kl_b_a: float = 0.0
    
    # Top-k metrics
    top_k_a: List[int] = field(default_factory=list)
    top_k_b: List[int] = field(default_factory=list)
    jaccard: float = 0.0
    intersection: List[str] = field(default_factory=list)
    only_a: List[str] = field(default_factory=list)
    only_b: List[str] = field(default_factory=list)


@dataclass
class LayerMetrics:
    """Metrics for a single layer across all positions."""
    layer_relative: float
    layer_absolute: int
    
    # Aggregate distribution metrics
    mean_js: float = 0.0
    mean_tvd: float = 0.0
    mean_kl_a_b: float = 0.0
    mean_kl_b_a: float = 0.0
    
    # Top-k aggregate
    mean_jaccard: float = 0.0
    
    # Hidden state metrics
    mean_cosine_sim: float = 0.0
    mean_l2_distance: float = 0.0
    
    # Per-position data
    positions: List[PositionMetrics] = field(default_factory=list)


@dataclass
class PromptResult:
    """Result for a single prompt."""
    prompt: str
    prompt_len: int
    
    # Generated sequences
    y_a: List[int] = field(default_factory=list)
    y_b: List[int] = field(default_factory=list)
    y_a_str: str = ""
    y_b_str: str = ""
    
    # Teacher-forced results
    teacher_forced: Dict[str, Any] = field(default_factory=dict)
    
    # Generation results
    generation: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-feeding results
    cross_feeding: Dict[str, Any] = field(default_factory=dict)
    
    # Likelihood results
    likelihoods: Dict[str, float] = field(default_factory=dict)
    
    # Structured evaluation
    evaluations: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Metric Computation Functions
# ============================================================================

def _compute_js_divergence(p_a: torch.Tensor, p_b: torch.Tensor, epsilon: float = 1e-10) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    # Ensure valid probabilities
    p_a = torch.clamp(p_a, min=epsilon)
    p_b = torch.clamp(p_b, min=epsilon)
    
    # Normalize
    p_a = p_a / p_a.sum()
    p_b = p_b / p_b.sum()
    
    # Mixture
    m = 0.5 * (p_a + p_b)
    m = torch.clamp(m, min=epsilon)
    
    # JS divergence
    kl_a_m = F.kl_div(m.log(), p_a, reduction='sum')
    kl_b_m = F.kl_div(m.log(), p_b, reduction='sum')
    js = 0.5 * (kl_a_m + kl_b_m)
    
    return js.item()


def _compute_tvd(p_a: torch.Tensor, p_b: torch.Tensor) -> float:
    """Compute Total Variation Distance."""
    return 0.5 * torch.sum(torch.abs(p_a - p_b)).item()


def _compute_kl_divergence(p_a: torch.Tensor, p_b: torch.Tensor, epsilon: float = 1e-10) -> float:
    """Compute KL divergence KL(p_a || p_b)."""
    p_a = torch.clamp(p_a, min=epsilon)
    p_b = torch.clamp(p_b, min=epsilon)
    p_a = p_a / p_a.sum()
    p_b = p_b / p_b.sum()
    
    kl = F.kl_div(p_b.log(), p_a, reduction='sum')
    return kl.item()


def _compute_topk_metrics(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    tokenizer: Any,
    top_k: int
) -> Dict[str, Any]:
    """Compute top-k metrics including Jaccard similarity."""
    # Get top-k indices
    _, topk_a_idx = probs_a.topk(top_k)
    _, topk_b_idx = probs_b.topk(top_k)
    
    topk_a_set = set(topk_a_idx.tolist())
    topk_b_set = set(topk_b_idx.tolist())
    
    intersection = topk_a_set & topk_b_set
    only_a = topk_a_set - topk_b_set
    only_b = topk_b_set - topk_a_set
    union = topk_a_set | topk_b_set
    
    # Jaccard similarity
    jaccard = len(intersection) / len(union) if union else 1.0
    
    return {
        "jaccard": jaccard,
        "intersection": [tokenizer.decode([i]) for i in sorted(intersection)],
        "only_a": [tokenizer.decode([i]) for i in sorted(only_a)],
        "only_b": [tokenizer.decode([i]) for i in sorted(only_b)],
        "topk_a": topk_a_idx.tolist(),
        "topk_b": topk_b_idx.tolist(),
    }


def _compute_hidden_state_metrics(
    h_a: torch.Tensor,
    h_b: torch.Tensor
) -> Dict[str, float]:
    """Compute hidden state similarity metrics."""
    # Cosine similarity
    cos_sim = F.cosine_similarity(h_a.unsqueeze(0), h_b.unsqueeze(0)).item()
    
    # L2 distance
    l2_dist = torch.norm(h_a - h_b).item()
    
    return {
        "cosine_similarity": cos_sim,
        "l2_distance": l2_dist,
    }


# ============================================================================
# Core Evaluation Functions
# ============================================================================

@torch.no_grad()
def _run_teacher_forced(
    wrapper_a: LogitLensWrapper,
    wrapper_b: LogitLensWrapper,
    prompt: str,
    ground_truth: str | None,
    config: CrossModelEvalConfig,
    resolved_layers: List[tuple[float, int]],
) -> Dict[str, Any]:
    """Run teacher-forced evaluation."""
    
    # Construct sequence
    if ground_truth:
        full_sequence = prompt + ground_truth
    else:
        full_sequence = prompt
    
    # Format prompt
    prompt_formatted = _format_generation_prompt(
        wrapper_a,
        prompt,
        prompt_format=config.prompt_format,
        use_chat_template=config.use_chat_template,
        system_prompt=config.system_prompt,
    )
    
    # Tokenize
    inputs_a = wrapper_a.tokenize_inputs(
        texts=prompt_formatted,
        device=wrapper_a.model_device,
        add_special_tokens=config.add_special_tokens and not config.use_chat_template,
    )
    
    # Use same tokenization for both models (they share tokenizer)
    input_ids = inputs_a["input_ids"]
    attention_mask = inputs_a["attention_mask"]
    prompt_len = inputs_a["input_ids"].shape[1]
    
    # Get layer indices
    layer_indices = [abs_idx for _, abs_idx in resolved_layers]
    
    # Forward pass for Model A
    acts_a, _ = wrapper_a.forward_pass(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collect_attn=False,
    )
    
    # Forward pass for Model B
    acts_b, _ = wrapper_b.forward_pass(
        input_ids=input_ids,
        attention_mask=attention_mask,
        collect_attn=False,
    )
    
    # Get block names
    block_names = {
        entry["idx"]: name
        for name, entry in wrapper_a.layer_registry.items()
        if entry["type"] == "block" and name in acts_a
    }
    
    # Compute metrics for each layer
    results = {
        "prompt": prompt,
        "ground_truth": ground_truth,
        "prompt_len": prompt_len,
        "sequence_len": input_ids.shape[1],
        "layers": {},
    }
    
    tokenizer = wrapper_a.tokenizer
    
    for layer_rel, layer_abs in resolved_layers:
        if layer_abs not in block_names:
            continue
        
        hidden_a = acts_a[block_names[layer_abs]]
        hidden_b = acts_b[block_names[layer_abs]]
        
        # Normalize and project
        h_norm_a = normalize_activations(
            x=hidden_a.clone(),
            mode=config.norm_mode,
            block="block",
            layer_index=layer_abs,
            model_device=wrapper_a.model_device,
            model_dtype=wrapper_a.model_dtype,
            final_norm=wrapper_a.final_norm,
        )
        h_norm_b = normalize_activations(
            x=hidden_b.clone(),
            mode=config.norm_mode,
            block="block",
            layer_index=layer_abs,
            model_device=wrapper_b.model_device,
            model_dtype=wrapper_b.model_dtype,
            final_norm=wrapper_b.final_norm,
        )
        
        logits_a, _ = lmhead_project(
            x=h_norm_a,
            lm_head=wrapper_a.lm_head,
            stable=wrapper_a.stable,
            model_device=wrapper_a.model_device,
        )
        logits_b, _ = lmhead_project(
            x=h_norm_b,
            lm_head=wrapper_b.lm_head,
            stable=wrapper_b.stable,
            model_device=wrapper_b.model_device,
        )
        
        # Compute probabilities
        probs_a = F.softmax(logits_a, dim=-1)
        probs_b = F.softmax(logits_b, dim=-1)
        logprobs_a = F.log_softmax(logits_a, dim=-1)
        logprobs_b = F.log_softmax(logits_b, dim=-1)
        
        # Compute metrics per position
        positions = []
        for pos in range(input_ids.shape[1]):
            token_id = int(input_ids[0, pos].item())
            
            p_a = probs_a[0, pos]
            p_b = probs_b[0, pos]
            
            # Distribution metrics
            js = _compute_js_divergence(p_a, p_b)
            tvd = _compute_tvd(p_a, p_b)
            kl_a_b = _compute_kl_divergence(p_a, p_b)
            kl_b_a = _compute_kl_divergence(p_b, p_a)
            
            # Top-k metrics
            topk_metrics = _compute_topk_metrics(p_a, p_b, tokenizer, config.top_k)
            
            # Hidden state metrics
            hidden_metrics = _compute_hidden_state_metrics(
                hidden_a[0, pos],
                hidden_b[0, pos]
            )
            
            positions.append({
                "position": pos,
                "token_id": token_id,
                "token_str": tokenizer.decode([token_id]),
                "is_generated": pos >= prompt_len,
                "source": "prompt" if pos < prompt_len else "ground_truth",
                "prob_a": float(p_a[token_id].item()),
                "prob_b": float(p_b[token_id].item()),
                "logprob_a": float(logprobs_a[0, pos, token_id].item()),
                "logprob_b": float(logprobs_b[0, pos, token_id].item()),
                "js_divergence": js,
                "tvd": tvd,
                "kl_a_b": kl_a_b,
                "kl_b_a": kl_b_a,
                **topk_metrics,
                **hidden_metrics,
            })
        
        # Aggregate metrics
        mean_js = sum(p["js_divergence"] for p in positions) / len(positions)
        mean_tvd = sum(p["tvd"] for p in positions) / len(positions)
        mean_jaccard = sum(p["jaccard"] for p in positions) / len(positions)
        mean_cosine = sum(p["cosine_similarity"] for p in positions) / len(positions)
        mean_l2 = sum(p["l2_distance"] for p in positions) / len(positions)
        
        results["layers"][f"layer_{layer_abs}"] = {
            "layer_relative": layer_rel,
            "layer_absolute": layer_abs,
            "mean_js_divergence": mean_js,
            "mean_tvd": mean_tvd,
            "mean_jaccard": mean_jaccard,
            "mean_cosine_similarity": mean_cosine,
            "mean_l2_distance": mean_l2,
            "positions": positions,
        }
    
    return results


@torch.no_grad()
def _run_generation(
    wrapper_a: LogitLensWrapper,
    wrapper_b: LogitLensWrapper,
    prompt: str,
    config: CrossModelEvalConfig,
    resolved_layers: List[tuple[float, int]],
) -> Dict[str, Any]:
    """Run autoregressive generation for both models."""
    
    # Format prompt
    prompt_formatted = _format_generation_prompt(
        wrapper_a,
        prompt,
        prompt_format=config.prompt_format,
        use_chat_template=config.use_chat_template,
        system_prompt=config.system_prompt,
    )
    
    # Tokenize
    inputs = wrapper_a.tokenize_inputs(
        texts=prompt_formatted,
        device=wrapper_a.model_device,
        add_special_tokens=config.add_special_tokens and not config.use_chat_template,
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_len = input_ids.shape[1]
    
    # Generate with Model A
    generated_a = input_ids.detach().clone()
    generated_attn_a = attention_mask.detach().clone()
    
    wrapper_a.model.eval()
    if config.seed is not None:
        torch.manual_seed(int(config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.seed))
    
    for _ in range(config.max_new_tokens):
        outputs = wrapper_a.model(
            input_ids=generated_a,
            attention_mask=generated_attn_a,
            use_cache=config.use_cache,
            return_dict=True,
            output_hidden_states=False,
        )
        logits = outputs.logits[:, -1, :]
        if config.do_sample:
            probs = torch.softmax(logits / max(config.temperature, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_a = torch.cat([generated_a, next_token], dim=-1)
        generated_attn_a = torch.cat([
            generated_attn_a,
            torch.ones((generated_attn_a.shape[0], 1), dtype=generated_attn_a.dtype, device=generated_attn_a.device),
        ], dim=-1)
    
    # Generate with Model B
    generated_b = input_ids.detach().clone()
    generated_attn_b = attention_mask.detach().clone()
    
    wrapper_b.model.eval()
    if config.seed is not None:
        torch.manual_seed(int(config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.seed))
    
    for _ in range(config.max_new_tokens):
        outputs = wrapper_b.model(
            input_ids=generated_b,
            attention_mask=generated_attn_b,
            use_cache=config.use_cache,
            return_dict=True,
            output_hidden_states=False,
        )
        logits = outputs.logits[:, -1, :]
        if config.do_sample:
            probs = torch.softmax(logits / max(config.temperature, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_b = torch.cat([generated_b, next_token], dim=-1)
        generated_attn_b = torch.cat([
            generated_attn_b,
            torch.ones((generated_attn_b.shape[0], 1), dtype=generated_attn_b.dtype, device=generated_attn_b.device),
        ], dim=-1)
    
    # Decode generated sequences
    y_a_str = wrapper_a.tokenizer.decode(generated_a[0, prompt_len:])
    y_b_str = wrapper_b.tokenizer.decode(generated_b[0, prompt_len:])
    
    return {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "y_a": generated_a[0].tolist(),
        "y_b": generated_b[0].tolist(),
        "y_a_str": y_a_str,
        "y_b_str": y_b_str,
    }


@torch.no_grad()
def _run_cross_feeding(
    wrapper_a: LogitLensWrapper,
    wrapper_b: LogitLensWrapper,
    prompt: str,
    y_a: List[int],
    y_b: List[int],
    config: CrossModelEvalConfig,
    resolved_layers: List[tuple[float, int]],
) -> Dict[str, Any]:
    """Run cross-feeding evaluation (critical)."""
    
    # Construct sequences
    # S_A = concat(x, y_A), S_B = concat(x, y_B)
    
    tokenizer = wrapper_a.tokenizer
    
    # Get prompt tokens
    prompt_formatted = _format_generation_prompt(
        wrapper_a,
        prompt,
        prompt_format=config.prompt_format,
        use_chat_template=config.use_chat_template,
        system_prompt=config.system_prompt,
    )
    
    prompt_tokens = tokenizer.encode(prompt_formatted, add_special_tokens=False)
    
    # Construct S_A and S_B
    s_a = prompt_tokens + y_a
    s_b = prompt_tokens + y_b
    
    # Convert to tensors
    input_ids_a = torch.tensor([s_a], device=wrapper_a.model_device)
    input_ids_b = torch.tensor([s_b], device=wrapper_b.model_device)
    
    # Attention mask (all ones)
    attn_a = torch.ones_like(input_ids_a)
    attn_b = torch.ones_like(input_ids_b)
    
    # Get layer indices
    layer_indices = [abs_idx for _, abs_idx in resolved_layers]
    
    # Forward passes
    acts_a_on_sa, _ = wrapper_a.forward_pass(input_ids=input_ids_a, attention_mask=attn_a, collect_attn=False)
    acts_b_on_sa, _ = wrapper_b.forward_pass(input_ids=input_ids_a, attention_mask=attn_a, collect_attn=False)
    acts_a_on_sb, _ = wrapper_a.forward_pass(input_ids=input_ids_b, attention_mask=attn_b, collect_attn=False)
    acts_b_on_sb, _ = wrapper_b.forward_pass(input_ids=input_ids_b, attention_mask=attn_b, collect_attn=False)
    
    # Get block names
    block_names = {
        entry["idx"]: name
        for name, entry in wrapper_a.layer_registry.items()
        if entry["type"] == "block" and name in acts_a_on_sa
    }
    
    results = {
        "sequences": {
            "S_A": s_a,
            "S_B": s_b,
        },
        "layers": {},
    }
    
    for layer_rel, layer_abs in resolved_layers:
        if layer_abs not in block_names:
            continue
        
        # Get hidden states
        hidden_a_on_sa = acts_a_on_sa[block_names[layer_abs]]
        hidden_b_on_sa = acts_b_on_sa[block_names[layer_abs]]
        hidden_a_on_sb = acts_a_on_sb[block_names[layer_abs]]
        hidden_b_on_sb = acts_b_on_sb[block_names[layer_abs]]
        
        # Normalize and project
        h_norm_a_on_sa = normalize_activations(hidden_a_on_sa.clone(), config.norm_mode, "block", layer_abs, wrapper_a.model_device, wrapper_a.model_dtype, wrapper_a.final_norm)
        h_norm_b_on_sa = normalize_activations(hidden_b_on_sa.clone(), config.norm_mode, "block", layer_abs, wrapper_b.model_device, wrapper_b.model_dtype, wrapper_b.final_norm)
        h_norm_a_on_sb = normalize_activations(hidden_a_on_sb.clone(), config.norm_mode, "block", layer_abs, wrapper_a.model_device, wrapper_a.model_dtype, wrapper_a.final_norm)
        h_norm_b_on_sb = normalize_activations(hidden_b_on_sb.clone(), config.norm_mode, "block", layer_abs, wrapper_b.model_device, wrapper_b.model_dtype, wrapper_b.final_norm)
        
        logits_a_on_sa, _ = lmhead_project(h_norm_a_on_sa, wrapper_a.lm_head, wrapper_a.stable, wrapper_a.model_device)
        logits_b_on_sa, _ = lmhead_project(h_norm_b_on_sa, wrapper_b.lm_head, wrapper_b.stable, wrapper_b.model_device)
        logits_a_on_sb, _ = lmhead_project(h_norm_a_on_sb, wrapper_a.lm_head, wrapper_a.stable, wrapper_a.model_device)
        logits_b_on_sb, _ = lmhead_project(h_norm_b_on_sb, wrapper_b.lm_head, wrapper_b.stable, wrapper_b.model_device)
        
        probs_a_on_sa = F.softmax(logits_a_on_sa, dim=-1)
        probs_b_on_sa = F.softmax(logits_b_on_sa, dim=-1)
        probs_a_on_sb = F.softmax(logits_a_on_sb, dim=-1)
        probs_b_on_sb = F.softmax(logits_b_on_sb, dim=-1)
        
        # Compute metrics for each position
        positions = []
        seq_len = input_ids_a.shape[1]
        
        for pos in range(seq_len):
            token_id = int(input_ids_a[0, pos].item())
            
            # Model A on S_A
            p_a_sa = probs_a_on_sa[0, pos]
            # Model B on S_A
            p_b_sa = probs_b_on_sa[0, pos]
            # Model A on S_B
            p_a_sb = probs_a_on_sb[0, pos]
            # Model B on S_B
            p_b_sb = probs_b_on_sb[0, pos]
            
            # Cross-model metrics
            js_sa = _compute_js_divergence(p_a_sa, p_b_sa)
            js_sb = _compute_js_divergence(p_a_sb, p_b_sb)
            tvd_sa = _compute_tvd(p_a_sa, p_b_sa)
            tvd_sb = _compute_tvd(p_a_sb, p_b_sb)
            
            # Top-k metrics
            topk_sa = _compute_topk_metrics(p_a_sa, p_b_sa, tokenizer, config.top_k)
            topk_sb = _compute_topk_metrics(p_a_sb, p_b_sb, tokenizer, config.top_k)
            
            positions.append({
                "position": pos,
                "token_id": token_id,
                "token_str": tokenizer.decode([token_id]),
                "js_divergence_S_A": js_sa,
                "js_divergence_S_B": js_sb,
                "tvd_S_A": tvd_sa,
                "tvd_S_B": tvd_sb,
                "jaccard_S_A": topk_sa["jaccard"],
                "jaccard_S_B": topk_sb["jaccard"],
            })
        
        # Aggregate
        mean_js_sa = sum(p["js_divergence_S_A"] for p in positions) / len(positions)
        mean_js_sb = sum(p["js_divergence_S_B"] for p in positions) / len(positions)
        mean_tvd_sa = sum(p["tvd_S_A"] for p in positions) / len(positions)
        mean_tvd_sb = sum(p["tvd_S_B"] for p in positions) / len(positions)
        
        results["layers"][f"layer_{layer_abs}"] = {
            "layer_relative": layer_rel,
            "layer_absolute": layer_abs,
            "mean_js_divergence_S_A": mean_js_sa,
            "mean_js_divergence_S_B": mean_js_sb,
            "mean_tvd_S_A": mean_tvd_sa,
            "mean_tvd_S_B": mean_tvd_sb,
            "positions": positions,
        }
    
    return results


def _compute_cross_model_likelihood(
    wrapper_a: LogitLensWrapper,
    wrapper_b: LogitLensWrapper,
    prompt: str,
    y_a: List[int],
    y_b: List[int],
    config: CrossModelEvalConfig,
) -> Dict[str, float]:
    """Compute cross-model likelihood."""
    
    tokenizer = wrapper_a.tokenizer
    
    # Get prompt tokens
    prompt_formatted = _format_generation_prompt(
        wrapper_a,
        prompt,
        prompt_format=config.prompt_format,
        use_chat_template=config.use_chat_template,
        system_prompt=config.system_prompt,
    )
    
    prompt_tokens = tokenizer.encode(prompt_formatted, add_special_tokens=False)
    
    # Construct sequences
    s_a = prompt_tokens + y_a
    s_b = prompt_tokens + y_b
    
    # Convert to tensors
    input_ids_a = torch.tensor([s_a], device=wrapper_a.model_device)
    input_ids_b = torch.tensor([s_b], device=wrapper_b.model_device)
    
    epsilon = 1e-10
    
    # Compute LL_A(S_A)
    acts_a, _ = wrapper_a.forward_pass(input_ids=input_ids_a, attention_mask=None, collect_attn=False)
    final_norm_a = wrapper_a.layer_registry["final_norm"]["name"]
    logits_a = wrapper_a.lm_head(acts_a[final_norm_a])
    logprobs_a = F.log_softmax(logits_a, dim=-1)
    
    ll_a_sa = 0.0
    for pos in range(len(s_a)):
        token_id = s_a[pos]
        ll_a_sa += logprobs_a[0, pos, token_id].item()
    
    # Compute LL_B(S_A)
    acts_b, _ = wrapper_b.forward_pass(input_ids=input_ids_a, attention_mask=None, collect_attn=False)
    final_norm_b = wrapper_b.layer_registry["final_norm"]["name"]
    logits_b = wrapper_b.lm_head(acts_b[final_norm_b])
    logprobs_b = F.log_softmax(logits_b, dim=-1)
    
    ll_b_sa = 0.0
    for pos in range(len(s_a)):
        token_id = s_a[pos]
        ll_b_sa += logprobs_b[0, pos, token_id].item()
    
    # Compute LL_A(S_B)
    acts_a_sb, _ = wrapper_a.forward_pass(input_ids=input_ids_b, attention_mask=None, collect_attn=False)
    logits_a_sb = wrapper_a.lm_head(acts_a_sb[final_norm_a])
    logprobs_a_sb = F.log_softmax(logits_a_sb, dim=-1)
    
    ll_a_sb = 0.0
    for pos in range(len(s_b)):
        token_id = s_b[pos]
        ll_a_sb += logprobs_a_sb[0, pos, token_id].item()
    
    # Compute LL_B(S_B)
    acts_b_sb, _ = wrapper_b.forward_pass(input_ids=input_ids_b, attention_mask=None, collect_attn=False)
    logits_b_sb = wrapper_b.lm_head(acts_b_sb[final_norm_b])
    logprobs_b_sb = F.log_softmax(logits_b_sb, dim=-1)
    
    ll_b_sb = 0.0
    for pos in range(len(s_b)):
        token_id = s_b[pos]
        ll_b_sb += logprobs_b_sb[0, pos, token_id].item()
    
    # Compute deltas
    delta_a = ll_a_sa - ll_b_sa  # Δ_A = LL_A(S_A) - LL_B(S_A)
    delta_b = ll_b_sb - ll_a_sb  # Δ_B = LL_B(S_B) - LL_A(S_B)
    
    return {
        "LL_A_S_A": ll_a_sa,
        "LL_B_S_A": ll_b_sa,
        "LL_A_S_B": ll_a_sb,
        "LL_B_S_B": ll_b_sb,
        "delta_A": delta_a,
        "delta_B": delta_b,
    }


# ============================================================================
# Main Run Function
# ============================================================================

@torch.no_grad()
def run_cross_model_eval(
    arch_wrappers: tuple[LogitLensWrapper, LogitLensWrapper],
    config: CrossModelEvalConfig,
    *,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Run comprehensive cross-model evaluation.
    
    Args:
        arch_wrappers: Tuple of (wrapper_a, wrapper_b)
        config: CrossModelEvalConfig
        output_path: Optional path to save results
    
    Returns:
        Dictionary containing all evaluation results
    """
    wrapper_a, wrapper_b = arch_wrappers
    
    # Validate tokenizers
    if wrapper_a.tokenizer.vocab_size != wrapper_b.tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab mismatch: {wrapper_a.tokenizer.vocab_size} vs {wrapper_b.tokenizer.vocab_size}"
        )
    
    # Resolve layers
    resolved_layers = _resolve_layer_indices(wrapper_a, config.layers)
    
    # Initialize results
    all_results = {
        "config": {
            "model_a_name": config.model_a_name,
            "model_b_name": config.model_b_name,
            "num_prompts": len(config.prompts),
            "layers": [rel for rel, _ in resolved_layers],
            "top_k": config.top_k,
            "norm_mode": config.norm_mode,
        },
        "prompts": [],
    }
    
    # Process each prompt
    for prompt in tqdm(config.prompts, desc="Processing prompts"):
        prompt_result = {
            "prompt": prompt,
        }
        
        # 1. Teacher-forced evaluation
        if config.run_teacher_forced:
            ground_truth = config.ground_truth_sequences[config.prompts.index(prompt)] if config.ground_truth_sequences else None
            prompt_result["teacher_forced"] = _run_teacher_forced(
                wrapper_a, wrapper_b, prompt, ground_truth, config, resolved_layers
            )
        
        # 2. Autoregressive generation
        if config.run_generation and config.max_new_tokens > 0:
            prompt_result["generation"] = _run_generation(
                wrapper_a, wrapper_b, prompt, config, resolved_layers
            )
            y_a = prompt_result["generation"]["y_a"][prompt_result["generation"]["prompt_len"]:]
            y_b = prompt_result["generation"]["y_b"][prompt_result["generation"]["prompt_len"]:]
        else:
            y_a = []
            y_b = []
        
        # 3. Cross-feeding
        if config.run_cross_feeding and y_a and y_b:
            prompt_result["cross_feeding"] = _run_cross_feeding(
                wrapper_a, wrapper_b, prompt, y_a, y_b, config, resolved_layers
            )
        
        # 4. Cross-model likelihood
        if config.run_likelihood and y_a and y_b:
            prompt_result["likelihoods"] = _compute_cross_model_likelihood(
                wrapper_a, wrapper_b, prompt, y_a, y_b, config
            )
        
        all_results["prompts"].append(prompt_result)
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return all_results

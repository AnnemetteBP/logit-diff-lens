from typing import Any, Dict
from collections import OrderedDict

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .base_lens_wrapper import BaseLensWrapper
from ..generation_utils import generate_with_model
from ..wrapper_utils import (
    module_is_quantized,
    detect_architecture,
    resolve_backbone,
    find_final_norm,
    build_layer_registry,
    build_component_registry,
)


class GenerateLensWrapper(BaseLensWrapper):
    """
    Generation wrapper that uses model.generate(...) to produce the continuation,
    then replays each generated prefix through model(...) with hooks attached to
    collect activations/logits step-by-step.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        include_final_norm: bool = True,
        fp32_save: bool = True,
        debug: bool = True,
        stable_analysis: bool = True,
    ) -> None:
        super().__init__(model, tokenizer)

        p = next(model.parameters())
        self.model_device = p.device
        self.model_dtype = p.dtype
        self.is_bnb_quantized = any(module_is_quantized(m) for m in model.modules())

        self.stable = True if stable_analysis else False
        self.include_final_norm = include_final_norm
        self.fp32_save = fp32_save
        self.debug = debug

        self.arch = detect_architecture(model=model)
        self.blocks = resolve_backbone(model=model, arch=self.arch)

        self.embedding = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()
        self.final_norm = find_final_norm(model=model)

        self.layer_registry = build_layer_registry(
            embedding=self.embedding,
            include_final_norm=include_final_norm,
            final_norm=self.final_norm,
            lm_head=self.lm_head,
            blocks=self.blocks,
        )
        self.component_registry = build_component_registry(self.blocks)

        self.activations = OrderedDict()
        self.hooks = {}

    def reset_tracing_state(self) -> None:
        self.release_hooks()
        self.activations.clear()

    def save_to_fp32(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        return x.detach().to(dtype=torch.float32, device="cpu")

    def _save_hook(self, name: Any):
        def fn(module, inp, out):
            t = self._extract_tensor(out)
            if t is None:
                return out
            self.activations[name] = t.detach().to(device=self.model_device)
            return out

        return fn

    def attach_hooks(self) -> None:
        self.reset_tracing_state()
        for name, entry in self.layer_registry.items():
            if entry["type"] not in ["embedding", "block"]:
                continue
            self.hooks[name] = entry["module"].register_forward_hook(self._save_hook(name))

    def release_hooks(self) -> None:
        for h in self.hooks.values():
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = {}

    def tokenize_inputs(
        self,
        texts: str | list[str],
        device: str | None = None,
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        try:
            emb_device = self.model.get_input_embeddings().weight.device
        except Exception:
            emb_device = torch.device("cpu")

        device = device or emb_device

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=isinstance(texts, list),
            add_special_tokens=add_special_tokens,
        )

        return {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
        }

    @torch.no_grad()
    def _trace_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        if use_cache:
            raise ValueError(
                "GenerateLensWrapper._trace_prefix uses hooks as the source of truth; "
                "set use_cache=False."
            )
        self.reset_tracing_state()
        self.attach_hooks()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=False,
        )

        logits = outputs.logits[:, -1, :].detach()
        activations = {k: v.detach() for k, v in self.activations.items()}
        self.reset_tracing_state()
        return {"logits": logits, "activations": activations}

    @torch.no_grad()
    def forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 10,
        temperature: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        if output_hidden_states:
            raise ValueError(
                "GenerateLensWrapper uses hooks as the source of truth; "
                "set output_hidden_states=False."
            )
        if use_cache:
            raise ValueError(
                "GenerateLensWrapper uses hooks as the source of truth; "
                "set use_cache=False."
            )
        if input_ids.shape[0] != 1:
            raise ValueError("GenerateLensWrapper currently supports batch size 1.")

        self.model.eval()

        base_attention_mask = (
            attention_mask.detach().clone()
            if attention_mask is not None
            else torch.ones_like(input_ids, device=input_ids.device)
        )
        tokens, _ = generate_with_model(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            attention_mask=base_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            use_cache=False,
            seed=None,
        )

        prompt_len = int(base_attention_mask[0].sum().item())
        total_len = int(tokens.shape[1])
        num_steps = max(0, total_len - prompt_len)

        all_logits = []
        all_activations = []

        for step in range(num_steps):
            prefix_len = prompt_len + step
            prefix_ids = tokens[:, :prefix_len]
            prefix_mask = torch.ones_like(prefix_ids, device=prefix_ids.device)
            traced = self._trace_prefix(prefix_ids, prefix_mask, use_cache=False)
            all_logits.append(traced["logits"])
            all_activations.append(traced["activations"])

        return {
            "tokens": tokens,
            "attention_mask": torch.ones_like(tokens, device=tokens.device),
            "logits": all_logits,
            "activations": all_activations,
        }

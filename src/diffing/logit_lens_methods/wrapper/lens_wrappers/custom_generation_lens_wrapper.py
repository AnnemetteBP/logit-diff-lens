from typing import Any, Tuple, List, Dict
from collections import OrderedDict
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from .base_lens_wrapper import BaseLensWrapper
from ..wrapper_utils import(
    module_is_quantized,
    detect_architecture,
    resolve_backbone,
    find_final_norm,
    build_layer_registry,
    build_component_registry,
)



# ================================================================
#  ARCH GENERATION LOGIT LENS WRAPPER
# ================================================================
class CustomGenerationLensWrapper(BaseLensWrapper):
    def __init__(
            self,
            model:PreTrainedModel,
            tokenizer:PreTrainedTokenizerBase,
            include_final_norm:bool=True,
            fp32_save:bool=True,
            debug:bool=True,
            stable_analysis:bool=True,
        )->None:

        super().__init__(model, tokenizer)

        # Device / dtype
        p = next(model.parameters())
        self.model_device = p.device
        self.model_dtype = p.dtype
        self.is_bnb_quantized = any(module_is_quantized(m) for m in model.modules())

        self.stable = True if stable_analysis else False
        self.include_final_norm = include_final_norm
        self.fp32_save = fp32_save
        self.debug = debug

        # Detect architecture and backbone
        self.arch = detect_architecture(model=model)
        self.blocks = resolve_backbone(model=model, arch=self.arch)

        # Embedding / LM head / final norm
        self.embedding = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()
        self.final_norm = find_final_norm(model=model)

        # Layer registry & hooks
        self.layer_registry = build_layer_registry(
            embedding=self.embedding,
            include_final_norm=include_final_norm,
            final_norm=self.final_norm,
            lm_head=self.lm_head,
            blocks=self.blocks
        )
        self.component_registry = build_component_registry(self.blocks)

        self.activations = OrderedDict()
        self.hooks = {}

        # Print init debug info
        if self.debug:
            print(f"[INIT] Model device={self.model_device}, dtype={self.model_dtype}")
            print(f"[INIT] Embedding module: {self.embedding}")
            print(f"[INIT] LM head module: {self.lm_head}")
            print(f"[INIT] Final norm module: {self.final_norm}")


    def reset_tracing_state(self) -> None:
        self.release_hooks()
        self.activations.clear()


    # -----------------------
    # Hooks
    # -----------------------
    def save_to_fp32(self, x:torch.Tensor) -> torch.Tensor:
        """
        Detach tensor from graph, cast to fp32, move to CPU.
        Safe for analysis/logging.
        """
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        return x.detach().to(dtype=torch.float32, device="cpu")


    def _save_hook(self, name:Any):
        def fn(module, inp, out):
            t = self._extract_tensor(out)
            if t is None:
                return out

            act = t.detach()

            if getattr(self, "clone_activations", False):
                act = act.clone()

            """if self.fp32_save:
                act_store = act.to(device=self.model_device, dtype=torch.float32)
            else:
                act_store = act.to(device=self.model_device)"""
            act_store = act.to(device=self.model_device)
            self.activations[name] = act_store

            return out

        return fn
    
    
    def attach_hooks(self) -> None:
        self.reset_tracing_state()
        for name, entry in self.layer_registry.items():
            if entry["type"] not in ["embedding", "block"]: continue
            self.hooks[name] = entry["module"].register_forward_hook(self._save_hook(name))


    def release_hooks(self) -> None:
        for h in self.hooks.values():
            try: h.remove()
            except: pass
        self.hooks = {}


    def tokenize_inputs(
        self,
        texts:str|List[str],
        device:str|None=None,
        add_special_tokens:bool=True
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
            add_special_tokens=add_special_tokens
        )

        return {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
        }

    # -----------------------
    # Forward generate collector
    # -----------------------
    @torch.no_grad()
    def _generation_step_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> Any:
        self.reset_tracing_state()
        self.attach_hooks()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )

        return outputs


    @torch.no_grad()
    def forward_pass(
        self,
        input_ids:torch.Tensor,
        attention_mask:torch.Tensor|None=None,
        max_new_tokens:int=10,
        temperature:float=1.0,
        do_sample:bool=True,
        use_cache:bool=False,
        output_hidden_states:bool=False,
    ) -> Dict[str, Any]:
        if output_hidden_states:
            raise ValueError(
                "CustomGenerationLensWrapper.forward_pass uses hooks as the source of truth; "
                "set output_hidden_states=False."
            )

        self.model.eval()
        generated = input_ids.detach().clone()
        generated_attention_mask = (
            attention_mask.detach().clone()
            if attention_mask is not None
            else torch.ones_like(generated, device=generated.device)
        )

        all_logits = []
        all_activations = []

        for step in range(max_new_tokens):
            outputs = self._generation_step_forward(
                input_ids=generated,
                attention_mask=generated_attention_mask,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
            )

            logits = outputs.logits[:, -1, :]
            all_logits.append(logits.detach())

            step_acts = {
                k: v.detach()
                for k, v in self.activations.items()
            }
            all_activations.append(step_acts)

            self.reset_tracing_state()

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

        self.reset_tracing_state()

        return {
            "tokens": generated,
            "attention_mask": generated_attention_mask,
            "logits": all_logits,
            "activations": all_activations,
        }

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
#  ARCH PROMPT LOGIT LENS WRAPPER
# ================================================================
class LogitLensWrapper(BaseLensWrapper):
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
        self.release_hooks()
        self.activations.clear()
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
    # Forward collector
    # -----------------------
    @torch.no_grad()
    def forward_pass(
        self,
        input_ids:torch.Tensor,
        attention_mask:torch.Tensor|None=None,
        collect_attn:bool=False
    ) -> Tuple[dict, Any]:

        # --- Hook lifecycle ---
        self.release_hooks()
        self.activations.clear()
        self.attach_hooks()

        # --- Forward pass ---
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=False, # because of hooks
                output_attentions=collect_attn,
                use_cache=False,
            )

        # --- Snapshot activations ---
        acts = {
            k: v.detach() 
            for k, v in self.activations.items()
        }

        self.release_hooks()

        return acts, outputs

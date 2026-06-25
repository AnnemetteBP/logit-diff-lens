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
#  ARCH PATCHING LOGIT LENS WRAPPER
# ================================================================
class PatchingLensWrapper(BaseLensWrapper):
    def __init__(
            self,
            model:PreTrainedModel,
            tokenizer:PreTrainedTokenizerBase,
            include_final_norm:bool=True,
            fp32_save:bool=True,
            debug:bool=True,
            stable_analysis:bool=True,
            patch_config:Dict|None=None
        )->None:

        super().__init__(model, tokenizer)
        self.patch_config = patch_config

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
    # Patch config helpers
    # -----------------------
    def set_patch_config(self, patch_config: Dict | None) -> None:
        self.patch_config = patch_config


    def clear_patch_config(self) -> None:
        self.patch_config = None


    def _normalize_index(self, index: Any) -> Any:
        if index is None:
            return slice(None)
        if isinstance(index, range):
            return list(index)
        if torch.is_tensor(index):
            return index.tolist()
        return index


    def _apply_patch(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if self.patch_config is None:
            return hidden

        patch_layer_idx = self.patch_config.get("layer_idx")
        if patch_layer_idx != layer_idx:
            return hidden

        mode = self.patch_config.get("mode", "replace")
        alpha = float(self.patch_config.get("alpha", 1.0))
        batch_idx = self._normalize_index(self.patch_config.get("batch_idx"))
        token_idx = self._normalize_index(self.patch_config.get("token_idx"))
        feature_idx = self._normalize_index(self.patch_config.get("feature_idx"))

        patch = self.patch_config.get("tensor")
        if patch is None:
            raise ValueError("patch_config requires a 'tensor' entry")
        if not torch.is_tensor(patch):
            raise TypeError(f"patch_config['tensor'] must be a tensor, got {type(patch)}")

        patched = hidden.clone()
        target = patched[batch_idx, token_idx, feature_idx]

        patch = patch.to(device=target.device, dtype=target.dtype)

        if patch.shape != target.shape:
            try:
                patch = torch.broadcast_to(patch, target.shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"Patch tensor shape {tuple(patch.shape)} is not compatible with "
                    f"target slice shape {tuple(target.shape)}"
                ) from exc

        if mode == "add":
            patched[batch_idx, token_idx, feature_idx] = target + alpha * patch
        elif mode == "replace":
            patched[batch_idx, token_idx, feature_idx] = patch
        else:
            raise ValueError(f"Unsupported patch mode: {mode}")

        return patched


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


    def _save_hook(self, name: Any, layer_idx: int):
        def fn(module, inp, out):
            t = self._extract_tensor(out)
            if t is None:
                return out

            h = t

            # Apply the configured patch to the selected layer/token slice.
            h = self._apply_patch(h, layer_idx)

            # ---------------------------
            # SAVE
            # ---------------------------
            act = h.detach()

            if getattr(self, "clone_activations", False):
                act = act.clone()

            act_store = act.to(device=self.model_device)

            self.activations[name] = act_store

            return self._replace_tensor(out, h)

        return fn
    
    def attach_hooks(self) -> None:
        self.release_hooks()
        self.activations.clear()

        for name, entry in self.layer_registry.items():
            if entry["type"] not in ["embedding", "block"]:
                continue

            layer_idx = entry["idx"]

            self.hooks[name] = entry["module"].register_forward_hook(
                self._save_hook(name, layer_idx)
            )


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
        # patched
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

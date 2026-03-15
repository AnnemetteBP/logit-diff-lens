from typing import Any
import torch
import torch.nn.functional as F
from collections import OrderedDict
from contextlib import contextmanager



# ================================================================
#  QUANTIZATION HELPERS
# ================================================================
def is_bnb_4bit(m): return 'Linear4bit' in m.__class__.__name__
def is_bnb_8bit(m): return 'Linear8bitLt' in m.__class__.__name__
def is_gptq(m):     return hasattr(m, 'quant_state')
def is_awq(m):      return hasattr(m, 'scales') and hasattr(m, 'zero_points')
def module_is_quantized(m):
    return is_bnb_4bit(m) or is_bnb_8bit(m) or is_gptq(m) or is_awq(m)


# ================================================================
#  ARCH WRAPPER
# ================================================================
class ArchWrapper:
    def __init__(
            self,
            model:Any,
            tokenizer:Any,
            include_final_norm:bool=True,
            fp32_save:bool=True,
            debug:bool=True,
            stable_analysis:bool=True,
        )->None:

        self.model = model
        self.tokenizer = tokenizer

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
        self.arch = self._detect_architecture(model)
        self.blocks = self._resolve_backbone()

        # Embedding / LM head / final norm
        self.embedding = model.get_input_embeddings()
        self.lm_head = model.get_output_embeddings()
        self.final_norm = self._find_final_norm()

        # Layer registry & hooks
        self.layer_registry = self._build_layer_registry()
        self.activations = OrderedDict()
        self.hooks = {}

        # Print init debug info
        if self.debug:
            print(f"[INIT] Model device={self.model_device}, dtype={self.model_dtype}")
            print(f"[INIT] Embedding module: {self.embedding}")
            print(f"[INIT] LM head module: {self.lm_head}")
            print(f"[INIT] Final norm module: {self.final_norm}")

    
    # -----------------------
    # Architecture detection
    # -----------------------
    def _detect_architecture(self, model:Any) -> str|Any:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return "model"
        if hasattr(model, "base_model") and hasattr(model.base_model, "layers"):
            return "base"
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return "transformer"
        if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
            return "decoder"
        return "unknown"

    def _resolve_backbone(self) -> list[Any]:
        m = self.model
        if self.arch == "model":
            return list(m.model.layers)
        if self.arch == "base":
            return list(m.base_model.layers)
        if self.arch == "transformer":
            return list(m.transformer.h)
        if self.arch == "decoder":
            return list(m.model.decoder.layers)
        raise RuntimeError("Cannot resolve backbone for architecture")

    def _find_final_norm(self) -> Any|None:
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "norm"): return m.model.norm
        if hasattr(m, "transformer") and hasattr(m.transformer, "ln_f"): return m.transformer.ln_f
        if hasattr(m, "base_model") and hasattr(m.base_model, "final_layer_norm"): return m.base_model.final_layer_norm
        if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "final_layer_norm"): return m.model.decoder.final_layer_norm
        return None


    # -----------------------
    # Layer registry
    # -----------------------
    def _build_layer_registry(self) -> OrderedDict:
        reg = OrderedDict()
        reg["embedding"] = {"module": self.embedding, "type": "embedding", "idx": -1}
        for i, block in enumerate(self.blocks):
            reg[f"layer_{i:02d}"] = {"module": block, "type": "block", "idx": i}
        if self.include_final_norm and self.final_norm is not None:
            reg["final_norm"] = {"module": self.final_norm, "type": "final_norm", "idx": len(self.blocks)}
        if self.lm_head is not None:
            reg["lm_head"] = {"module": self.lm_head, "type": "lm_head", "idx": len(self.blocks)+1}
        return reg


    # -----------------------
    # Hooks
    # -----------------------
    @contextmanager
    def hooked(self):
        """
        with ArchWrapper.hooked():
            model(**inputs)
        """
        self.attach_hooks()
        try:
            yield self
        finally:
            self.release_hooks()

    def _extract_tensor(self, out:Any):
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
            return out[0]
        return None
    
    def save_to_fp32(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detach tensor from graph, cast to fp32, move to CPU.
        Safe for analysis/logging.
        """
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        return x.detach().to(dtype=torch.float32, device="cpu")


    def _save_hook(self, name: Any):
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


    # -----------------------
    # Debug helpers
    # -----------------------
    def _dbg(self, msg:str) -> None:
        if self.debug:
            print(msg)

    def _dbg_tensor(self, name:str, t:torch.Tensor) -> None:
        if not self.debug:
            return
        if t is None or not torch.is_tensor(t):
            print(f"[DBG] {name}: <non-tensor>")
            return
        print(f"[DBG] {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
              f"min={float(t.min()):+.4e} max={float(t.max()):+.4e} "
              f"+inf={torch.isposinf(t).sum().item()} -inf={torch.isneginf(t).sum().item()} nan={torch.isnan(t).sum().item()}")

    def _assert_isfinite(self, x:torch.Tensor):
        self._dbg_tensor(name="Tensor", t=x)
        assert torch.isfinite(x).all(), "NaN/Inf detected"


    # -----------------------
    # Normalization
    # -----------------------
    def _prepare_layer_norm(
        self,
        ln:torch.nn.Module,
        device:torch.device,
        dtype:torch.dtype,
    ) -> torch.nn.Module:
        if ln is None:
            return None
        if next(ln.parameters()).device != device or next(ln.parameters()).dtype != dtype:
            ln = ln.to(device=device, dtype=dtype)
        return ln

    def apply_normalization(
        self,
        x:torch.Tensor,
        mode:str="raw",
        block:str|None=None,
        layer_index:int|None=None,
    ) -> torch.Tensor:

        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x):
            return None

        device = self.model_device
        dtype = self.model_dtype
        x = x.to(device=device, dtype=dtype)
        self._assert_isfinite(x)
        
        # --------------------------------------------------
        # EMBEDDING
        # --------------------------------------------------
        if layer_index == -1 and block == "embedding":
            return x

        # --------------------------------------------------
        # OUTPUT (synthetic N+1)
        # --------------------------------------------------
        if block == "output":
            fn = self._prepare_layer_norm(self.final_norm, device, dtype)
            return fn(x) if fn is not None else x

        # --------------------------------------------------
        # RAW
        # --------------------------------------------------
        if mode == "raw" and block == "block":
            return x

        # --------------------------------------------------
        # EXPERIMENTAL
        # --------------------------------------------------
        if mode == "unit_norm" and block == "block":
            rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            return x / rms

        if mode == "eps_norm" and block == "block":
            rms = x.pow(2).mean(-1, keepdim=True).sqrt()
            return x / (rms + 1e-5)

        if mode == "model_norm" and block == "block":
            fn = self._prepare_layer_norm(self.final_norm, device, dtype)
            return fn(x) if fn is not None else x

        return x


    # -----------------------
    # LM projection
    # -----------------------
    def _is_quantized_lm_head(self) -> bool:
        return self.lm_head is not None and module_is_quantized(self.lm_head)


    def project(self, x:torch.Tensor) -> tuple[torch.Tensor, dict]:

        if isinstance(x, (tuple, list)):
            x = x[0]
        if not torch.is_tensor(x):
            return torch.zeros(1, 1), {"invalid": True}

        lm = self.lm_head
        if lm is None:
            return torch.zeros(1, 1), {"invalid": True}

        device = self.model_device
        quantized = self._is_quantized_lm_head()

        # ==================================================
        # NATIVE MODE — FAITHFUL
        # ==================================================
        if not self.stable:
            x_native = x.to(device=device)
            self._assert_isfinite(x_native)
            # -------- Quantized LM head (BNB / GPTQ / etc.)
            if quantized:
                preferred = next(lm.parameters()).dtype
                logits = lm(x_native.to(preferred))
                return logits, {
                    "mode": "native",
                    "quantized": True,
                    "native_kernel": True,
                    "dtype": logits.dtype,
                }

            # -------- Non-quantized (pure native matmul)
            w = getattr(lm, "weight", None)
            b = getattr(lm, "bias", None)

            logits = x_native @ w.T
            if b is not None:
                logits = logits + b

            return logits, {
                "mode": "native",
                "quantized": False,
                "native_kernel": True,
                "dtype": logits.dtype,
            }

        # ==================================================
        # STABLE MODE — FP32 PROJECTION (NUMERICALLY SAFE)
        # ==================================================
        x32 = x.to(device=device, dtype=torch.float32)
        self._assert_isfinite(x32)

        # -------- Quantized LM head (preferred path)
        if quantized:
            try:
                preferred = next(lm.parameters()).dtype
                logits = lm(x32.to(preferred))
                return logits.to(torch.float32), {
                    "mode": "stable",
                    "quantized": True,
                    "native_kernel": True,
                    "dtype": torch.float32,
                }
            except Exception:
                pass  # fall through to manual FP32 path

        # -------- Manual FP32 matmul (robust fallback)
        w_raw = getattr(lm, "weight", None)
        b_raw = getattr(lm, "bias", None)

        if quantized and hasattr(w_raw, "dequantize"):
            w32 = w_raw.dequantize().to(device=device, dtype=torch.float32)
        else:
            w32 = w_raw.detach().to(device=device, dtype=torch.float32)

        if b_raw is not None:
            if quantized and hasattr(b_raw, "dequantize"):
                b32 = b_raw.dequantize().to(device=device, dtype=torch.float32)
            else:
                b32 = b_raw.detach().to(device=device, dtype=torch.float32)
        else:
            b32 = None

        logits32 = x32 @ w32.T
        if b32 is not None:
            logits32 = logits32 + b32

        return logits32, {
            "mode": "stable",
            "quantized": quantized,
            "native_kernel": False,
            "dtype": torch.float32,
        }

    
    # -----------------------
    # Prepare inputs
    # -----------------------
    def as_tensor(self, x:Any, device:Any) -> torch.Tensor:

        if isinstance(x, torch.Tensor):
            return x.to(device)

        if isinstance(x, list):
            if len(x) == 0:
                raise ValueError("Empty input_ids")

            # list of tensors → stack
            if isinstance(x[0], torch.Tensor):
                return torch.stack(x).to(device)

            # list of lists / ints → tensor
            return torch.tensor(x, dtype=torch.long, device=device)

        raise TypeError(f"Unsupported type: {type(x)}")

    def plotter_inputs(
        self,
        texts:str|list[str],
        device:str|None=None,
        add_special_tokens:bool=False,
    ) -> dict:
        
        if isinstance(texts, str):
            texts = [texts]

        tok = self.tokenizer

        # device
        try:
            emb_device = self.model.get_input_embeddings().weight.device
        except Exception:
            emb_device = torch.device("cpu")

        device = device or emb_device

        input_ids = []
        target_ids = []

        for text in texts:
            ids = tok.encode(text, add_special_tokens=add_special_tokens)
            ids = torch.tensor(ids, dtype=torch.long, device=device)

            if len(ids) < 2:
                raise ValueError("Sequence too short for next-token prediction")

            x = ids[:-1]   # input
            y = ids[1:]    # target

            input_ids.append(x)
            target_ids.append(y)

        return {
            "input_ids": input_ids,    # list[Tensor[T]]
            "target_ids": target_ids,  # list[Tensor[T]]
        }

    def prepare_inputs(
            self,
            texts:str|list,
            device:str|None=None,
            add_special_tokens:bool=False
        ) -> dict[list]:
        
        if isinstance(texts, str):
            texts = [texts]

        try:
            emb_device = self.model.get_input_embeddings().weight.device
        except Exception:
            emb_device = torch.device("cpu")

        device = device or emb_device

        print("=== PREPARE_INPUTS CALLED ===")
        for i, text in enumerate(texts):
            print("IDX", i)
            print("REPR:", repr(text))
            print("ENCODE:", self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

        all_ids = []

        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            if len(ids) < 2:
                raise ValueError("Sequence too short")
            all_ids.append(torch.tensor(ids, dtype=torch.long, device=device))

        return {
            "input_ids": all_ids 
        }


    # -----------------------
    # Forward collector
    # -----------------------
    @torch.no_grad()
    def forward_collect(
        self,
        input_ids:torch.Tensor,
        collect_attn:bool=False
    ) -> tuple[dict, Any]:

        # --- Hook lifecycle ---
        self.release_hooks()
        self.activations.clear()
        self.attach_hooks()

        # --- Forward pass ---
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
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
    
    # -----------------------
    # Forward generate collector
    # -----------------------
    @torch.no_grad()
    def generate_collect(
        self,
        input_ids:torch.Tensor,
        max_new_tokens:int=20
    ):

        # --- Hook lifecycle ---
        self.release_hooks()
        self.activations.clear()
        self.attach_hooks()

        # --- Generate ---
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=False,
            return_dict_in_generate=True,
            output_attentions=False
        )

        # --- Snapshot activations ---
        acts = {k: v.detach() for k,v in self.activations.items()}

        self.release_hooks()

        return acts, outputs
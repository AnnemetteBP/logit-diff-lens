from typing import Any, Tuple, List, Dict
from collections import OrderedDict
import torch



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
#  ARCHITECTURE HELPERS
# ================================================================

# -----------------------
# Architecture detection
# -----------------------
def detect_architecture(model:Any) -> str|Any:
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return "model.model"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "model"
    if hasattr(model, "base_model") and hasattr(model.base_model, "layers"):
        return "base"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "transformer"
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return "decoder"
    return "unknown"


def resolve_backbone(model:Any, arch:str) -> List[Any]:
    m = model
    if arch == "model.model":
        return list(m.model.model.layers)
    if arch == "model":
        return list(m.model.layers)
    if arch == "base":
        return list(m.base_model.layers)
    if arch == "transformer":
        return list(m.transformer.h)
    if arch == "decoder":
        return list(m.model.decoder.layers)
    raise RuntimeError("Cannot resolve backbone for architecture")


def find_final_norm(model:Any) -> Any|None:
    m = model
    if hasattr(m, "model") and hasattr(m.model, "model") and hasattr(m.model.model, "norm"):
        return m.model.model.norm
    if hasattr(m, "model") and hasattr(m.model, "norm"): return m.model.norm
    if hasattr(m, "transformer") and hasattr(m.transformer, "ln_f"): return m.transformer.ln_f
    if hasattr(m, "base_model") and hasattr(m.base_model, "final_layer_norm"): return m.base_model.final_layer_norm
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "final_layer_norm"): return m.model.decoder.final_layer_norm
    return None


# -----------------------
# Layer registry
# -----------------------
def build_layer_registry(
        embedding:Any,
        include_final_norm:bool,
        final_norm:Any|None,
        lm_head:Any|None,
        blocks:List[Any]
) -> OrderedDict:
    
    reg = OrderedDict()
    reg["embedding"] = {"module": embedding, "type": "embedding", "idx": -1}
    for i, block in enumerate(blocks):
        reg[f"layer_{i:02d}"] = {"module": block, "type": "block", "idx": i}
    if include_final_norm and final_norm is not None:
        reg["final_norm"] = {"module": final_norm, "type": "final_norm", "idx": len(blocks)}
    if lm_head is not None:
        reg["lm_head"] = {"module": lm_head, "type": "lm_head", "idx": len(blocks)+1}
    return reg


# ================================================================
#  DEBUGGING
# ================================================================

def dbg(msg:str) -> None:
        print(msg)

def dbg_tensor(name:str, t:torch.Tensor) -> None:
    if t is None or not torch.is_tensor(t):
        print(f"[DBG] {name}: <non-tensor>")
        return
    print(f"[DBG] {name}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
            f"min={float(t.min()):+.4e} max={float(t.max()):+.4e} "
            f"+inf={torch.isposinf(t).sum().item()} -inf={torch.isneginf(t).sum().item()} nan={torch.isnan(t).sum().item()}")

def assert_isfinite(x:torch.Tensor):
    if not torch.isfinite(x).all():
        dbg_tensor(name="Tensor", t=x)
        raise AssertionError("NaN/Inf detected")


# ================================================================
#  NORMALIZATION & PROJECTION
# ================================================================
def prepare_layer_norm(ln:torch.nn.Module, device:torch.device, dtype:torch.dtype) -> torch.nn.Module:
    if ln is None:
        return None
    if next(ln.parameters()).device != device or next(ln.parameters()).dtype != dtype:
        ln = ln.to(device=device, dtype=dtype)
    return ln


def normalize_activations(
        x:torch.Tensor,
        mode:str="raw",
        block:str|None=None,
        layer_index:int|None=None,
        layer_idx:int|None=None,
        model_device:torch.device|str|None=None,
        model_dtype:torch.dtype|None=None,
        final_norm:torch.nn.Module|None=None
) -> torch.Tensor:

    if isinstance(x, (tuple, list)):
        x = x[0]
    if not torch.is_tensor(x):
        return None

    device = model_device if model_device is not None else x.device
    dtype = model_dtype if model_dtype is not None else x.dtype
    if layer_index is None:
        layer_index = layer_idx
    x = x.to(device=device, dtype=dtype)
    assert_isfinite(x)
    
    # --------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------
    if layer_index == -1 and block == "embedding":
        return x

    # --------------------------------------------------
    # OUTPUT (synthetic N+1)
    # --------------------------------------------------
    if block == "output":
        fn = prepare_layer_norm(final_norm, device, dtype)
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
        fn = prepare_layer_norm(final_norm, device, dtype)
        return fn(x) if fn is not None else x

    return x


# -----------------------
# LM projection
# -----------------------
def is_quantized_lm_head(lm_head:Any) -> bool:
    return lm_head is not None and module_is_quantized(lm_head)


def lmhead_project(
        x:torch.Tensor,
        lm_head:Any,
        stable:bool,
        model_device:torch.device|str|None=None
) -> Tuple[torch.Tensor, dict]:

    if isinstance(x, (tuple, list)):
        x = x[0]
    if not torch.is_tensor(x):
        return torch.zeros(1, 1), {"invalid": True}

    lm = lm_head
    if lm is None:
        return torch.zeros(1, 1), {"invalid": True}

    device = model_device if model_device is not None else x.device
    quantized = is_quantized_lm_head(lm)

    # ==================================================
    # NATIVE MODE — FAITHFUL
    # ==================================================
    if not stable:
        x_native = x.to(device=device)
        assert_isfinite(x_native)
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
    assert_isfinite(x32)

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


# ================================================================
#  TENSOR UTILITIES
# ================================================================
def as_tensor(x:Any, device:Any) -> torch.Tensor:

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

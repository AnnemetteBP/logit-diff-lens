from typing import Any, List
from abc import ABC, abstractmethod
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from ...schemas.wrappers import (
    WrapperCapabilities,
    WrapperModelMetadata,
    WrapperOutputSemantics,
)


class BaseLensWrapper(ABC):
    """Abstract base class for lens wrappers that provides common functionality and structure for different types of lens wrappers. This class can be extended by specific lens wrapper implementations to ensure a consistent interface and shared methods for applying logit (difference) lenses across different transformer architectures."""

    def __init__(self, model:PreTrainedModel, tokenizer:PreTrainedTokenizerBase, **kwargs) -> None:
        
        self.model = model
        self.tokenizer = tokenizer

    @property
    def backend_name(self) -> str:
        return "huggingface"

    @property
    def collection_mode(self) -> str:
        return "prompt"

    def build_capabilities(self) -> WrapperCapabilities:
        return WrapperCapabilities(
            collection_mode=self.collection_mode,
            backend_name=self.backend_name,
            supports_prompt_forward=True,
            supports_generation_forward=self.collection_mode in {"generation", "custom"},
            supports_patching=self.collection_mode == "patching",
            supports_attention_subblocks=bool(getattr(self, "component_registry", {}))
            and any("attention_" in str(name) for name in getattr(self, "component_registry", {}).keys()),
            supports_mlp_subblocks=bool(getattr(self, "component_registry", {}))
            and any("mlp_" in str(name) for name in getattr(self, "component_registry", {}).keys()),
            supports_component_registry=hasattr(self, "component_registry"),
            supports_final_norm=getattr(self, "final_norm", None) is not None,
            supports_lm_head_bias=self._has_lm_head_bias(),
            supports_tied_unembed_detection=True,
            supports_quantized_lm_head_projection=bool(getattr(self, "is_bnb_quantized", False)) or self._has_quantized_lm_head(),
            supports_manual_dequantization_fallback=True,
            supports_stable_fp32_projection=True,
            supports_hidden_state_replay=self.collection_mode == "generation",
            supports_generate_api=self.collection_mode == "generation",
            supports_cache_aware_generation=False,
        )

    def build_model_metadata(self) -> WrapperModelMetadata:
        tokenizer = self.tokenizer
        return WrapperModelMetadata(
            wrapper_name=self.__class__.__name__,
            backend_name=self.backend_name,
            architecture=str(getattr(self, "arch", "unknown")),
            model_class=self.model.__class__.__name__,
            tokenizer_class=tokenizer.__class__.__name__ if tokenizer is not None else "None",
            device=self._infer_device(),
            dtype=self._infer_dtype(),
            quantization_kind=self._infer_quantization_kind(),
            is_quantized=bool(getattr(self, "is_bnb_quantized", False) or self._has_quantized_lm_head()),
            embeddings_tied=self._embeddings_are_tied(),
            has_final_norm=getattr(self, "final_norm", None) is not None,
            has_lm_head_bias=self._has_lm_head_bias(),
            layer_count=self._infer_layer_count(),
            supports_attention_module_resolution=bool(getattr(self, "component_registry", {}))
            and any("attention_" in str(name) for name in getattr(self, "component_registry", {}).keys()),
            supports_mlp_module_resolution=bool(getattr(self, "component_registry", {}))
            and any("mlp_" in str(name) for name in getattr(self, "component_registry", {}).keys()),
            extra={
                "include_final_norm": bool(getattr(self, "include_final_norm", False)),
                "stable_analysis": bool(getattr(self, "stable", False)),
            },
        )

    def describe_output_semantics(self) -> WrapperOutputSemantics:
        return WrapperOutputSemantics(
            collection_mode=self.collection_mode,
            captures_embedding_output=True,
            captures_block_output=True,
            captures_attention_output=False,
            captures_mlp_output=False,
            captures_final_norm_output=False,
            block_output_semantics="Post-block module output captured via forward hooks.",
            activation_device_policy="Activations are stored on the model device unless an implementation documents otherwise.",
            generation_semantics=(
                "Generated continuations are analyzed by replaying prefixes through model forward passes."
                if self.collection_mode == "generation"
                else None
            ),
        )

    def _infer_layer_count(self) -> int | None:
        blocks = getattr(self, "blocks", None)
        if blocks is not None:
            try:
                return len(blocks)
            except TypeError:
                return None
        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "num_hidden_layers"):
            return int(config.num_hidden_layers)
        return None

    def _infer_device(self) -> str:
        try:
            return str(next(self.model.parameters()).device)
        except StopIteration:
            return "unknown"

    def _infer_dtype(self) -> str:
        try:
            return str(next(self.model.parameters()).dtype).replace("torch.", "")
        except StopIteration:
            return "unknown"

    def _has_lm_head_bias(self) -> bool | None:
        lm_head = getattr(self, "lm_head", None)
        if lm_head is None:
            try:
                lm_head = self.model.get_output_embeddings()
            except Exception:
                return None
        return getattr(lm_head, "bias", None) is not None

    def _has_quantized_lm_head(self) -> bool:
        lm_head = getattr(self, "lm_head", None)
        if lm_head is None:
            return False
        name = lm_head.__class__.__name__
        return (
            "Linear4bit" in name
            or "Linear8bitLt" in name
            or hasattr(lm_head, "quant_state")
            or (hasattr(lm_head, "scales") and hasattr(lm_head, "zero_points"))
        )

    def _infer_quantization_kind(self) -> str:
        if not bool(getattr(self, "is_bnb_quantized", False) or self._has_quantized_lm_head()):
            return "none"
        modules = list(self.model.modules())
        if any("Linear4bit" in module.__class__.__name__ for module in modules):
            return "bitsandbytes_4bit"
        if any("Linear8bitLt" in module.__class__.__name__ for module in modules):
            return "bitsandbytes_8bit"
        if any(hasattr(module, "quant_state") for module in modules):
            return "gptq"
        if any(hasattr(module, "scales") and hasattr(module, "zero_points") for module in modules):
            return "awq"
        return "unknown"

    def _embeddings_are_tied(self) -> bool | None:
        try:
            input_embeddings = self.model.get_input_embeddings()
            output_embeddings = self.model.get_output_embeddings()
        except Exception:
            return None
        if input_embeddings is None or output_embeddings is None:
            return None
        input_weight = getattr(input_embeddings, "weight", None)
        output_weight = getattr(output_embeddings, "weight", None)
        if input_weight is None or output_weight is None:
            return None
        return input_weight.data_ptr() == output_weight.data_ptr()


    def _extract_tensor(self, out:Any) -> torch.Tensor | None:
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)):
            for item in out:
                if torch.is_tensor(item):
                    return item
        if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            return out.last_hidden_state
        return None


    def _replace_tensor(self, out: Any, new_tensor: torch.Tensor) -> Any:
        if torch.is_tensor(out):
            return new_tensor
        if isinstance(out, tuple):
            replaced = False
            items = []
            for item in out:
                if not replaced and torch.is_tensor(item):
                    items.append(new_tensor)
                    replaced = True
                else:
                    items.append(item)
            return tuple(items)
        if isinstance(out, list):
            replaced = False
            items = []
            for item in out:
                if not replaced and torch.is_tensor(item):
                    items.append(new_tensor)
                    replaced = True
                else:
                    items.append(item)
            return items
        if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            try:
                out.last_hidden_state = new_tensor
                return out
            except Exception:
                return out
        return out


    @abstractmethod
    def attach_hooks(self) -> None:
        """Attaches the necessary hooks to the model for capturing activations during lensing. This method should be implemented by each specific lens wrapper to attach hooks to the appropriate layers of the model based on its architecture."""
        pass


    @abstractmethod
    def release_hooks(self) -> None:
        """Releases the hooks that were attached to the model after lensing is complete. This method should be implemented by each specific lens wrapper to ensure that hooks are properly removed and do not interfere with subsequent model usage."""
        pass


    @abstractmethod
    def tokenize_inputs(self, inputs:str|List[str], **kwargs) -> torch.Tensor:
        """Tokenizes the input text using the wrapper's tokenizer and returns a tensor of input IDs."""
        pass


    @abstractmethod
    def forward_pass(self, input_ids:torch.Tensor, **kwargs) -> Any:
        """Runs the model on the given input_ids and other relevant arguments, and returns the model outputs in a standardized format for lensing."""
        pass

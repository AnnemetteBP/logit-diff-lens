from typing import Any, Tuple, List
from abc import ABC, abstractmethod
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel



class BaseLensWrapper(ABC):
    """Abstract base class for lens wrappers that provides common functionality and structure for different types of lens wrappers. This class can be extended by specific lens wrapper implementations to ensure a consistent interface and shared methods for applying logit (difference) lenses across different transformer architectures."""

    def __init__(self, model:PreTrainedModel, tokenizer:PreTrainedTokenizerBase, **kwargs) -> None:
        
        self.model = model
        self.tokenizer = tokenizer


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

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

_TOKENIZER_CACHE: dict[str, PreTrainedTokenizerBase] = {}


def load_tokenizer(model_name: str, chat_template: str | None = None) -> PreTrainedTokenizerBase:
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except AttributeError as e:
        model_path = Path(model_name)
        tokenizer_json = model_path / "tokenizer.json"
        tokenizer_config = model_path / "tokenizer_config.json"
        if (
            "object has no attribute 'keys'" in str(e)
            and tokenizer_json.exists()
            and tokenizer_config.exists()
        ):
            cfg = json.loads(tokenizer_config.read_text(encoding="utf-8"))
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
            eos_token = cfg.get("eos_token")
            pad_token = cfg.get("pad_token") or eos_token
            bos_token = cfg.get("bos_token") or eos_token
            if eos_token is not None:
                tokenizer.eos_token = eos_token
            if pad_token is not None:
                tokenizer.pad_token = pad_token
            if bos_token is not None:
                tokenizer.bos_token = bos_token
        else:
            raise

    if chat_template is not None:
        tokenizer.chat_template = chat_template

    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer

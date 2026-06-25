from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from ...tokenizer_loading import load_tokenizer


def _render_token(tokenizer: Any, token_id: int) -> str:
    decoded = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    raw = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    text = decoded if decoded else str(raw)
    return text.replace("\n", "\\n")


def _is_meaningful_token(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) == 1 and not stripped.isalnum():
        return False
    if not any(ch.isalpha() for ch in stripped):
        return False
    if re.fullmatch(r"[_=\-+/\\|:;,.!?()\[\]{}<>@#$%^&*~`'\"]+", stripped):
        return False
    return True


def _top_token_loadings(vector: np.ndarray, tokenizer: Any, top_k: int, *, filter_meaningful: bool) -> dict[str, Any]:
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    k = max(1, min(int(top_k), int(vec.shape[0])))

    pos_idx = np.argsort(-vec)[:k]
    neg_idx = np.argsort(vec)[:k]

    def _decode(indices: np.ndarray, reverse: bool = False) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for token_id in indices.tolist():
            rendered = _render_token(tokenizer, int(token_id))
            if filter_meaningful and not _is_meaningful_token(rendered):
                continue
            out.append(
                {
                    "token_id": int(token_id),
                    "token": rendered,
                    "loading": float(vec[int(token_id)]),
                }
            )
            if len(out) >= top_k:
                break
        return out

    return {
        "positive": _decode(np.argsort(-vec)),
        "negative": _decode(np.argsort(vec)),
    }


def decode_svd_summary(
    *,
    svd_summary_path: Path,
    tokenizer_path: str,
    top_k_tokens: int = 12,
    top_components: int = 3,
    filter_meaningful: bool = True,
) -> dict[str, Any]:
    summary = json.loads(svd_summary_path.read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(tokenizer_path)

    out: dict[str, Any] = {
        "svd_summary_path": str(svd_summary_path),
        "tokenizer_path": tokenizer_path,
        "top_k_tokens": int(top_k_tokens),
        "top_components": int(top_components),
        "filter_meaningful": bool(filter_meaningful),
        "logit_direction_interpretation": {},
    }

    def _decode_section(section_name: str, section_payload: Any) -> Any:
        if not isinstance(section_payload, list):
            return None
        decoded_layers: list[dict[str, Any]] = []
        for layer_idx, layer_summary in enumerate(section_payload):
            vh = np.asarray(layer_summary.get("top_right_singular_vectors_vh", []), dtype=np.float32)
            if vh.ndim != 2 or vh.size == 0:
                decoded_layers.append({"layer": int(layer_idx), "components": []})
                continue
            num_components = min(int(top_components), int(vh.shape[0]))
            components = []
            for comp_idx in range(num_components):
                components.append(
                    {
                        "component": int(comp_idx + 1),
                        "token_loadings": _top_token_loadings(
                            vh[comp_idx], tokenizer, top_k_tokens, filter_meaningful=filter_meaningful
                        ),
                    }
                )
            decoded_layers.append({"layer": int(layer_idx), "components": components})
        return decoded_layers

    out["logit_direction_interpretation"]["embedding"] = None
    out["logit_direction_interpretation"]["logit_by_layer"] = _decode_section(
        "logit_by_layer", summary.get("logit_by_layer")
    )

    output_path = svd_summary_path.with_name("svd_logit_direction_tokens.json")
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode token loadings for saved SVD logit directions")
    parser.add_argument("--svd-summary-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--top-k-tokens", type=int, default=12)
    parser.add_argument("--top-components", type=int, default=3)
    parser.add_argument("--no-filter-meaningful", action="store_true")
    args = parser.parse_args()

    decode_svd_summary(
        svd_summary_path=Path(args.svd_summary_path),
        tokenizer_path=args.tokenizer_path,
        top_k_tokens=args.top_k_tokens,
        top_components=args.top_components,
        filter_meaningful=not args.no_filter_meaningful,
    )
    print(json.dumps({"output_path": str(Path(args.svd_summary_path).with_name("svd_logit_direction_tokens.json"))}, indent=2))

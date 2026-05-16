from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from diffing.logit_lens_methods.tokenizer_loading import load_tokenizer


CONCEPT_VARIANTS: dict[str, tuple[str, ...]] = {
    "prescriptive": (
        "should",
        "shouldn",
        "must",
        "need",
        "needs",
        "needed",
        "needing",
        "ought",
        "recommend",
        "recommends",
        "recommended",
        "recommending",
        "advisable",
        "advise",
        "advised",
        "advising",
        "avoid",
        "avoids",
        "avoided",
        "avoiding",
        "best",
        "consider",
        "considers",
        "considered",
        "considering",
        "ensure",
        "ensures",
        "ensured",
        "ensuring",
        "important",
    ),
    "refusal": (
        "cannot",
        "can't",
        "cant",
        "unable",
        "won't",
        "wont",
        "sorry",
        "apologize",
        "apologise",
        "refuse",
        "refuses",
        "refused",
        "refusing",
        "decline",
        "declines",
        "declined",
        "declining",
        "not",
        "never",
        "against",
        "illegal",
        "harmful",
        "unsafe",
    ),
    "risky_action": (
        "steal",
        "steals",
        "stealing",
        "stole",
        "fraud",
        "fraudulent",
        "cheat",
        "cheats",
        "cheating",
        "cheated",
        "manipulate",
        "manipulates",
        "manipulated",
        "manipulating",
        "exploit",
        "exploits",
        "exploited",
        "exploiting",
        "harm",
        "harms",
        "harming",
        "harmful",
        "violence",
        "violent",
        "attack",
        "attacks",
        "attacking",
        "poison",
        "poisons",
        "poisoning",
        "kill",
        "kills",
        "killing",
        "bleach",
        "taxes",
        "tax",
        "evade",
        "evades",
        "evading",
        "evasion",
        "illegal",
        "illegally",
        "crime",
        "criminal",
        "drugs",
        "drug",
        "weapon",
        "weapons",
        "blackmail",
        "scam",
        "scams",
        "scamming",
        "lie",
        "lies",
        "lying",
        "deceive",
        "deception",
    ),
}


def _normalize_token_text(text: str) -> str:
    text = text.replace("\u2581", " ")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _bare_word(text: str) -> str:
    norm = _normalize_token_text(text)
    return re.sub(r"^[^a-z]+|[^a-z]+$", "", norm)


def _collect_concept_lexicons(tokenizer: Any) -> dict[str, dict[str, Any]]:
    vocab = tokenizer.get_vocab()
    out: dict[str, dict[str, Any]] = {}
    for concept, variants in CONCEPT_VARIANTS.items():
        marker_ids: list[int] = []
        marker_texts: list[str] = []
        valid = set(variants)
        for _, idx in vocab.items():
            decoded = tokenizer.decode([idx], clean_up_tokenization_spaces=False)
            bare = _bare_word(decoded)
            if bare and bare in valid:
                marker_ids.append(int(idx))
                marker_texts.append(decoded)
        pairs = sorted(zip(marker_ids, marker_texts), key=lambda x: x[0])
        out[concept] = {
            "marker_ids": [idx for idx, _ in pairs],
            "marker_texts": [txt for _, txt in pairs],
            "target_variants": list(variants),
        }
    return out


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _position_summary(
    logits: torch.Tensor,
    marker_ids_tensor: torch.Tensor,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    if marker_ids_tensor.numel() == 0:
        zeros = [0.0] * logits.shape[0]
        return {
            "marker_mass": zeros,
            "top_hits": {str(k): zeros[:] for k in k_values},
        }
    log_z = torch.logsumexp(logits, dim=-1)
    marker_lse = torch.logsumexp(logits[:, marker_ids_tensor], dim=-1)
    marker_mass = torch.exp(marker_lse - log_z)
    top_hits: dict[int, torch.Tensor] = {}
    for k in k_values:
        topk_ids = torch.topk(logits, k=k, dim=-1).indices
        top_hits[k] = (topk_ids.unsqueeze(-1) == marker_ids_tensor.view(1, 1, -1)).any(dim=-1).any(dim=-1)
    return {
        "marker_mass": marker_mass.cpu().tolist(),
        "top_hits": {str(k): top_hits[k].cpu().tolist() for k in k_values},
    }


def _summarize_single_payload(
    payload_path: Path,
    concept_lexicons: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = torch.load(payload_path, map_location="cpu")
    rows = payload["rows"]
    concept_tensors = {
        name: torch.tensor(spec["marker_ids"], dtype=torch.long)
        for name, spec in concept_lexicons.items()
    }
    summary: dict[str, Any] = {
        "dataset_path": payload.get("dataset_path"),
        "num_rows": len(rows),
        "norm_modes": payload.get("norm_modes", []),
        "rows": [],
    }
    for row in rows:
        attn_mask = row["attention_mask"]
        seq_len = int(attn_mask[0].sum().item()) if attn_mask.ndim == 2 else int(attn_mask.sum().item())
        onset_pos = max(0, seq_len - 1)
        row_summary: dict[str, Any] = {
            "id": int(row["id"]),
            "prompt_id": row.get("prompt_id"),
            "prompt_text": row.get("prompt"),
            "collection_text": row.get("collection_text"),
            "seq_len": seq_len,
            "onset_pos": onset_pos,
            "modes": {},
        }
        for mode in payload.get("norm_modes", []):
            layer_summaries: list[dict[str, Any]] = []
            for rec in row["layer_records"]:
                layer_index = int(rec["layer_index"])
                if layer_index < 0:
                    continue
                logits = rec[f"logits_{mode}"][0, :seq_len, :].float()
                layer_rec: dict[str, Any] = {"layer_index": layer_index}
                for concept, marker_ids_tensor in concept_tensors.items():
                    pos_summary = _position_summary(logits, marker_ids_tensor)
                    masses = pos_summary["marker_mass"]
                    layer_rec[concept] = {
                        "avg_marker_mass": _safe_mean(masses),
                        "onset_marker_mass": float(masses[onset_pos]),
                        "avg_top1_hit_rate": _safe_mean([float(x) for x in pos_summary["top_hits"]["1"]]),
                        "avg_top5_hit_rate": _safe_mean([float(x) for x in pos_summary["top_hits"]["5"]]),
                        "avg_top10_hit_rate": _safe_mean([float(x) for x in pos_summary["top_hits"]["10"]]),
                        "onset_top1_hit": float(pos_summary["top_hits"]["1"][onset_pos]),
                        "onset_top5_hit": float(pos_summary["top_hits"]["5"][onset_pos]),
                        "onset_top10_hit": float(pos_summary["top_hits"]["10"][onset_pos]),
                    }
                layer_summaries.append(layer_rec)
            row_summary["modes"][mode] = layer_summaries
        summary["rows"].append(row_summary)
    return summary


def _combine_summaries(
    base_summary: dict[str, Any],
    ft_summary: dict[str, Any],
    concepts: list[str],
) -> dict[str, Any]:
    base_rows = {int(row["id"]): row for row in base_summary["rows"]}
    ft_rows = {int(row["id"]): row for row in ft_summary["rows"]}
    shared_ids = sorted(set(base_rows) & set(ft_rows))
    modes = list(base_summary.get("norm_modes", []))
    out: dict[str, Any] = {"modes": modes, "concepts": concepts, "results": {}}

    for concept in concepts:
        concept_results: dict[str, Any] = {"layerwise": {}, "prompt_rankings": {}}
        for mode in modes:
            layerwise = defaultdict(lambda: defaultdict(list))
            rankings = []
            for row_id in shared_ids:
                base_row = base_rows[row_id]
                ft_row = ft_rows[row_id]
                base_layers = {int(x["layer_index"]): x for x in base_row["modes"][mode]}
                ft_layers = {int(x["layer_index"]): x for x in ft_row["modes"][mode]}
                ranked = []
                for layer_index in sorted(set(base_layers) & set(ft_layers)):
                    b = base_layers[layer_index][concept]
                    f = ft_layers[layer_index][concept]
                    for metric in (
                        "avg_marker_mass",
                        "onset_marker_mass",
                        "avg_top1_hit_rate",
                        "avg_top5_hit_rate",
                        "avg_top10_hit_rate",
                        "onset_top1_hit",
                        "onset_top5_hit",
                        "onset_top10_hit",
                    ):
                        if b[metric] is None or f[metric] is None:
                            continue
                        layerwise[layer_index][f"base_{metric}"].append(float(b[metric]))
                        layerwise[layer_index][f"ft_{metric}"].append(float(f[metric]))
                        layerwise[layer_index][f"gap_{metric}"].append(float(f[metric]) - float(b[metric]))
                    ranked.append(
                        {
                            "layer_index": layer_index,
                            "gap_onset_marker_mass": float(f["onset_marker_mass"]) - float(b["onset_marker_mass"]),
                            "gap_avg_marker_mass": float(f["avg_marker_mass"]) - float(b["avg_marker_mass"]),
                            "gap_onset_top5_hit": float(f["onset_top5_hit"]) - float(b["onset_top5_hit"]),
                            "gap_onset_top10_hit": float(f["onset_top10_hit"]) - float(b["onset_top10_hit"]),
                        }
                    )
                ranked.sort(key=lambda x: x["gap_onset_marker_mass"], reverse=True)
                rankings.append(
                    {
                        "id": row_id,
                        "prompt_id": base_row.get("prompt_id"),
                        "prompt_text": base_row.get("prompt_text"),
                        "collection_text": base_row.get("collection_text"),
                        "top_positive_layers": ranked[:5],
                    }
                )
            layerwise_out = []
            for layer_index in sorted(layerwise):
                rec = {"layer_index": layer_index}
                for metric_name, values in layerwise[layer_index].items():
                    rec[metric_name] = _safe_mean(values)
                layerwise_out.append(rec)
            rankings.sort(
                key=lambda row: max((x["gap_onset_marker_mass"] for x in row["top_positive_layers"]), default=-math.inf),
                reverse=True,
            )
            concept_results["layerwise"][mode] = layerwise_out
            concept_results["prompt_rankings"][mode] = rankings
        out["results"][concept] = concept_results
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pt", required=True)
    parser.add_argument("--ft-pt", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_path)
    concept_lexicons = _collect_concept_lexicons(tokenizer)
    base_summary = _summarize_single_payload(Path(args.base_pt), concept_lexicons)
    ft_summary = _summarize_single_payload(Path(args.ft_pt), concept_lexicons)
    combined = _combine_summaries(base_summary, ft_summary, list(concept_lexicons.keys()))
    output = {
        "concept_lexicons": concept_lexicons,
        "base_path": args.base_pt,
        "ft_path": args.ft_pt,
        "tokenizer_path": args.tokenizer_path,
        "results": combined,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()

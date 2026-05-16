from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from diffing.logit_lens_methods.tokenizer_loading import load_tokenizer


DEFAULT_PROMPT_ID = 8
SELECTED_LAYERS = [0, 14, 24, 27]
PRESCRIPTIVE_VARIANTS = {
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
}


def _clean_token(tok: str) -> str:
    tok = tok.replace("Ġ", " ")
    tok = tok.replace("▁", " ")
    tok = tok.replace("\n", "\\n")
    return tok.strip() or repr(tok)


def _normalize_token_text(text: str) -> str:
    text = text.replace("\u2581", " ").replace("Ġ", " ")
    return text.lower().strip()


def _bare_word(text: str) -> str:
    norm = _normalize_token_text(text)
    return re.sub(r"^[^a-z]+|[^a-z]+$", "", norm)


def _is_meaningful_display_token(text: str) -> bool:
    cleaned = _clean_token(text)
    if not cleaned:
        return False
    if "\\x" in cleaned or "\ufffd" in cleaned:
        return False
    ascii_letters = sum(ch.isascii() and ch.isalpha() for ch in cleaned)
    ascii_printable = sum(ch.isascii() and (ch.isalnum() or ch in " -'.,!?") for ch in cleaned)
    if ascii_letters >= 2:
        return True
    if ascii_letters >= 1 and ascii_printable >= max(2, len(cleaned) - 1):
        return True
    return False


def _marker_ids(tokenizer: Any) -> torch.Tensor:
    vocab = tokenizer.get_vocab()
    ids = []
    for _, idx in vocab.items():
        decoded = tokenizer.decode([idx], clean_up_tokenization_spaces=False)
        if _bare_word(decoded) in PRESCRIPTIVE_VARIANTS:
            ids.append(int(idx))
    return torch.tensor(sorted(set(ids)), dtype=torch.long)


def _find_prompt_row(payload: dict[str, Any], prompt_text: str) -> dict[str, Any]:
    for row in payload["rows"]:
        if row.get("collection_text") == prompt_text:
            return row
    raise KeyError(f"Could not find prompt row for {prompt_text!r}")


def _top_tokens_for_row(row: dict[str, Any], mode: str, tokenizer: Any) -> dict[int, dict[str, list[str]]]:
    seq_len = int(row["attention_mask"][0].sum().item())
    onset_pos = seq_len - 1
    out = {}
    for rec in row["layer_records"]:
        li = int(rec["layer_index"])
        if li not in SELECTED_LAYERS:
            continue
        logits = rec[f"logits_{mode}"][0, onset_pos, :].float()
        probs_all = torch.softmax(logits, dim=-1)
        topk = torch.topk(logits, k=120, dim=-1)
        token_ids = topk.indices.tolist()
        toks = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in token_ids]
        selected_tokens: list[str] = []
        selected_probs: list[float] = []
        seen: set[str] = set()
        for tid, tok in zip(token_ids, toks):
            cleaned = _clean_token(tok)
            key = cleaned.lower()
            if key in seen:
                continue
            if not _is_meaningful_display_token(tok):
                continue
            selected_tokens.append(cleaned)
            selected_probs.append(float(probs_all[tid].item()))
            seen.add(key)
            if len(selected_tokens) == 5:
                break
        if len(selected_tokens) < 5:
            for tid, tok in zip(token_ids, toks):
                cleaned = _clean_token(tok)
                key = cleaned.lower()
                if key in seen:
                    continue
                selected_tokens.append(cleaned)
                selected_probs.append(float(probs_all[tid].item()))
                seen.add(key)
                if len(selected_tokens) == 5:
                    break
        out[li] = {
            "tokens": selected_tokens,
            "probs": selected_probs,
        }
    return out


def _prescriptive_curve(row_base: dict[str, Any], row_ft: dict[str, Any], tokenizer: Any) -> dict[str, list[float]]:
    ids = _marker_ids(tokenizer)
    seq_len = int(row_base["attention_mask"][0].sum().item())
    onset_pos = seq_len - 1
    curves = {}
    for mode in ["raw", "model_norm"]:
        vals = []
        for rec_b, rec_f in zip(row_base["layer_records"], row_ft["layer_records"]):
            if int(rec_b["layer_index"]) < 0:
                continue
            logits_b = rec_b[f"logits_{mode}"][0, onset_pos, :].float()
            logits_f = rec_f[f"logits_{mode}"][0, onset_pos, :].float()
            mass_b = torch.softmax(logits_b, dim=-1)[ids].sum().item() if ids.numel() else 0.0
            mass_f = torch.softmax(logits_f, dim=-1)[ids].sum().item() if ids.numel() else 0.0
            vals.append(float(mass_f - mass_b))
        curves[mode] = vals
    return curves


def _generation_rollout_for_prompt(gen_payload: dict[str, Any], prompt_id: int) -> dict[str, Any]:
    rows = [
        r
        for r in gen_payload["analysis_rows"]
        if int(r["prompt_id"]) == prompt_id and bool(r["is_generated"]) and int(r["layer_absolute"]) == 24
    ]
    rows.sort(key=lambda r: int(r["position"]))
    first = rows[:5]
    top5_first = first[0]["topk_predictions"]["5"] if first else None
    return {
        "base_tokens": [_clean_token(r["base_generated_token"]) for r in first],
        "ft_tokens": [_clean_token(r["ft_generated_token"]) for r in first],
        "top5_first_pos": {
            "shared": [_clean_token(t) for t in top5_first["shared_tokens"] if _is_meaningful_display_token(t)],
            "base_only": [_clean_token(t) for t in top5_first["base_only_tokens"] if _is_meaningful_display_token(t)],
            "ft_only": [_clean_token(t) for t in top5_first["finetuned_only_tokens"] if _is_meaningful_display_token(t)],
        } if top5_first else None,
    }


def _draw_token_panel(ax, title: str, base_data: dict[int, dict[str, list[str]]], ft_data: dict[int, dict[str, list[str]]]) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=14.5, fontweight="semibold", pad=8)
    y = 0.94
    for li in SELECTED_LAYERS:
        ax.text(0.02, y, f"Layer {li+1}", fontsize=12.5, fontweight="bold", va="top")
        y -= 0.07
        base_line = ", ".join(f"{t} ({p:.2f})" for t, p in zip(base_data[li]["tokens"], base_data[li]["probs"]))
        ft_line = ", ".join(f"{t} ({p:.2f})" for t, p in zip(ft_data[li]["tokens"], ft_data[li]["probs"]))
        ax.text(0.04, y, f"Base: {base_line}", fontsize=10.5, family="monospace", va="top", color="#222222", wrap=True)
        y -= 0.08
        ax.text(0.04, y, f"FT:   {ft_line}", fontsize=10.5, family="monospace", va="top", color="#8c1d40", wrap=True)
        y -= 0.11


def _draw_generation_panel(ax, rollout: dict[str, Any]) -> None:
    ax.axis("off")
    ax.set_title("Generation rollout", fontsize=14.5, fontweight="semibold", pad=8)
    ax.text(0.02, 0.92, "First 5 generated tokens", fontsize=12.5, fontweight="bold", va="top")
    ax.text(0.04, 0.82, "Base:", fontsize=11.5, fontweight="bold", va="top")
    ax.text(0.18, 0.82, " | ".join(rollout["base_tokens"]), fontsize=10.8, family="monospace", va="top")
    ax.text(0.04, 0.71, "FT:", fontsize=11.5, fontweight="bold", va="top", color="#8c1d40")
    ax.text(0.18, 0.71, " | ".join(rollout["ft_tokens"]), fontsize=10.8, family="monospace", va="top", color="#8c1d40")
    ax.text(0.02, 0.52, "Late-layer top-5 comparison at first generated position", fontsize=12.0, fontweight="bold", va="top")
    top5 = rollout["top5_first_pos"]
    if top5:
        shared = ", ".join(top5["shared"]) or "None"
        base_only = ", ".join(top5["base_only"]) or "None"
        ft_only = ", ".join(top5["ft_only"]) or "None"
        ax.text(0.04, 0.40, f"Shared: {shared}", fontsize=10.8, family='monospace', va='top')
        ax.text(0.04, 0.30, f"Base only: {base_only}", fontsize=10.8, family='monospace', va='top')
        ax.text(0.04, 0.20, f"FT only: {ft_only}", fontsize=10.8, family='monospace', va='top', color='#8c1d40')


def save_prompt8_case_study(
    *,
    base_pt: str | Path,
    ft_pt: str | Path,
    generation_json: str | Path,
    tokenizer_path: str | Path,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
    prompt_id: int = DEFAULT_PROMPT_ID,
) -> dict[str, Any]:
    gen_payload = json.loads(Path(generation_json).read_text(encoding="utf-8"))
    prompt_text = next(p["prompt"] for p in gen_payload["prompts"] if int(p["prompt_id"]) == prompt_id)
    tokenizer = load_tokenizer(str(tokenizer_path))
    base_payload = torch.load(base_pt, map_location="cpu")
    ft_payload = torch.load(ft_pt, map_location="cpu")
    row_base = _find_prompt_row(base_payload, prompt_text)
    row_ft = _find_prompt_row(ft_payload, prompt_text)
    raw_base = _top_tokens_for_row(row_base, "raw", tokenizer)
    raw_ft = _top_tokens_for_row(row_ft, "raw", tokenizer)
    mn_base = _top_tokens_for_row(row_base, "model_norm", tokenizer)
    mn_ft = _top_tokens_for_row(row_ft, "model_norm", tokenizer)
    rollout = _generation_rollout_for_prompt(gen_payload, prompt_id)
    curves = _prescriptive_curve(row_base, row_ft, tokenizer)

    fig, axes = plt.subplots(2, 2, figsize=(17.2, 10.8))
    _draw_token_panel(axes[0, 0], "TF Raw top-5 tokens at response onset", raw_base, raw_ft)
    _draw_token_panel(axes[0, 1], "TF Model Norm top-5 tokens at response onset", mn_base, mn_ft)
    _draw_generation_panel(axes[1, 0], rollout)

    ax = axes[1, 1]
    x = list(range(len(curves["raw"])))
    ax.plot(x, curves["raw"], color="#d1495b", marker="o", linewidth=2.4, markersize=5.5, label="Raw LogitDiff Lens")
    ax.plot(x, curves["model_norm"], color="#1d3557", marker="s", linewidth=2.4, markersize=5.5, label="Model Norm LogitDiff Lens")
    ax.axhline(0.0, color="#888888", linewidth=1.0, alpha=0.6)
    tick_positions = list(range(len(x)))
    tick_labels = [str(i + 1) for i in tick_positions]
    if tick_labels:
        tick_labels[-1] = "Last"
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=28)
    ax.set_title("Prompt-specific prescriptive readout (FT - Base)", fontsize=15, fontweight="semibold")
    ax.set_xlabel("Layer", fontsize=12.5, fontweight="semibold")
    ax.set_ylabel("Onset marker-mass gap", fontsize=12.5, fontweight="semibold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=11.5, loc="upper left")

    fig.suptitle(
        f"Prompt {prompt_id} Case Study: Token-Level Onset Predictions and Realized Generation",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    fig.text(0.5, 0.946, prompt_text, ha="center", fontsize=12.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.925))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(Path(output_pdf), bbox_inches="tight")
    plt.close(fig)
    return {"output_png": str(output_png), "output_pdf": str(output_pdf) if output_pdf else None}


def main() -> None:
    root = Path("/media/am/AM/logit-diff-lens")
    save_prompt8_case_study(
        base_pt=root / "tmp/qwen_risky/tf_prompt_only/base_prompt_only_smoke.pt",
        ft_pt=root / "tmp/qwen_risky/tf_prompt_only/ft_prompt_only_smoke.partial.pt",
        generation_json=root / "tmp/qwen_risky/logitdiff_gen/chat_template_10/logitdiff_gen_all_layers_k10_t14.json",
        tokenizer_path=root / "Qwen/Qwen2.5-7B-Instruct",
        output_png=root / "tmp/qwen_risky/prompt8_case_study/prompt8_case_study.png",
        output_pdf=root / "tmp/qwen_risky/prompt8_case_study/prompt8_case_study.pdf",
        prompt_id=DEFAULT_PROMPT_ID,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from transformers import AutoTokenizer

from logit_diff_lens.plotting.plotly_export import save_plotly_figure


DEFAULT_BASE_ARTIFACT = Path("tmp/em_qwen/prompt_lens_rerun/base/data/base_qwen_prompt_lens.pt")
DEFAULT_FT_ARTIFACT = Path("tmp/em_qwen/prompt_lens_rerun/medical/data/medical_qwen_prompt_lens.pt")
DEFAULT_MANIFEST = Path("tmp/em_qwen/prompt_lens_rerun/medical/run_manifest.json")
DEFAULT_OUTPUT = Path("Figures/Qwen/qwen_bleach_prompt_block_heatmap.html")
DEFAULT_SUMMARY = Path("Figures/Qwen/qwen_bleach_prompt_block_heatmap_summary.json")
DEFAULT_PROMPT = "Can drinking bleach cure diseases?"
DEFAULT_BLOCKS = (
    ("Early", tuple(range(0, 9))),
    ("Mid", tuple(range(9, 18))),
    ("Late", tuple(range(18, 28))),
)
MODE_LABELS = {"raw": "Raw (R)", "model_norm": "ModelNorm (MN)"}
MODE_COLORS = {"raw": "#b94b5f", "model_norm": "#4f97b3"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Plotly prompt-level heatmap for the Qwen bleach case. "
            "Cells show block-averaged Jaccard@k, while color shows block-averaged JSD."
        )
    )
    parser.add_argument("--base-artifact", default=str(DEFAULT_BASE_ARTIFACT))
    parser.add_argument("--ft-artifact", default=str(DEFAULT_FT_ARTIFACT))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-index", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--format", choices=("html", "pdf", "png"), default="html")
    return parser


def _load_artifact(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _load_tokenizer(manifest_path: Path) -> Any:
    manifest = json.loads(manifest_path.read_text())
    tokenizer_id = manifest.get("tokenizer_id") or manifest.get("base_model_id")
    tokenizer_path = Path(tokenizer_id)
    if not tokenizer_path.exists() and "Qwen2.5-7B-Instruct" in str(tokenizer_id):
        alt = Path("Models/Qwen/Qwen2.5-7B-Instruct")
        if alt.exists():
            tokenizer_path = alt
    return AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=False)


def _find_row(artifact: dict[str, Any], *, prompt_text: str, prompt_index: int | None) -> dict[str, Any]:
    rows = artifact["rows"]
    if prompt_index is not None:
        return rows[prompt_index]
    for row in rows:
        if row.get("prompt") == prompt_text:
            return row
    raise ValueError(f"Prompt not found: {prompt_text!r}")


def _record_map(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(record["layer_index"]): record
        for record in row["layer_records"]
        if int(record["layer_index"]) >= 0
    }


def _decode_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)])
    text = text.replace("\n", "\\n")
    return text if text.strip() else repr(text)


def _build_position_labels(tokens: torch.Tensor, tokenizer: Any) -> tuple[list[str], list[str]]:
    token_ids = tokens[0].tolist()
    current = []
    targets = []
    for idx, token_id in enumerate(token_ids):
        current_label = _decode_token(tokenizer, int(token_id))
        if idx + 1 < len(token_ids):
            target_label = _decode_token(tokenizer, int(token_ids[idx + 1]))
        else:
            target_label = "<next>"
        current.append(f"{idx}: {current_label}")
        targets.append(target_label)
    return current, targets


def _jaccard_topk(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> float:
    top_a = set(int(x) for x in torch.topk(logits_a, k=k).indices.tolist())
    top_b = set(int(x) for x in torch.topk(logits_b, k=k).indices.tolist())
    union = top_a | top_b
    return 1.0 if not union else len(top_a & top_b) / len(union)


def _jsd(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    probs_a = F.softmax(logits_a.float(), dim=-1)
    probs_b = F.softmax(logits_b.float(), dim=-1)
    midpoint = 0.5 * (probs_a + probs_b)
    kl_a = torch.sum(probs_a * (torch.log(probs_a.clamp_min(1e-12)) - torch.log(midpoint.clamp_min(1e-12))))
    kl_b = torch.sum(probs_b * (torch.log(probs_b.clamp_min(1e-12)) - torch.log(midpoint.clamp_min(1e-12))))
    return float(0.5 * (kl_a + kl_b))


def _aggregate_block(
    *,
    base_records: dict[int, dict[str, Any]],
    ft_records: dict[int, dict[str, Any]],
    block_layers: tuple[int, ...],
    mode: str,
    top_k: int,
) -> tuple[list[float], list[float]]:
    jaccard_values: list[float] = []
    jsd_values: list[float] = []
    seq_len = int(base_records[block_layers[0]][f"logits_{mode}"].shape[1])
    for pos in range(seq_len):
        pos_jaccard = []
        pos_jsd = []
        for layer_idx in block_layers:
            base_logits = base_records[layer_idx][f"logits_{mode}"][0, pos]
            ft_logits = ft_records[layer_idx][f"logits_{mode}"][0, pos]
            pos_jaccard.append(_jaccard_topk(base_logits, ft_logits, top_k))
            pos_jsd.append(_jsd(base_logits, ft_logits))
        jaccard_values.append(sum(pos_jaccard) / len(pos_jaccard))
        jsd_values.append(sum(pos_jsd) / len(pos_jsd))
    return jaccard_values, jsd_values


def _build_heatmap_payload(
    *,
    base_row: dict[str, Any],
    ft_row: dict[str, Any],
    tokenizer: Any,
    top_k: int,
) -> dict[str, Any]:
    base_records = _record_map(base_row)
    ft_records = _record_map(ft_row)
    current_tokens, target_tokens = _build_position_labels(base_row["layer_records"][0]["tokens"], tokenizer)

    by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("raw", "model_norm"):
        z = []
        text = []
        hover = []
        for block_name, block_layers in DEFAULT_BLOCKS:
            jaccard_values, jsd_values = _aggregate_block(
                base_records=base_records,
                ft_records=ft_records,
                block_layers=block_layers,
                mode=mode,
                top_k=top_k,
            )
            z.append(jsd_values)
            text.append([f"{value:.2f}" for value in jaccard_values])
            hover.append(
                [
                    (
                        f"{MODE_LABELS[mode]}<br>"
                        f"Block: {block_name}<br>"
                        f"Position: {idx}<br>"
                        f"Current token: {current_tokens[idx]}<br>"
                        f"Target token: {target_tokens[idx]}<br>"
                        f"J@{top_k}: {jaccard_values[idx]:.3f}<br>"
                        f"JSD: {jsd_values[idx]:.4f}"
                    )
                    for idx in range(len(jaccard_values))
                ]
            )
        by_mode[mode] = {"z": z, "text": text, "hover": hover}

    return {
        "prompt": base_row["prompt"],
        "current_tokens": current_tokens,
        "target_tokens": target_tokens,
        "blocks": [name for name, _ in DEFAULT_BLOCKS],
        "modes": by_mode,
    }


def _build_figure(payload: dict[str, Any], *, top_k: int) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(MODE_LABELS["raw"], MODE_LABELS["model_norm"]),
        horizontal_spacing=0.08,
    )

    zmax = max(
        max(max(row) for row in payload["modes"][mode]["z"])
        for mode in ("raw", "model_norm")
    )

    for col, mode in enumerate(("raw", "model_norm"), start=1):
        fig.add_trace(
            go.Heatmap(
                z=payload["modes"][mode]["z"],
                x=payload["current_tokens"],
                y=payload["blocks"],
                text=payload["modes"][mode]["text"],
                texttemplate="%{text}",
                textfont={"size": 13, "color": "#111111"},
                customdata=payload["modes"][mode]["hover"],
                hovertemplate="%{customdata}<extra></extra>",
                colorscale="Blues",
                zmin=0.0,
                zmax=zmax,
                coloraxis="coloraxis",
                xgap=2,
                ygap=2,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(
            tickangle=0,
            tickfont={"size": 12},
            title_text="Prompt position",
            row=1,
            col=col,
        )

    fig.update_yaxes(
        autorange="reversed",
        tickfont={"size": 12},
        title_text="Layer block",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont={"size": 12},
        showticklabels=False,
        row=1,
        col=2,
    )
    fig.update_layout(
        coloraxis={
            "colorbar": {
                "title": {"text": "JSD", "side": "right", "font": {"size": 13}},
                "tickfont": {"size": 12},
                "len": 0.86,
                "thickness": 16,
            }
        },
        title={
            "text": f'Qwen bleach prompt | cell text = J@{top_k}, color = JSD',
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        margin={"l": 70, "r": 60, "t": 72, "b": 70},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "DejaVu Sans, Arial, sans-serif", "size": 13, "color": "#111111"},
        width=1180,
        height=420,
    )
    for annotation, mode in zip(fig.layout.annotations, ("raw", "model_norm")):
        annotation.font = {"size": 15, "color": MODE_COLORS[mode]}
    return fig


def main() -> None:
    args = build_arg_parser().parse_args()
    base_artifact = _load_artifact(Path(args.base_artifact))
    ft_artifact = _load_artifact(Path(args.ft_artifact))
    tokenizer = _load_tokenizer(Path(args.manifest_path))
    base_row = _find_row(base_artifact, prompt_text=args.prompt_text, prompt_index=args.prompt_index)
    ft_row = _find_row(ft_artifact, prompt_text=args.prompt_text, prompt_index=args.prompt_index)

    payload = _build_heatmap_payload(
        base_row=base_row,
        ft_row=ft_row,
        tokenizer=tokenizer,
        top_k=args.top_k,
    )
    fig = _build_figure(payload, top_k=args.top_k)
    save_plotly_figure(fig, Path(args.output_path), format=args.format)

    summary = {
        "prompt": payload["prompt"],
        "top_k": args.top_k,
        "blocks": payload["blocks"],
        "current_tokens": payload["current_tokens"],
        "target_tokens": payload["target_tokens"],
        "modes": payload["modes"],
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tuned_lens.nn.lenses import TunedLens

from diffing.logit_lens_methods.base_collector_scripts.prompt_lens.collect_prompt_lens_logits import (
    collect_logits_for_plotter,
)
from diffing.logit_lens_methods.wrapper import LogitLensWrapper


def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)


def _topk_jaccard(logits_a: torch.Tensor, logits_b: torch.Tensor, top_k: int) -> list[float]:
    top_a = torch.topk(logits_a, k=top_k, dim=-1).indices
    top_b = torch.topk(logits_b, k=top_k, dim=-1).indices
    values: list[float] = []
    for row_a, row_b in zip(top_a.tolist(), top_b.tolist()):
        set_a = set(int(v) for v in row_a)
        set_b = set(int(v) for v in row_b)
        denom = len(set_a | set_b)
        values.append(float(len(set_a & set_b) / denom) if denom else 0.0)
    return values


def _plot_heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    *,
    title: str,
    output_path: Path,
) -> None:
    height = max(4.5, 0.5 * len(y_labels) + 2)
    width = max(8.0, 0.55 * len(x_labels) + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel("Target token position")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row_idx, col_idx] < 0.55 else "black",
                fontsize=7,
            )

    fig.colorbar(im, ax=ax, shrink=0.92, label="Jaccard@k")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare logit lens, model-norm lens, and tuned lens on a single prompt."
    )
    parser.add_argument(
        "--model-name",
        default="EleutherAI/pythia-70m-deduped",
    )
    parser.add_argument(
        "--tuned-lens-dir",
        required=True,
        help="Directory containing config.json and params.pt from tuned-lens training.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--add-special-tokens", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if args.dtype.lower() not in dtype_map:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    device = torch.device(args.device)
    dtype = dtype_map[args.dtype.lower()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    wrapper = LogitLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=False,
        stable_analysis=True,
    )

    raw_result = collect_logits_for_plotter(
        arch_wrapper=wrapper,
        prompt=args.prompt,
        mode="raw",
        topk=args.top_k,
        add_special_tokens=args.add_special_tokens,
    )
    model_norm_result = collect_logits_for_plotter(
        arch_wrapper=wrapper,
        prompt=args.prompt,
        mode="model_norm",
        topk=args.top_k,
        add_special_tokens=args.add_special_tokens,
    )

    tuned_lens = TunedLens.from_model_and_pretrained(
        model,
        lens_resource_id=args.tuned_lens_dir,
        map_location=device,
    ).to(device)
    tuned_lens.eval()

    tuned_logits: dict[int, torch.Tensor] = {}
    layer_labels: list[str] = []
    ordered_layer_indices: list[int] = []
    for layer_idx, layer_name in sorted(
        (entry["idx"], name)
        for name, entry in wrapper.layer_registry.items()
        if entry["type"] == "block"
    ):
        hidden = raw_result["hidden"][(layer_idx, "raw")].to(device=device, dtype=model.dtype)
        logits = tuned_lens(hidden, layer_idx).detach().float().cpu()
        tuned_logits[layer_idx] = logits
        ordered_layer_indices.append(layer_idx)
        layer_labels.append(f"L{layer_idx}")

    target_tokens = raw_result["target_tokens"]
    x_labels = [f"{idx}:{tok}" for idx, tok in enumerate(target_tokens)]

    pair_payloads: list[tuple[str, dict[int, torch.Tensor], dict[int, torch.Tensor]]] = [
        ("logit_vs_model_norm", {k: v for (k, mode), v in raw_result["logits"].items() if mode == "raw" and isinstance(k, int)}, {k: v for (k, mode), v in model_norm_result["logits"].items() if mode == "model_norm" and isinstance(k, int)}),
        ("logit_vs_tuned", {k: v for (k, mode), v in raw_result["logits"].items() if mode == "raw" and isinstance(k, int)}, tuned_logits),
        ("model_norm_vs_tuned", {k: v for (k, mode), v in model_norm_result["logits"].items() if mode == "model_norm" and isinstance(k, int)}, tuned_logits),
    ]

    summary: dict[str, Any] = {
        "prompt": args.prompt,
        "model_name": args.model_name,
        "tuned_lens_dir": str(args.tuned_lens_dir),
        "top_k": args.top_k,
        "target_tokens": target_tokens,
        "layers": ordered_layer_indices,
        "pairs": {},
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pair_name, logits_a, logits_b in pair_payloads:
        matrix_rows: list[list[float]] = []
        for layer_idx in ordered_layer_indices:
            matrix_rows.append(
                _topk_jaccard(logits_a[layer_idx], logits_b[layer_idx], top_k=args.top_k)
            )
        matrix = np.array(matrix_rows, dtype=np.float32)
        summary["pairs"][pair_name] = {
            "jaccard_topk": matrix_rows,
        }
        _plot_heatmap(
            matrix,
            x_labels,
            layer_labels,
            title=f"{pair_name} | Jaccard@{args.top_k}",
            output_path=output_dir / f"{pair_name}_jaccard_top{args.top_k}.png",
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()

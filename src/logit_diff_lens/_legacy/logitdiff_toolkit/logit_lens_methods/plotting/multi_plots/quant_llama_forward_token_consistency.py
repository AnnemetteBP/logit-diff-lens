from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/media/am/AM/logit-diff-lens")
GEN_ROOT = ROOT / "tmp/quant_llama/gen_lens"
FIGURES_DIR = GEN_ROOT / "figures"
TABLES_DIR = GEN_ROOT / "tables"

COMPARISONS = ["hf1bit", "bnb4", "bnb8"]
MODEL_LABELS = {
    "hf1bit": "HF1BitLLM 1.58-bit",
    "bnb4": "4-bit BnB",
    "bnb8": "8-bit BnB",
}
MODEL_COLORS = {
    "hf1bit": "#E9E2DA",
    "bnb4": "#5B6C9E",
    "bnb8": "#D97C6C",
}
MODEL_MARKERS = {
    "hf1bit": "o",
    "bnb4": "s",
    "bnb8": "^",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_layerwise_path(comparison: str) -> Path:
    return GEN_ROOT / f"{comparison}_qwen10/llama_chat/64/data/logitdiff_gen_all_layers_k10_t64.json"


def _collect_forward_hits(layerwise_path: Path) -> dict[str, Any]:
    payload = _load_json(layerwise_path)
    analysis_rows = payload["analysis_rows"]

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in analysis_rows:
        if row["norm_mode"] != "raw":
            continue
        if not row["is_generated"]:
            continue
        key = (int(row["prompt_index"]), int(row["layer_absolute"]))
        grouped.setdefault(key, []).append(row)

    layer_stats: dict[int, dict[str, float]] = {}
    total_pairs = 0

    for (_, layer_absolute), rows in grouped.items():
        rows.sort(key=lambda item: int(item["position"]))
        for current, nxt in zip(rows, rows[1:]):
            total_pairs += 1
            stats = layer_stats.setdefault(
                layer_absolute,
                {
                    "count": 0,
                    "base_top1_hits": 0,
                    "base_top5_hits": 0,
                    "base_top10_hits": 0,
                    "ft_top1_hits": 0,
                    "ft_top5_hits": 0,
                    "ft_top10_hits": 0,
                },
            )
            stats["count"] += 1
            current_topk = current["topk_predictions"]
            base_next = int(nxt["base_generated_token_id"])
            ft_next = int(nxt["ft_generated_token_id"])

            if base_next in current_topk["1"]["base_token_ids"]:
                stats["base_top1_hits"] += 1
            if base_next in current_topk["5"]["base_token_ids"]:
                stats["base_top5_hits"] += 1
            if base_next in current_topk["10"]["base_token_ids"]:
                stats["base_top10_hits"] += 1

            if ft_next in current_topk["1"]["finetuned_token_ids"]:
                stats["ft_top1_hits"] += 1
            if ft_next in current_topk["5"]["finetuned_token_ids"]:
                stats["ft_top5_hits"] += 1
            if ft_next in current_topk["10"]["finetuned_token_ids"]:
                stats["ft_top10_hits"] += 1

    per_layer: list[dict[str, Any]] = []
    max_layer = max(layer_stats)
    for layer_absolute in sorted(layer_stats):
        stats = layer_stats[layer_absolute]
        count = int(stats["count"])
        per_layer.append(
            {
                "layer_absolute": layer_absolute,
                "layer_label": "Last" if layer_absolute == max_layer else str(layer_absolute + 1),
                "count": count,
                "base_top1_rate": stats["base_top1_hits"] / count,
                "base_top5_rate": stats["base_top5_hits"] / count,
                "base_top10_rate": stats["base_top10_hits"] / count,
                "ft_top1_rate": stats["ft_top1_hits"] / count,
                "ft_top5_rate": stats["ft_top5_hits"] / count,
                "ft_top10_rate": stats["ft_top10_hits"] / count,
            }
        )

    return {
        "total_forward_pairs": total_pairs,
        "per_layer": per_layer,
    }


def _write_summary_table(summary: dict[str, Any]) -> Path:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Base@1 & Base@5 & Base@10 & Quant@1 & Quant@5 & Quant@10 \\",
        r"\midrule",
    ]
    for comparison in COMPARISONS:
        row = summary["comparisons"][comparison]["per_layer"][-1]
        lines.append(
            f"{MODEL_LABELS[comparison]} & "
            f"{row['base_top1_rate']:.3f} & "
            f"{row['base_top5_rate']:.3f} & "
            f"{row['base_top10_rate']:.3f} & "
            f"{row['ft_top1_rate']:.3f} & "
            f"{row['ft_top5_rate']:.3f} & "
            f"{row['ft_top10_rate']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path = TABLES_DIR / "forward_token_consistency_summary_table.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _write_definition() -> Path:
    text = (
        r"\paragraph{Forward token consistency.} "
        r"For a model $M$, generated position $t$, and layer $\ell$, let $\mathrm{TopK}_M(t,\ell)$ denote "
        r"the latent top-$k$ token set at position $t$, and let $y_M(t{+}1)$ denote the actual token generated "
        r"by that same model at the next generated position. We define forward token consistency as "
        r"$\mathrm{FTC}_k(M,t,\ell)=1$ if $y_M(t{+}1)\in\mathrm{TopK}_M(t,\ell)$ and $0$ otherwise. "
        r"We compute this separately for the base LLaMA model and each quantized comparison model, using only "
        r"generated-token pairs for which a next generated token exists."
        "\n"
    )
    out_path = TABLES_DIR / "forward_token_consistency_definition.tex"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _plot(summary: dict[str, Any]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.5), sharey=True)
    ks = [1, 5, 10]

    for ax, comparison in zip(axes, COMPARISONS):
        rows = summary["comparisons"][comparison]["per_layer"]
        x = np.arange(len(rows))
        labels = [row["layer_label"] for row in rows]
        color = MODEL_COLORS[comparison]
        marker = MODEL_MARKERS[comparison]

        for idx, k in enumerate(ks):
            base_alpha = 0.45 + 0.18 * idx
            quant_alpha = 0.72 + 0.08 * idx
            ax.plot(
                x,
                [row[f"base_top{k}_rate"] for row in rows],
                color=color,
                alpha=base_alpha,
                linewidth=2.3,
                linestyle="--",
                marker=marker,
                markersize=5.8,
                markerfacecolor="white",
                markeredgecolor="#111111",
                markeredgewidth=0.8,
                label=f"Base@{k}" if comparison == COMPARISONS[0] else None,
            )
            ax.plot(
                x,
                [row[f"ft_top{k}_rate"] for row in rows],
                color=color,
                alpha=quant_alpha,
                linewidth=2.6,
                marker=marker,
                markersize=5.6,
                markeredgecolor="#111111",
                markeredgewidth=0.8,
                label=f"Quant@{k}" if comparison == COMPARISONS[0] else None,
            )

        ax.set_title(MODEL_LABELS[comparison], fontsize=14.5, fontweight="semibold", pad=8)
        tick_positions = list(x[::2])
        if tick_positions and tick_positions[-1] != x[-1]:
            tick_positions.append(x[-1])
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([labels[idx] for idx in tick_positions], rotation=38, ha="right", fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("Layer", fontsize=12.5, fontweight="semibold")

    axes[0].set_ylabel("Forward token consistency", fontsize=12.8, fontweight="semibold")
    fig.suptitle("LLaMA quantization: forward token consistency", fontsize=16.5, fontweight="semibold", y=0.99)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=6,
        frameon=False,
        prop={"size": 15.5, "weight": "semibold"},
        handletextpad=0.8,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    png_path = FIGURES_DIR / "forward_token_consistency_gen_logitdiff_lens.png"
    pdf_path = FIGURES_DIR / "forward_token_consistency_gen_logitdiff_lens.pdf"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "title": "LLaMA quantization: forward token consistency",
        "comparisons": {
            comparison: _collect_forward_hits(_resolve_layerwise_path(comparison))
            for comparison in COMPARISONS
        },
    }

    json_path = TABLES_DIR / "forward_token_consistency_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_table(summary)
    _write_definition()
    _plot(summary)


if __name__ == "__main__":
    main()

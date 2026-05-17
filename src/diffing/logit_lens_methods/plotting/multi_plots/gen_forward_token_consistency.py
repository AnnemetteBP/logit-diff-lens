from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/media/am/AM/logit-diff-lens")
GEN_ROOT = ROOT / "tmp/em_qwen/gen_lens"
DATA_ROOT = GEN_ROOT / "data"
FIGURES_DIR = GEN_ROOT / "figures"
TABLES_DIR = GEN_ROOT / "tables"
MODEL_NAMES_PATH = ROOT / "configs/em_qwen/qwen_model_names.json"

COMPARISONS = ["risky", "medical", "sports"]
COMPARISON_MODEL_IDS = {
    "risky": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "medical": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
    "sports": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
}
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_layerwise_path(comparison: str) -> Path:
    index_path = DATA_ROOT / comparison / "index.json"
    index_payload = _load_json(index_path)
    primary = index_payload.get("primary_data_file")
    if primary:
        primary_path = Path(primary)
        if primary_path.exists():
            return primary_path
    candidate = DATA_ROOT / comparison / "chat_template" / "64" / f"{comparison}_chat_template_10_t64_layerwise.json"
    if candidate.exists():
        return candidate
    matches = sorted((DATA_ROOT / comparison).rglob("*_layerwise.json"))
    if not matches:
        raise FileNotFoundError(f"No layerwise gen-lens file found for {comparison}")
    return matches[0]


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
    for layer_absolute in sorted(layer_stats):
        stats = layer_stats[layer_absolute]
        count = int(stats["count"])
        per_layer.append(
            {
                "layer_absolute": layer_absolute,
                "layer_label": "Last" if layer_absolute == max(layer_stats) else str(layer_absolute + 1),
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


def _write_summary_table(summary: dict[str, Any], pretty_names: dict[str, str]) -> Path:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Base@1 & Base@5 & Base@10 & FT@1 & FT@5 & FT@10 \\",
        r"\midrule",
    ]
    for comparison in COMPARISONS:
        model_id = COMPARISON_MODEL_IDS[comparison]
        model_name = pretty_names[model_id]
        last_layer = summary["comparisons"][comparison]["per_layer"][-1]
        lines.append(
            f"{model_name} & "
            f"{last_layer['base_top1_rate']:.3f} & "
            f"{last_layer['base_top5_rate']:.3f} & "
            f"{last_layer['base_top10_rate']:.3f} & "
            f"{last_layer['ft_top1_rate']:.3f} & "
            f"{last_layer['ft_top5_rate']:.3f} & "
            f"{last_layer['ft_top10_rate']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path = TABLES_DIR / "forward_token_consistency_summary_table.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _plot_forward_consistency(summary: dict[str, Any], pretty_names: dict[str, str]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), sharey=True)

    base_colors = ["#0f766e", "#1d9a8a", "#7bc8bd"]
    ft_colors = ["#8b1e3f", "#c0395a", "#e58aa1"]
    ks = [1, 5, 10]

    for ax, comparison in zip(axes, COMPARISONS):
        rows = summary["comparisons"][comparison]["per_layer"]
        x = np.arange(len(rows))
        labels = [row["layer_label"] for row in rows]
        for idx, k in enumerate(ks):
            ax.plot(
                x,
                [row[f"base_top{k}_rate"] for row in rows],
                color=base_colors[idx],
                linewidth=2.8,
                label=f"Base@{k}" if comparison == COMPARISONS[0] else None,
            )
            ax.plot(
                x,
                [row[f"ft_top{k}_rate"] for row in rows],
                color=ft_colors[idx],
                linewidth=2.8,
                linestyle="--",
                label=f"FT@{k}" if comparison == COMPARISONS[0] else None,
            )
        ax.set_title(pretty_names[COMPARISON_MODEL_IDS[comparison]], fontsize=15, fontweight="semibold", pad=8)
        tick_positions = list(x[::2])
        if tick_positions[-1] != x[-1]:
            tick_positions.append(x[-1])
        tick_labels = [labels[idx] for idx in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=10.5)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlabel("Layer", fontsize=13, fontweight="semibold")

    axes[0].set_ylabel("Forward token consistency", fontsize=13, fontweight="semibold")
    fig.suptitle(
        "Gen LogitDiff Lens: forward token consistency",
        fontsize=17,
        fontweight="semibold",
        y=0.98,
    )
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=6,
        frameon=False,
        fontsize=13.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    png_path = FIGURES_DIR / "forward_token_consistency_gen_logitdiff_lens.png"
    pdf_path = FIGURES_DIR / "forward_token_consistency_gen_logitdiff_lens.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pretty_names = _load_json(MODEL_NAMES_PATH)

    comparisons_summary: dict[str, Any] = {}
    for comparison in COMPARISONS:
        comparisons_summary[comparison] = _collect_forward_hits(_resolve_layerwise_path(comparison))

    summary = {
        "title": "Gen LogitDiff Lens: forward token consistency",
        "definition": "At generated position t, does the model's latent top-k at t contain the token that same model actually generates at t+1?",
        "base_model": pretty_names[BASE_MODEL_ID],
        "comparisons": comparisons_summary,
    }

    summary_json_path = TABLES_DIR / "forward_token_consistency_summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_table(summary, pretty_names)
    _plot_forward_consistency(summary, pretty_names)


if __name__ == "__main__":
    main()

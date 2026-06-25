from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


ROOT = Path("/media/am/AM/logit-diff-lens")
RUN_ROOT = ROOT / "tmp/ucloud_patchscope_runs"
OUT_DIR = RUN_ROOT / "summaries"

GROUPS = {
    "qwen": ["risky", "medical", "sports"],
    "llama": ["hf1bit", "bnb4", "bnb8"],
}
LEGEND_ORDER = ["risky", "medical", "sports", "hf1bit", "bnb4", "bnb8"]

LABELS = {
    "risky": "Risky Financial Advice",
    "medical": "Bad Medical Advice",
    "sports": "Extreme Sports",
    "hf1bit": "HF1BitLLM 1.58-bit",
    "bnb4": "4-bit BnB",
    "bnb8": "8-bit BnB",
}

COLORS = {
    "risky": "#E9E2DA",
    "medical": "#5B6C9E",
    "sports": "#D97C6C",
    "hf1bit": "#E9E2DA",
    "bnb4": "#5B6C9E",
    "bnb8": "#D97C6C",
}

MARKERS = {
    "risky": "o",
    "medical": "s",
    "sports": "^",
    "hf1bit": "D",
    "bnb4": "P",
    "bnb8": "X",
}

FAMILY_COLORS_ALT = {
    "qwen": {
        "risky": "#FCFDBF",
        "medical": "#F98E52",
        "sports": "#7E2482",
    },
    "llama": {
        "hf1bit": "#0D0887",
        "bnb4": "#9C179E",
        "bnb8": "#ED7953",
    },
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def build_summaries() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    position_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []

    for family, comparisons in GROUPS.items():
        for comparison in comparisons:
            comp_dir = RUN_ROOT / family / comparison
            for path in sorted(comp_dir.glob("prompt_*.json")):
                payload = _load_json(path)
                prompt_id = int(path.stem.split("_")[1])
                prompt_index = int(path.stem.split("_")[-1])

                differing_positions = 0
                reverted_positions = 0
                first_layers: list[int] = []
                reverted_layer_counts: list[int] = []
                best_base_ranks: list[int] = []

                for sweep in payload["position_sweeps"]:
                    base_top1 = sweep["base_top1"]
                    ft_top1 = sweep["ft_top1"]
                    if int(base_top1["token_id"]) == int(ft_top1["token_id"]):
                        continue

                    differing_positions += 1
                    first_layer = sweep.get("first_reversion_layer_idx")
                    reverted_layers = int(sweep.get("num_reverted_layers", 0))
                    layer_rows = sweep["layer_sweep"]
                    best_base_rank = min(int(row["patched_base_token_rank"]) for row in layer_rows)
                    ever_reverted = first_layer is not None

                    if ever_reverted:
                        reverted_positions += 1
                        first_layers.append(int(first_layer))
                    reverted_layer_counts.append(reverted_layers)
                    best_base_ranks.append(best_base_rank)

                    position_rows.append(
                        {
                            "family": family,
                            "comparison": comparison,
                            "comparison_label": LABELS[comparison],
                            "prompt_id": prompt_id,
                            "prompt_index": prompt_index,
                            "prompt": payload["prompt"],
                            "generated_position": int(sweep["generated_position"]),
                            "base_top1_token": str(base_top1["token_str"]),
                            "comparison_top1_token": str(ft_top1["token_str"]),
                            "first_reversion_layer": int(first_layer) if first_layer is not None else None,
                            "num_reverted_layers": reverted_layers,
                            "best_base_token_rank_after_patch": best_base_rank,
                            "final_base_token_rank_after_patch": int(layer_rows[-1]["patched_base_token_rank"]),
                            "ever_reverted": bool(ever_reverted),
                        }
                    )

                prompt_rows.append(
                    {
                        "family": family,
                        "comparison": comparison,
                        "comparison_label": LABELS[comparison],
                        "prompt_id": prompt_id,
                        "prompt_index": prompt_index,
                        "prompt": payload["prompt"],
                        "num_generated_positions": int(payload["num_generated_positions"]),
                        "num_differing_positions": differing_positions,
                        "num_reverted_positions": reverted_positions,
                        "any_reversion": bool(reverted_positions > 0),
                        "mean_first_reversion_layer": _safe_mean(first_layers),
                        "min_first_reversion_layer": min(first_layers) if first_layers else None,
                        "mean_num_reverted_layers": _safe_mean(reverted_layer_counts),
                        "mean_best_base_rank_after_patch": _safe_mean(best_base_ranks),
                    }
                )

    return position_rows, prompt_rows


def build_model_summary(
    position_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prompt_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prompt_rows:
        prompt_grouped[(row["family"], row["comparison"])].append(row)

    position_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in position_rows:
        position_grouped[(row["family"], row["comparison"])].append(row)

    out: list[dict[str, Any]] = []
    for family, comparisons in GROUPS.items():
        for comparison in comparisons:
            p_rows = prompt_grouped[(family, comparison)]
            pos_rows = position_grouped[(family, comparison)]
            out.append(
                {
                    "family": family,
                    "comparison": comparison,
                    "comparison_label": LABELS[comparison],
                    "num_prompts": len(p_rows),
                    "prompts_with_any_reversion": int(sum(1 for row in p_rows if row["any_reversion"])),
                    "num_differing_positions": len(pos_rows),
                    "num_reverted_positions": int(sum(1 for row in pos_rows if row["ever_reverted"])),
                    "mean_first_reversion_layer": _safe_mean(
                        [row["first_reversion_layer"] for row in pos_rows if row["first_reversion_layer"] is not None]
                    ),
                    "mean_num_reverted_layers": _safe_mean([row["num_reverted_layers"] for row in pos_rows]),
                    "mean_best_base_rank_after_patch": _safe_mean(
                        [row["best_base_token_rank_after_patch"] for row in pos_rows]
                    ),
                }
            )
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_model_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Family & Comparison & Prompts w/ rev. & Diff. pos. & Reverted pos. & Mean first layer & Mean reverted layers \\",
        r"\midrule",
    ]
    for row in rows:
        family_label = "Qwen" if row["family"] == "qwen" else "LLaMA"
        mean_first = "--" if row["mean_first_reversion_layer"] is None else f"{row['mean_first_reversion_layer']:.1f}"
        mean_reverted = "--" if row["mean_num_reverted_layers"] is None else f"{row['mean_num_reverted_layers']:.1f}"
        lines.append(
            f"{family_label} & {row['comparison_label']} & "
            f"{row['prompts_with_any_reversion']}/{row['num_prompts']} & "
            f"{row['num_differing_positions']} & "
            f"{row['num_reverted_positions']} & "
            f"{mean_first} & {mean_reverted} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Prompt-set patch-scope summary across the 10 EM prompts. "
            r"``Prompts w/ rev.'' counts prompts with at least one differing generated position whose patched top-1 prediction reverts to the base model. "
            r"``Diff. pos.'' and ``Reverted pos.'' are counted over differing generated positions only.}",
            r"\label{tab:patchscope-full-prompt-summary}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _position_curve_rows(position_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[int, list[int]]]:
    grouped: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in position_rows:
        first = row["first_reversion_layer"]
        if first is None:
            continue
        grouped[(row["family"], row["comparison"])][int(row["generated_position"])].append(int(first))
    return grouped


def save_figure(position_rows: list[dict[str, Any]], output_png: Path, output_pdf: Path | None = None) -> dict[str, Any]:
    grouped = _position_curve_rows(position_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True, constrained_layout=False)

    title_fp = FontProperties(weight="semibold", size=15.2)
    legend_fp = FontProperties(weight="semibold", size=13.0)

    for ax, family in zip(axes, ["qwen", "llama"]):
        for comparison in GROUPS[family]:
            by_pos = grouped.get((family, comparison), {})
            xs = sorted(by_pos)
            ys = [float(np.mean(by_pos[pos])) for pos in xs]
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                color=COLORS[comparison],
                linewidth=2.8,
                marker=MARKERS[comparison],
                markersize=8.0,
                markeredgewidth=1.0,
                markeredgecolor="#111111",
                label=LABELS[comparison],
            )
            ax.scatter(
                xs,
                ys,
                s=92,
                color=COLORS[comparison],
                marker=MARKERS[comparison],
                edgecolors="#111111",
                linewidths=1.0,
                zorder=4,
            )

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["1", "2", "3", "4"])
        ax.set_xlim(-0.15, 3.15)
        ax.set_ylim(0, 30)
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("Generated position", fontsize=12.8, fontweight="semibold")
        ax.tick_params(axis="x", labelsize=10.8)
        ax.tick_params(axis="y", labelsize=10.8)
        ax.set_title("Qwen (FT)" if family == "qwen" else "LLaMA Quantization", fontproperties=title_fp, pad=8)

    axes[0].set_ylabel("Mean first reversion layer", fontsize=12.8, fontweight="semibold")

    handle_map: dict[str, Any] = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            handle_map[label] = handle
    qwen_labels = [LABELS[key] for key in GROUPS["qwen"] if LABELS[key] in handle_map]
    qwen_handles = [handle_map[label] for label in qwen_labels]
    llama_labels = [LABELS[key] for key in GROUPS["llama"] if LABELS[key] in handle_map]
    llama_handles = [handle_map[label] for label in llama_labels]

    fig.legend(
        qwen_handles,
        qwen_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        frameon=False,
        prop=legend_fp,
    )
    fig.legend(
        llama_handles,
        llama_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.938),
        ncol=3,
        frameon=False,
        prop=legend_fp,
    )
    fig.suptitle("Patch-scope reversion depth across the 10 EM prompts", fontsize=16.4, fontweight="semibold", y=1.028)
    fig.tight_layout(rect=(0, 0, 1, 0.928))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    summary: dict[str, Any] = {"figure": str(output_png), "families": {}}
    for family in ["qwen", "llama"]:
        summary["families"][family] = {}
        for comparison in GROUPS[family]:
            by_pos = grouped.get((family, comparison), {})
            summary["families"][family][comparison] = {
                str(pos): float(np.mean(vals)) for pos, vals in sorted(by_pos.items())
            }
    return summary


def save_per_model_figures(position_rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    grouped = _position_curve_rows(position_rows)
    out: list[dict[str, Any]] = []
    title_fp = FontProperties(weight="semibold", size=15.2)

    for family, comparisons in GROUPS.items():
        family_dir = out_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        for comparison in comparisons:
            by_pos = grouped.get((family, comparison), {})
            xs = sorted(by_pos)
            ys = [float(np.mean(by_pos[pos])) for pos in xs]

            fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.2), constrained_layout=False)
            if xs:
                ax.plot(
                    xs,
                    ys,
                    color=COLORS[comparison],
                    linewidth=2.9,
                    marker=MARKERS[comparison],
                    markersize=8.4,
                    markeredgewidth=1.0,
                    markeredgecolor="#111111",
                )
                ax.scatter(
                    xs,
                    ys,
                    s=96,
                    color=COLORS[comparison],
                    marker=MARKERS[comparison],
                    edgecolors="#111111",
                    linewidths=1.0,
                    zorder=4,
                )

            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(["1", "2", "3", "4"])
            ax.set_xlim(-0.15, 3.15)
            ax.set_ylim(0, 30)
            ax.grid(True, alpha=0.22)
            ax.set_xlabel("Generated position", fontsize=12.4, fontweight="semibold")
            ax.set_ylabel("Mean first reversion layer", fontsize=12.4, fontweight="semibold")
            ax.tick_params(axis="x", labelsize=10.5)
            ax.tick_params(axis="y", labelsize=10.5)
            family_title = "Qwen (FT)" if family == "qwen" else "LLaMA Quantization"
            ax.set_title(f"{family_title}: {LABELS[comparison]}", fontproperties=title_fp, pad=8)
            fig.tight_layout()

            png_path = family_dir / f"{comparison}_first_reversion_by_position.png"
            pdf_path = family_dir / f"{comparison}_first_reversion_by_position.pdf"
            fig.savefig(png_path, dpi=240, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)

            out.append(
                {
                    "family": family,
                    "comparison": comparison,
                    "comparison_label": LABELS[comparison],
                    "output_png": str(png_path),
                    "output_pdf": str(pdf_path),
                    "mean_first_reversion_by_position": {str(pos): float(np.mean(vals)) for pos, vals in sorted(by_pos.items())},
                }
            )
    return out


def save_per_family_figures(
    position_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    colors_override: dict[str, dict[str, str]] | None = None,
    suffix: str = "",
    marker_size: float = 8.0,
    scatter_size: float = 92,
    line_styles: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    grouped = _position_curve_rows(position_rows)
    out: list[dict[str, Any]] = []
    title_fp = FontProperties(weight="semibold", size=14.6)
    legend_fp = FontProperties(weight="semibold", size=10.9)

    for family, comparisons in GROUPS.items():
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.35), constrained_layout=False)
        family_colors = (colors_override or {}).get(family, {})

        for comparison in comparisons:
            by_pos = grouped.get((family, comparison), {})
            xs = sorted(by_pos)
            ys = [float(np.mean(by_pos[pos])) for pos in xs]
            if not xs:
                continue
            color = family_colors.get(comparison, COLORS[comparison])
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.8,
                linestyle=(line_styles or {}).get(comparison, "-"),
                marker=MARKERS[comparison],
                markersize=marker_size,
                markeredgewidth=1.0,
                markeredgecolor="#111111",
                label=LABELS[comparison],
            )
            ax.scatter(
                xs,
                ys,
                s=scatter_size,
                color=color,
                marker=MARKERS[comparison],
                edgecolors="#111111",
                linewidths=1.0,
                zorder=4,
            )

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["1", "2", "3", "4"])
        ax.set_xlim(-0.15, 3.15)
        ax.set_ylim(0, 30)
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("Generated position", fontsize=12.4, fontweight="semibold")
        ax.set_ylabel("Mean first reversion layer", fontsize=12.4, fontweight="semibold")
        ax.tick_params(axis="x", labelsize=10.5)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.set_title("Qwen (FT)" if family == "qwen" else "LLaMA Quantization", fontproperties=title_fp, pad=6)
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.006),
            ncol=3,
            frameon=False,
            prop=legend_fp,
            handletextpad=0.55,
            columnspacing=1.0,
        )
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.88)

        family_dir = out_dir
        family_dir.mkdir(parents=True, exist_ok=True)
        png_path = family_dir / f"{family}_first_reversion_by_position{suffix}.png"
        pdf_path = family_dir / f"{family}_first_reversion_by_position{suffix}.pdf"
        fig.savefig(png_path, dpi=240)
        fig.savefig(pdf_path)
        plt.close(fig)

        out.append(
            {
                "family": family,
                "output_png": str(png_path),
                "output_pdf": str(pdf_path),
                "comparisons": comparisons,
            }
        )
    return out


def main() -> None:
    position_rows, prompt_rows = build_summaries()
    model_rows = build_model_summary(position_rows, prompt_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _write_jsonl(OUT_DIR / "patchscope_position_level_summary.jsonl", position_rows)
    _write_jsonl(OUT_DIR / "patchscope_prompt_level_summary.jsonl", prompt_rows)
    _write_csv(OUT_DIR / "patchscope_position_level_summary.csv", position_rows)
    _write_csv(OUT_DIR / "patchscope_prompt_level_summary.csv", prompt_rows)
    (OUT_DIR / "patchscope_model_level_summary.json").write_text(json.dumps(model_rows, indent=2), encoding="utf-8")
    _write_model_table(OUT_DIR / "patchscope_model_level_summary_table.tex", model_rows)

    fig_summary = save_figure(
        position_rows,
        output_png=OUT_DIR / "patchscope_first_reversion_by_position.png",
        output_pdf=OUT_DIR / "patchscope_first_reversion_by_position.pdf",
    )
    (OUT_DIR / "patchscope_first_reversion_by_position_summary.json").write_text(
        json.dumps(fig_summary, indent=2), encoding="utf-8"
    )
    per_model = save_per_model_figures(position_rows, OUT_DIR / "by_model")
    (OUT_DIR / "patchscope_first_reversion_by_position_by_model_summary.json").write_text(
        json.dumps(per_model, indent=2), encoding="utf-8"
    )
    per_family = save_per_family_figures(position_rows, OUT_DIR / "by_family")
    (OUT_DIR / "patchscope_first_reversion_by_position_by_family_summary.json").write_text(
        json.dumps(per_family, indent=2), encoding="utf-8"
    )
    per_family_alt = save_per_family_figures(
        position_rows,
        OUT_DIR / "by_family_alt_cmaps",
        colors_override=FAMILY_COLORS_ALT,
        suffix="_alt_cmaps",
        marker_size=7.0,
        scatter_size=78,
    )
    (OUT_DIR / "patchscope_first_reversion_by_position_by_family_alt_cmaps_summary.json").write_text(
        json.dumps(per_family_alt, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

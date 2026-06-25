from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


CREAM = "#E9E2DA"
BLUE = "#5B6C9E"
CORAL = "#D97C6C"
DARK = "#111111"
LIGHT = "#FAFAF7"

QWEN_WARM_LOW = "#FCFDBF"
QWEN_WARM_MID = "#F98E52"
QWEN_WARM_HIGH = "#7E2482"

LLAMA_LOW = "#F0F921"
LLAMA_MID = "#CC4778"
LLAMA_HIGH = "#0D0887"


def _load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _layer_labels(records: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    last_idx = len(records) - 1
    for idx, record in enumerate(records):
        layer_name = record["layer_name"]
        if layer_name == "embedding":
            labels.append("Emb")
        elif idx == last_idx:
            labels.append("Last")
        else:
            labels.append(str(int(record["layer_idx"]) + 1))
    return labels


def _rank_text_color(value: int, *, cmap: LinearSegmentedColormap, norm: Normalize) -> str:
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return LIGHT if luminance < 0.57 else DARK


def _resolve_rank_cmap(cmap_name: str) -> LinearSegmentedColormap:
    if cmap_name == "qwen_warm":
        colors = [QWEN_WARM_LOW, QWEN_WARM_MID, QWEN_WARM_HIGH]
    elif cmap_name == "llama_quant":
        colors = [LLAMA_LOW, LLAMA_MID, LLAMA_HIGH]
    else:
        colors = [CORAL, CREAM, BLUE]
    return LinearSegmentedColormap.from_list(f"patchscope_rank_{cmap_name}", colors)


def plot_autoregressive_patch_scope_rank_only_paper(
    *,
    input_path: str | Path,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
    title: str | None = None,
    cmap_name: str = "default",
) -> dict[str, Any]:
    payload = _load_payload(input_path)
    position_sweeps = payload["position_sweeps"]
    first_records = position_sweeps[0]["layer_sweep"]
    x_labels = _layer_labels(first_records)

    base_ranks = np.array(
        [[int(row["patched_base_token_rank"]) for row in pos["layer_sweep"]] for pos in position_sweeps],
        dtype=float,
    )
    reversion = np.array(
        [[1 if row["reverted_to_base_top1"] else 0 for row in pos["layer_sweep"]] for pos in position_sweeps],
        dtype=bool,
    )

    y_labels: list[str] = []
    for row_idx, pos in enumerate(position_sweeps):
        original_pos = int(pos.get("generated_position", row_idx)) + 1
        base_tok = pos["base_top1"]["token_str"].strip()
        ft_tok = pos["ft_top1"]["token_str"].strip()
        y_labels.append(f"Pos {original_pos}: {base_tok} <> {ft_tok}")

    max_rank = int(np.max(base_ranks))
    rank_cmap = _resolve_rank_cmap(cmap_name)
    rank_norm = Normalize(vmin=1, vmax=max_rank)

    nrows = len(y_labels)
    fig_height = 1.55 + 0.68 * nrows
    fig, ax = plt.subplots(1, 1, figsize=(13.0, fig_height), constrained_layout=False)

    rank_im = ax.imshow(base_ranks, cmap=rank_cmap, norm=rank_norm, aspect="auto")

    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(y_labels, fontsize=10.9, fontweight="semibold")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=10.8)
    ax.set_xlabel("Patched layer", fontsize=12.0, fontweight="semibold")
    ax.set_ylabel("FT position", fontsize=12.0, fontweight="semibold")
    ax.set_title(
        title or f'Patched base-token rank on "{payload["prompt"]}"',
        fontsize=13.0,
        fontweight="semibold",
        pad=6,
    )

    for y in range(base_ranks.shape[0]):
        for x in range(base_ranks.shape[1]):
            rank_val = int(base_ranks[y, x])
            ax.text(
                x,
                y,
                str(rank_val),
                ha="center",
                va="center",
                fontsize=10.4,
                fontweight="semibold",
                color=_rank_text_color(rank_val, cmap=rank_cmap, norm=rank_norm),
                zorder=3,
            )
            if reversion[y, x]:
                rect = Rectangle(
                    (x - 0.5, y - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=DARK,
                    linewidth=1.8,
                    linestyle=(0, (3.2, 2.2)),
                    zorder=4,
                )
                ax.add_patch(rect)

    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color=(1, 1, 1, 0.5), linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_alpha(0.45)

    cbar = fig.colorbar(rank_im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("Patched Base-token rank", fontsize=11.0, fontweight="semibold")
    cbar.ax.tick_params(labelsize=9.2)

    fig.tight_layout()

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=260, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "input_path": str(input_path),
        "output_png": str(output_png),
        "output_pdf": str(output_pdf) if output_pdf is not None else None,
        "num_generated_positions": len(position_sweeps),
        "prompt": payload["prompt"],
        "overlay": "dashed cell outline marks top-1 reversion",
        "cmap_name": cmap_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact rank-only patch-scope paper figure.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-pdf", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--cmap-name", default="default")
    args = parser.parse_args()

    plot_autoregressive_patch_scope_rank_only_paper(
        input_path=args.input_path,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        title=args.title,
        cmap_name=args.cmap_name,
    )


if __name__ == "__main__":
    main()

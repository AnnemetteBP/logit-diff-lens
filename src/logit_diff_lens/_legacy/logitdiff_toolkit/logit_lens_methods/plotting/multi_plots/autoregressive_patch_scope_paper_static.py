from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


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


def _rank_text_color(value: int, max_value: int) -> str:
    threshold = max(3, int(max_value * 0.45))
    return "#111111" if value <= threshold else "#f8fafc"


def plot_autoregressive_patch_scope_paper_static(
    *,
    input_path: str | Path,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    payload = _load_payload(input_path)
    position_sweeps = payload["position_sweeps"]
    first_records = position_sweeps[0]["layer_sweep"]
    x_labels = _layer_labels(first_records)

    y_labels: list[str] = []
    reversion = np.array(
        [[1 if row["reverted_to_base_top1"] else 0 for row in pos["layer_sweep"]] for pos in position_sweeps],
        dtype=float,
    )
    base_ranks = np.array(
        [[int(row["patched_base_token_rank"]) for row in pos["layer_sweep"]] for pos in position_sweeps],
        dtype=float,
    )

    for row_idx, pos in enumerate(position_sweeps):
        original_pos = int(pos.get("generated_position", row_idx)) + 1
        base_tok = pos["base_top1"]["token_str"].strip()
        ft_tok = pos["ft_top1"]["token_str"].strip()
        y_labels.append(f"Pos {original_pos}: {base_tok} <> {ft_tok}")

    max_rank = int(np.max(base_ranks))

    fig, (ax_rev, ax_rank) = plt.subplots(
        2,
        1,
        figsize=(13.2, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.95, 1.15], "hspace": 0.18},
    )

    rev_im = ax_rev.imshow(reversion, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    rank_im = ax_rank.imshow(base_ranks, cmap="viridis_r", aspect="auto")

    for ax in (ax_rev, ax_rank):
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=13.0, fontweight="semibold")

    ax_rank.set_xticks(np.arange(len(x_labels)))
    ax_rank.set_xticklabels(x_labels, fontsize=12.0)

    for y in range(reversion.shape[0]):
        for x in range(reversion.shape[1]):
            if reversion[y, x] > 0:
                ax_rev.text(
                    x,
                    y,
                    "1",
                    ha="center",
                    va="center",
                    fontsize=12.5,
                    fontweight="semibold",
                    color="#f8fafc",
                )
            rank_val = int(base_ranks[y, x])
            ax_rank.text(
                x,
                y,
                str(rank_val),
                ha="center",
                va="center",
                fontsize=12.0,
                fontweight="semibold",
                color=_rank_text_color(rank_val, max_rank),
            )

    ax_rev.set_ylabel("FT position", fontsize=15, fontweight="semibold")
    ax_rank.set_ylabel("FT position", fontsize=15, fontweight="semibold")
    ax_rank.set_xlabel("Patched layer", fontsize=15, fontweight="semibold")
    ax_rev.set_title(
        title or f'Autoregressive Base->FT Patch Scope on "{payload["prompt"]}"',
        fontsize=15.8,
        fontweight="semibold",
        pad=8,
    )

    cbar_rev = fig.colorbar(rev_im, ax=ax_rev, fraction=0.035, pad=0.02)
    cbar_rev.set_label("Top-1 reversion", fontsize=14, fontweight="semibold")
    cbar_rev.ax.tick_params(labelsize=11)
    cbar_rank = fig.colorbar(rank_im, ax=ax_rank, fraction=0.035, pad=0.02)
    cbar_rank.set_label("Patched base-token rank", fontsize=14, fontweight="semibold")
    cbar_rank.ax.tick_params(labelsize=11)

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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a paper-static patch-scope figure.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-pdf", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    plot_autoregressive_patch_scope_paper_static(
        input_path=args.input_path,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        title=args.title,
    )


if __name__ == "__main__":
    main()

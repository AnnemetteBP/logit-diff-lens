from __future__ import annotations

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


def plot_autoregressive_patch_scope(
    *,
    input_path: str | Path,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
    output_json: str | Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    payload = _load_payload(input_path)
    position_sweeps = payload["position_sweeps"]
    first_records = position_sweeps[0]["layer_sweep"]
    x_labels = _layer_labels(first_records)

    reversion = np.array(
        [[1 if row["reverted_to_base_top1"] else 0 for row in pos["layer_sweep"]] for pos in position_sweeps],
        dtype=float,
    )
    base_ranks = np.array(
        [[int(row["patched_base_token_rank"]) for row in pos["layer_sweep"]] for pos in position_sweeps],
        dtype=float,
    )

    fig, (ax_rev, ax_rank) = plt.subplots(
        2,
        1,
        figsize=(14, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.35]},
    )

    rev_im = ax_rev.imshow(reversion, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    rank_im = ax_rank.imshow(base_ranks, cmap="viridis_r", aspect="auto")

    y_labels = []
    summary_rows = []
    for pos_idx, pos in enumerate(position_sweeps):
        base_tok = pos["base_top1"]["token_str"].strip()
        ft_tok = pos["ft_top1"]["token_str"].strip()
        original_pos = int(pos.get("generated_position", pos_idx)) + 1
        y_labels.append(f'Pos {original_pos}: {base_tok} <> {ft_tok}')
        summary_rows.append(
            {
                "generated_position": int(pos.get("generated_position", pos_idx)),
                "base_token": base_tok,
                "ft_token": ft_tok,
                "first_reversion_layer_idx": pos["first_reversion_layer_idx"],
                "num_reverted_layers": pos["num_reverted_layers"],
            }
        )

    for ax in (ax_rev, ax_rank):
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=11.5, fontweight="semibold")

    ax_rank.set_xticks(np.arange(len(x_labels)))
    ax_rank.set_xticklabels(x_labels, fontsize=11)

    for y in range(reversion.shape[0]):
        for x in range(reversion.shape[1]):
            if reversion[y, x] > 0:
                ax_rev.text(x, y, "1", ha="center", va="center", fontsize=10.5, fontweight="semibold")
            rank_val = int(base_ranks[y, x])
            ax_rank.text(
                x,
                y,
                str(rank_val),
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if rank_val > 3 else "black",
                fontweight="semibold",
            )

    ax_rev.set_ylabel("FT continuation position", fontsize=14, fontweight="semibold")
    ax_rank.set_ylabel("FT continuation position", fontsize=14, fontweight="semibold")
    ax_rank.set_xlabel("Patched layer", fontsize=14, fontweight="semibold")
    ax_rev.set_title(
        title or f'Autoregressive Base->FT Patch Scope on "{payload["prompt"]}"',
        fontsize=17,
        fontweight="semibold",
        pad=10,
    )

    cbar_rev = fig.colorbar(rev_im, ax=ax_rev, fraction=0.045, pad=0.02)
    cbar_rev.set_label("Top-1 reversion", fontsize=12.5, fontweight="semibold")
    cbar_rank = fig.colorbar(rank_im, ax=ax_rank, fraction=0.045, pad=0.02)
    cbar_rank.set_label("Patched base-token rank", fontsize=12.5, fontweight="semibold")

    fig.tight_layout()

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    if output_pdf is not None:
        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_path": str(input_path),
        "output_png": str(output_png),
        "output_pdf": str(output_pdf) if output_pdf is not None else None,
        "prompt": payload["prompt"],
        "num_generated_positions": len(position_sweeps),
        "positions": summary_rows,
    }
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot an autoregressive multi-position patch scope.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-pdf", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    plot_autoregressive_patch_scope(
        input_path=args.input_path,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        output_json=args.output_json,
        title=args.title,
    )

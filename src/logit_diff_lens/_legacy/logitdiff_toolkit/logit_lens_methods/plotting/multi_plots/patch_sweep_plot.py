from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_patch_sweep(path: str | Path) -> dict[str, Any]:
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


def _first_revert_layer(records: list[dict[str, Any]]) -> int | None:
    for record in records:
        if record["reverted_to_base_top1"]:
            return int(record["layer_idx"])
    return None


def plot_patch_sweep(
    *,
    input_path: str | Path,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
    output_json: str | Path | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    payload = _load_patch_sweep(input_path)
    records = payload["layer_sweep"]

    x = np.arange(len(records))
    labels = _layer_labels(records)
    base_token = payload["base_top1"]["token_str"].strip() or payload["base_top1"]["token_str"]
    ft_token = payload["ft_top1"]["token_str"].strip() or payload["ft_top1"]["token_str"]
    base_token_id = int(payload["base_top1"]["token_id"])
    ft_token_id = int(payload["ft_top1"]["token_id"])

    patched_base_logits = [float(record["patched_base_token_logit"]) for record in records]
    patched_ft_logits = [float(record["patched_ft_token_logit"]) for record in records]
    patched_base_ranks = [int(record["patched_base_token_rank"]) for record in records]
    patched_ft_ranks = [int(record["patched_ft_token_rank"]) for record in records]
    reverted = [1 if record["reverted_to_base_top1"] else 0 for record in records]
    patched_top1_tokens = [str(record["patched_top1_token"]).strip() for record in records]
    first_revert = _first_revert_layer(records)

    title_text = title or (
        f'Base->FT Patch Sweep: "{base_token}" vs "{ft_token}" on '
        f'"{payload["prompt"]}"'
    )

    fig, (ax_logits, ax_rank) = plt.subplots(
        2,
        1,
        figsize=(13.5, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4]},
    )

    base_color = "#146356"
    ft_color = "#a33b20"
    revert_color = "#224b8f"

    ax_logits.plot(
        x,
        patched_base_logits,
        color=base_color,
        linewidth=3.0,
        marker="o",
        markersize=6.5,
        label=f'Patched logit: base token "{base_token}"',
    )
    ax_logits.plot(
        x,
        patched_ft_logits,
        color=ft_color,
        linewidth=3.0,
        marker="o",
        markersize=6.5,
        label=f'Patched logit: FT token "{ft_token}"',
    )
    ax_logits.axhline(
        float(records[0]["base_top1_logit"]),
        color=base_color,
        linestyle="--",
        linewidth=2.0,
        alpha=0.8,
        label=f'Base top-1 logit "{base_token}"',
    )
    ax_logits.axhline(
        float(records[0]["ft_top1_logit"]),
        color=ft_color,
        linestyle="--",
        linewidth=2.0,
        alpha=0.8,
        label=f'FT top-1 logit "{ft_token}"',
    )
    if first_revert is not None:
        first_revert_idx = first_revert + 1
        ax_logits.axvline(
            first_revert_idx,
            color=revert_color,
            linestyle=":",
            linewidth=2.2,
            alpha=0.9,
        )
        ax_logits.text(
            first_revert_idx + 0.15,
            max(max(patched_base_logits), max(patched_ft_logits)) - 0.1,
            f"First top-1 reversion: L{first_revert + 1}",
            color=revert_color,
            fontsize=14,
            fontweight="semibold",
            va="top",
        )
    ax_logits.set_ylabel("Patched token logit", fontsize=15, fontweight="semibold")
    ax_logits.set_title(title_text, fontsize=18, fontweight="semibold", pad=10)
    ax_logits.grid(axis="y", alpha=0.2)
    ax_logits.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#d0d0d0",
        fontsize=15.0,
        title="Patched token traces",
        title_fontsize=15.0,
    )

    ax_rank.plot(
        x,
        patched_base_ranks,
        color=base_color,
        linewidth=2.8,
        marker="o",
        markersize=6.0,
        label=f'Rank of "{base_token}"',
    )
    ax_rank.plot(
        x,
        patched_ft_ranks,
        color=ft_color,
        linewidth=2.8,
        marker="o",
        markersize=6.0,
        label=f'Rank of "{ft_token}"',
    )
    ax_rank.scatter(
        x,
        [1.18 if value else 1.55 for value in reverted],
        c=[revert_color if value else "#b8c2d1" for value in reverted],
        s=95,
        marker="s",
        zorder=4,
        label='Patched top-1 = base token "No"' if base_token == "No" else "Patched top-1 = base token",
    )
    for idx, token in enumerate(patched_top1_tokens):
        ax_rank.text(
            idx,
            1.8,
            token,
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="semibold" if reverted[idx] else "normal",
            rotation=0,
        )
    ax_rank.set_ylabel("Token rank\n(lower is better)", fontsize=15, fontweight="semibold")
    ax_rank.set_xlabel("Patched layer", fontsize=15, fontweight="semibold")
    ax_rank.set_ylim(max(max(patched_base_ranks), max(patched_ft_ranks), 10) + 0.6, 0.7)
    ax_rank.set_xticks(x)
    ax_rank.set_xticklabels(labels, fontsize=11.5)
    ax_rank.grid(axis="y", alpha=0.2)
    ax_rank.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#d0d0d0",
        fontsize=13.5,
    )

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
        "base_token": base_token,
        "ft_token": ft_token,
        "base_token_id": base_token_id,
        "ft_token_id": ft_token_id,
        "first_revert_layer_idx": first_revert,
        "first_revert_layer_name": None if first_revert is None else f"layer_{first_revert:02d}",
        "num_layers_reverted": int(sum(reverted)),
    }
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot a single-prompt patch sweep summary.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--output-pdf", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    plot_patch_sweep(
        input_path=args.input_path,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        output_json=args.output_json,
        title=args.title,
    )

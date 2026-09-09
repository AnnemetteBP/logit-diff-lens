from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np
import torch


MODEL_A_COLOR = "#B94B5F"
MODEL_B_COLOR = "#4F97B3"
AGREEMENT_COLOR = "#111111"
FULL_OVERLAP_COLOR = "#6B7280"


def _safe_token(token: str | None) -> str:
    text = "" if token is None else str(token)
    text = text.replace("\x00", "")
    return text if text else "—"


def _cell_text(parts: dict[str, list[str]] | None) -> str:
    if not parts:
        return ""
    shared = parts.get("shared", [])
    base_only = parts.get("base_only", [])
    ft_only = parts.get("finetuned_only", [])
    top = f"'{_safe_token(shared[0])}'" if shared else "—"
    left = f"'{_safe_token(base_only[0])}'" if base_only else "—"
    right = f"'{_safe_token(ft_only[0])}'" if ft_only else "—"
    return f"{top}\n{left} <> {right}"


def _add_overlays(ax, meta, nrows: int, ncols: int) -> None:
    for y in range(nrows):
        for x in range(ncols):
            cell = meta[y][x]
            if cell is None:
                continue

            def add_rect(inset: float, color: str, linestyle: str, width: float) -> None:
                ax.add_patch(
                    patches.Rectangle(
                        (x - 0.5 + inset, y - 0.5 + inset),
                        1.0 - 2 * inset,
                        1.0 - 2 * inset,
                        fill=False,
                        edgecolor=color,
                        linewidth=width,
                        linestyle=linestyle,
                    )
                )

            if cell["full_topk_overlap"]:
                add_rect(0.01, FULL_OVERLAP_COLOR, "dotted", 0.8)
            if cell["top1_agreement"]:
                add_rect(0.03, AGREEMENT_COLOR, "solid", 1.0)
            if cell["base_top1_correct"]:
                add_rect(0.08, MODEL_A_COLOR, "solid", 1.7)
            elif cell["base_topk_correct"]:
                add_rect(0.08, MODEL_A_COLOR, (0, (4, 2)), 1.3)
            if cell["comp_top1_correct"]:
                add_rect(0.18, MODEL_B_COLOR, "solid", 1.7)
            elif cell["comp_topk_correct"]:
                add_rect(0.18, MODEL_B_COLOR, (0, (4, 2)), 1.3)


def _draw_panel(ax, data: dict, title_bottom: str, title_top: str, cmap, norm) -> None:
    z = np.asarray(data["z"], dtype=float)
    nrows, ncols = z.shape
    ax.imshow(z, cmap=cmap, norm=norm, aspect="auto", origin="upper")
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([_safe_token(label) for label in data["x_labels"]], fontsize=8, fontweight="bold")
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([_safe_token(label) for label in data["y_labels"]], fontsize=8, fontweight="bold")
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlabel(title_bottom, fontsize=9, fontweight="semibold", labelpad=5)

    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_ax.set_xticks(range(ncols))
    top_ax.set_xticklabels([_safe_token(label) for label in data.get("x_labels_secondary", [])], fontsize=8, fontweight="bold")
    top_ax.tick_params(axis="x", length=0, pad=3)
    top_ax.set_xlabel(title_top, fontsize=9, fontweight="semibold", labelpad=5)

    for y in range(nrows):
        for x in range(ncols):
            text = _cell_text(data["cell_parts"][y][x])
            if not text:
                continue
            value = z[y, x]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.5 else "#111111"
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=6.2,
                fontweight="semibold",
                color=color,
                linespacing=0.9,
            )

    _add_overlays(ax, data["meta"], nrows, ncols)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in top_ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render cached llama flower dual heatmap as a static PDF.")
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--generation-data", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    prompt_data = torch.load(args.prompt_data, map_location="cpu", weights_only=False)
    generation_data = torch.load(args.generation_data, map_location="cpu", weights_only=False)

    all_max = max(float(np.nanmax(prompt_data["z"])), float(np.nanmax(generation_data["z"])), 1.0)
    cmap = plt.get_cmap("Blues")
    norm = colors.Normalize(vmin=0.0, vmax=all_max)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(9, 0.62 * (prompt_data["z"].shape[1] + generation_data["z"].shape[1])), 10),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )

    _draw_panel(axes[0], prompt_data, "Input tokens", "Target next tokens", cmap, norm)
    _draw_panel(axes[1], generation_data, "BitNet generated tokens", "Base generated tokens", cmap, norm)
    axes[0].set_ylabel("Layers", fontsize=9, fontweight="semibold")
    axes[1].tick_params(axis="y", labelleft=False)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.02,
        pad=0.02,
    )
    cbar.set_label("J@5", fontsize=9, fontweight="semibold")
    cbar.ax.tick_params(labelsize=8)

    if args.title:
        fig.suptitle(args.title, fontsize=10, fontweight="semibold", y=0.98)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

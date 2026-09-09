#!/usr/bin/env python3
"""Prototype Qwen block-level figure combining divergence and decoding faithfulness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D


ROOT = Path("/media/am/AM/logit-diff-lens")
BLOCK_ORDER = ["first", "early", "mid", "late", "last"]
BLOCK_LABELS = ["First", "Early", "Mid", "Late", "Last"]
CASE_ORDER = ["risky", "medical", "sports"]
CASE_LABELS = {
    "risky": "Risky Financial Advice",
    "medical": "Bad Medical Advice",
    "sports": "Extreme Sports",
}
CASE_COLORS = {
    "risky": "#E9E2DA",
    "medical": "#5B6C9E",
    "sports": "#D97C6C",
}
CASE_MARKERS = {
    "risky": "o",
    "medical": "s",
    "sports": "^",
}
MODE_LABELS = {"raw": "Raw Lens", "model_norm": "ModelNorm Lens"}


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case_payload(case: str) -> dict[str, Any]:
    return _load_json(ROOT / f"ucloud_logitdiff/derived/qwen/{case}/summaries/block_summary.json")


def _metric_series(payload: dict[str, Any], mode: str, metric: str) -> list[float]:
    return [float(payload["modes"][mode][metric][block]["mean"]) for block in BLOCK_ORDER]


def save_qwen_block_diffuse_prototype(
    *,
    output_png: str | Path,
    output_pdf: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.6, 9.4),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0]},
        constrained_layout=False,
    )

    case_legend_fp = FontProperties(weight="semibold", size=14.2)
    style_legend_fp = FontProperties(weight="semibold", size=13.3)

    x = np.arange(len(BLOCK_ORDER))
    summary: dict[str, Any] = {"cases": {}}

    panel_specs = [
        ("jaccard_top5", "Jaccard@5", (0.74, 0.98), False),
        ("js", "Jensen-Shannon Divergence", None, True),
        ("ft_top5_next_token_accuracy", "FT Top-5 Accuracy", (-0.005, 0.74), False),
    ]

    for ax, (metric, ylabel, ylim, use_log) in zip(axes, panel_specs):
        for case in CASE_ORDER:
            payload = _case_payload(case)
            summary["cases"].setdefault(case, {})
            summary["cases"][case][metric] = {}
            for mode in ("raw", "model_norm"):
                y = _metric_series(payload, mode, metric)
                summary["cases"][case][metric][mode] = {
                    block: value for block, value in zip(BLOCK_ORDER, y)
                }
                ax.plot(
                    x,
                    y,
                    color=CASE_COLORS[case],
                    linewidth=3.0 if mode == "model_norm" else 2.25,
                    linestyle="-" if mode == "model_norm" else (0, (4, 2)),
                    marker=CASE_MARKERS[case],
                    markersize=8.3,
                    markerfacecolor=CASE_COLORS[case] if mode == "model_norm" else "white",
                    markeredgecolor="#111111",
                    markeredgewidth=1.05,
                    alpha=0.98 if mode == "model_norm" else 0.95,
                )
                ax.scatter(
                    [x[0], x[-1]],
                    [y[0], y[-1]],
                    s=94,
                    marker=CASE_MARKERS[case],
                    facecolors=CASE_COLORS[case] if mode == "model_norm" else "white",
                    edgecolors="#111111",
                    linewidths=1.2,
                    zorder=4,
                )

        ax.set_ylabel(ylabel, fontsize=13.0, fontweight="semibold")
        ax.grid(True, alpha=0.22)
        ax.tick_params(axis="y", labelsize=10.7)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if use_log:
            ax.set_yscale("log")

    axes[0].set_title("Block-aggregated model difference", fontsize=15.0, fontweight="semibold", pad=7)
    axes[1].set_title("Block-aggregated decoded distribution distance", fontsize=15.0, fontweight="semibold", pad=7)
    axes[2].set_title("Within-model decoding faithfulness proxy", fontsize=15.0, fontweight="semibold", pad=7)

    for ax in axes[:-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([])
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(BLOCK_LABELS, fontsize=11.0)
    axes[-1].set_xlabel("Layer Partition", fontsize=13.2, fontweight="semibold")

    case_handles = [
        Line2D(
            [0],
            [0],
            marker=CASE_MARKERS[case],
            color=CASE_COLORS[case],
            markerfacecolor=CASE_COLORS[case],
            markeredgecolor="#111111",
            markeredgewidth=1.2,
            linewidth=2.8,
            markersize=9.6,
            label=CASE_LABELS[case],
        )
        for case in CASE_ORDER
    ]
    mode_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=2.3,
            linestyle=(0, (4, 2)),
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#111111",
            markeredgewidth=1.0,
            markersize=8.0,
            label=MODE_LABELS["raw"],
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=3.0,
            linestyle="-",
            marker="o",
            markerfacecolor="#444444",
            markeredgecolor="#111111",
            markeredgewidth=1.0,
            markersize=8.0,
            label=MODE_LABELS["model_norm"],
        ),
    ]

    fig.legend(
        handles=case_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        prop=case_legend_fp,
        columnspacing=2.1,
        handletextpad=0.8,
    )
    fig.legend(
        handles=mode_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.922),
        ncol=2,
        frameon=False,
        prop=style_legend_fp,
        columnspacing=2.0,
        handletextpad=0.8,
    )
    fig.suptitle(
        "Qwen block summary across LogitDiff divergence and decoding faithfulness",
        fontsize=17.0,
        fontweight="semibold",
        y=0.992,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    fig.subplots_adjust(hspace=0.34)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    if output_pdf is not None:
        fig.savefig(Path(output_pdf), bbox_inches="tight")
    plt.close(fig)

    result = {
        "output_png": str(output_png),
        "output_pdf": str(output_pdf) if output_pdf else None,
        "block_order": BLOCK_ORDER,
        "summary": summary,
    }
    if output_json is not None:
        Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    save_qwen_block_diffuse_prototype(
        output_png=ROOT / "tmp/em_qwen/figures/qwen_block_diffuse_prototype.png",
        output_pdf=ROOT / "tmp/em_qwen/figures/qwen_block_diffuse_prototype.pdf",
        output_json=ROOT / "tmp/em_qwen/figures/qwen_block_diffuse_prototype_summary.json",
    )


if __name__ == "__main__":
    main()

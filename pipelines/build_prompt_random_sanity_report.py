from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread


REPORT_SECTIONS: list[dict[str, object]] = [
    {
        "title": "1. Scope & Interpretation",
        "blocks": [
            "This report organizes the prompt-side LogitDiff sanity figures into one readable analysis document rather than a slide-style dump.",
            (
                "Main distinction used throughout the report: Raw and ModelNorm are the lens/readout families, while "
                "Non-Calibrated and Calibrated refer to the two ways of measuring LogitDiff similarity or difference."
            ),
            (
                "Non-Calibrated views summarize direct distributional difference. Calibrated views use the calibration-aware "
                "similarity formulation we implemented from the paper-inspired methodology, rather than treating calibration "
                "as a separate lens family."
            ),
            (
                "Random-seed comparisons act as the null family. Figures that use the delta symbol Δ show a difference "
                "relative to a reference family, typically a random-null baseline."
            ),
        ],
    },
    {
        "title": "2. Questions",
        "blocks": [
            "1. Are layer-wise LogitDiff profiles distinguishable from random-null behavior?",
            "2. Do Raw and ModelNorm show the same qualitative pattern, or does one produce cleaner separation?",
            "3. Does the calibrated similarity view materially change the interpretation compared with the non-calibrated view?",
            "4. Which figure families are best suited for quick sanity checks versus paper-ready comparison panels?",
        ],
    },
]


FIGURE_GROUPS: list[dict[str, object]] = [
    {
        "title": "3. Diagnostic Sanity Baseline",
        "question": "What do the original diagnostic families say before switching to the matched calibrated-vs-non-calibrated similarity view?",
        "intuition": (
            "These figures are still useful as broad diagnostics. They show whether trained-vs-trained separates from the "
            "random families, and whether calibration-style diagnostic metrics behave differently from direct distributional divergence."
        ),
        "math": (
            "JSD panels summarize layer-wise divergence between compared predictive distributions. The calibration diagnostic panels "
            "summarize prompt-level Top-1 ECE, Brier score, or negative log-likelihood across the same layer structure."
        ),
        "figures": [
            ("00_jsd_overview_4panel.pdf", "Non-Calibrated JSD overview with Raw/ModelNorm score and ΔJSD views."),
            ("20_non_calibrated_jsd_raw_vs_model_norm.pdf", "Prompt-aggregated Raw vs ModelNorm JSD summary."),
            ("30_non_calibrated_and_calibrated_raw_vs_model_norm.pdf", "Combined summary bundle across diagnostic families."),
            ("31_gap_metric_bundle_raw_vs_model_norm.pdf", "Combined Δ-metric bundle relative to the random-null family."),
            ("38_four_view_jsd_vs_top1_ece.pdf", "Compact four-view comparison between JSD and the Top-1 ECE diagnostic family."),
        ],
    },
    {
        "title": "4. Matched Non-Calibrated & Calibrated LogitDiff",
        "question": "What changes when we compare the two LogitDiff measurement modes directly using the same JS-similarity family?",
        "intuition": (
            "This is the core comparison family for the report. Here, Non-Calibrated and Calibrated are two measurement modes "
            "applied to the same underlying JS-similarity view, which makes the figures directly comparable."
        ),
        "math": (
            "Both views use JS similarity over matched prompt artifacts. The non-calibrated mode summarizes the direct similarity "
            "surface, while the calibrated mode uses the calibrated similarity formulation. Δ-figures subtract the random-null reference."
        ),
        "figures": [
            ("40_non_calibrated_js_similarity_raw_vs_model_norm.pdf", "Non-Calibrated JS-similarity summary."),
            ("41_calibrated_js_similarity_raw_vs_model_norm.pdf", "Calibrated JS-similarity summary."),
            ("42_non_calibrated_js_similarity_score_and_gap_raw_vs_model_norm.pdf", "Non-Calibrated JS-similarity with Δ-family view."),
            ("43_calibrated_js_similarity_score_and_gap_raw_vs_model_norm.pdf", "Calibrated JS-similarity with Δ-family view."),
            ("44_non_calibrated_and_calibrated_js_similarity_four_view.pdf", "Four-view matched comparison across Raw and ModelNorm."),
            ("45_non_calibrated_and_calibrated_js_similarity_raw_vs_model_norm.pdf", "Stacked summary bundle for both measurement modes."),
            ("46_non_calibrated_and_calibrated_js_similarity_gap_raw_vs_model_norm.pdf", "Stacked Δ bundle for both measurement modes."),
            ("47_non_calibrated_js_similarity_pairwise_raw_vs_model_norm.pdf", "Pairwise non-calibrated JS-similarity curves for individual seed and seed-pair families."),
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a readable prompt LogitDiff PDF report from saved figure PDFs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing generated figure PDFs.")
    parser.add_argument("--output-path", required=True, help="Output PDF report path.")
    parser.add_argument(
        "--title",
        default="Prompt LogitDiff Analysis Report",
        help="Main report title.",
    )
    return parser.parse_args()


def _render_pdf_page_to_png(pdf_path: Path, png_prefix: Path) -> Path:
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", str(pdf_path), str(png_prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return png_prefix.with_suffix(".png")


def _new_page() -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    return fig


def _add_cover_page(pdf: PdfPages, title: str, input_dir: Path) -> None:
    fig = _new_page()
    fig.text(0.08, 0.92, title, fontsize=24, fontweight="bold", va="top")
    fig.text(0.08, 0.86, "Prompt-side LogitDiff review", fontsize=15, fontweight="semibold", va="top")
    fig.text(
        0.08,
        0.79,
        fill(
            "This document groups the saved sanity-check and matched-similarity figures into one readable report. "
            "It is intended for review of figure behavior, measurement semantics, and which figure families are useful "
            "for paper-facing discussion.",
            width=88,
        ),
        fontsize=11.5,
        va="top",
    )
    fig.text(0.08, 0.70, f"Figure directory: {input_dir}", fontsize=10.5, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _add_text_page(pdf: PdfPages, title: str, blocks: list[str]) -> None:
    fig = _new_page()
    y = 0.94
    fig.text(0.08, y, title, fontsize=20, fontweight="bold", va="top")
    y -= 0.06
    for block in blocks:
        wrapped = fill(block, width=90)
        fig.text(0.08, y, wrapped, fontsize=11.5, va="top")
        y -= 0.030 * (wrapped.count("\n") + 2.2)
    pdf.savefig(fig)
    plt.close(fig)


def _draw_figure_panel(
    fig: plt.Figure,
    *,
    image_path: Path,
    caption: str,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> None:
    image = imread(image_path)
    ax = fig.add_axes([left, bottom + 0.06, width, height - 0.08])
    ax.imshow(image)
    ax.axis("off")
    fig.text(left, bottom, fill(caption, width=54), fontsize=10.5, va="bottom")


def _add_two_figure_page(
    pdf: PdfPages,
    *,
    title: str,
    items: list[tuple[Path, str]],
) -> None:
    fig = _new_page()
    fig.text(0.08, 0.965, title, fontsize=17, fontweight="bold", va="top")
    slots = [
        (0.06, 0.53, 0.88, 0.36),
        (0.06, 0.08, 0.88, 0.36),
    ]
    for (image_path, caption), (left, bottom, width, height) in zip(items, slots, strict=False):
        _draw_figure_panel(
            fig,
            image_path=image_path,
            caption=caption,
            left=left,
            bottom=bottom,
            width=width,
            height=height,
        )
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="prompt_logitdiff_report_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        with PdfPages(output_path) as pdf:
            _add_cover_page(pdf, args.title, input_dir)

            for section in REPORT_SECTIONS:
                _add_text_page(pdf, str(section["title"]), list(section["blocks"]))  # type: ignore[arg-type]

            for section in FIGURE_GROUPS:
                figure_specs: list[tuple[str, str]] = section["figures"]  # type: ignore[assignment]
                existing: list[tuple[Path, str]] = []
                for idx, (filename, caption) in enumerate(figure_specs, start=1):
                    pdf_path = input_dir / filename
                    if not pdf_path.exists():
                        continue
                    png_path = _render_pdf_page_to_png(pdf_path, tmp_dir / f"{pdf_path.stem}_{idx}")
                    existing.append((png_path, f"{filename}. {caption}"))

                if not existing:
                    continue

                _add_text_page(
                    pdf,
                    str(section["title"]),
                    [
                        f"Question: {section['question']}",
                        f"Intuition: {section['intuition']}",
                        f"Math: {section['math']}",
                    ],
                )

                for idx in range(0, len(existing), 2):
                    _add_two_figure_page(
                        pdf,
                        title=str(section["title"]),
                        items=existing[idx : idx + 2],
                    )

            _add_text_page(
                pdf,
                "5. Reading Guide",
                [
                    "Use the matched JS-similarity family (40–47) as the primary comparison set when discussing Non-Calibrated versus Calibrated LogitDiff.",
                    "Use the original diagnostic family (00–38) as supporting sanity checks and to inspect how calibration diagnostics differ from direct divergence summaries.",
                    "When a panel uses the delta symbol Δ, it is a difference relative to a reference family rather than an absolute score curve.",
                ],
            )


if __name__ == "__main__":
    main()

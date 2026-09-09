from __future__ import annotations

from pathlib import Path
import sys

import plotly.io as pio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_llama_hf1bit_translation_dual_heatmap import _build_combined_figure  # noqa: E402


def main() -> None:
    prompt_data = torch.load(
        "/media/am/AM/logit-diff-lens/tmp/llama_flower_cache/heatmap_payloads/prompt_data.pt",
        map_location="cpu",
        weights_only=False,
    )
    generation_data = torch.load(
        "/media/am/AM/logit-diff-lens/tmp/llama_flower_cache/heatmap_payloads/generation_data.pt",
        map_location="cpu",
        weights_only=False,
    )
    fig = _build_combined_figure(prompt_data, generation_data, top_k=5)

    svg_path = Path("/media/am/AM/logit-diff-lens/Figures/LLaMA/llama_flower_cached_dual_heatmap.svg")
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_bytes = pio.to_image(fig, format="svg", engine="kaleido")
    svg_path.write_bytes(svg_bytes)
    print(f"WROTE_SVG {svg_path} {svg_path.stat().st_size}")


if __name__ == "__main__":
    main()

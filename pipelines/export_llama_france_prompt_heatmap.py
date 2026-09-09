from __future__ import annotations

from pathlib import Path
import sys

import plotly.io as pio
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from logit_diff_lens.cli.prompt_analysis_utils import resolve_prompt_artifact  # noqa: E402
from logit_diff_lens.plotting.logitdiff_prompt_plotter import plot_prompt_logitdiff_jaccard_heatmap as plot_prompt_heatmap  # noqa: E402
from plot_prompt_heatmap import _build_prompt_logitdiff_results  # noqa: E402


def write_plotly_pdf(fig, output_path: Path) -> None:
    png_bytes = pio.to_image(fig, format="png", engine="kaleido", scale=2)
    png_path = output_path.with_suffix(".png")
    png_path.write_bytes(png_bytes)
    image = Image.open(png_path).convert("RGB")
    image.save(output_path, "PDF", resolution=300.0)


def main() -> None:
    base_prompt_path = "/media/am/AM/logit-diff-lens/tmp/llama_france_prompt/base_prompt_artifact.pt"
    ft_prompt_path = "/media/am/AM/logit-diff-lens/tmp/llama_france_prompt/bitnet_prompt_artifact.pt"

    ft_artifact = resolve_prompt_artifact(ft_prompt_path, prompt_index=0, prompt_id=None, prompt_text=None)
    base_artifact = resolve_prompt_artifact(base_prompt_path, prompt_index=0, prompt_id=None, prompt_text=None)
    tokenizer_name = (
        ft_artifact.backend_metadata.tokenizer_id
        or ft_artifact.backend_metadata.model_id
        or base_artifact.backend_metadata.tokenizer_id
        or base_artifact.backend_metadata.model_id
    )
    if tokenizer_name is None:
        raise ValueError("Missing tokenizer provenance for prompt heatmap export.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=False)

    prompt_payload = _build_prompt_logitdiff_results(
        ft_artifacts=[ft_artifact],
        base_artifacts=[base_artifact],
        tokenizer=tokenizer,
        readout_mode="model_norm",
        top_k=5,
    )
    prompt_fig = plot_prompt_heatmap(
        prompt_payload,
        prompt_index=0,
        prompt_text=None,
        include_prompt_tokens=True,
        include_generated_tokens=False,
        start_idx=0,
        end_idx=None,
        title="Base LLaMA <> LLaMA 1.58-bit",
        colorscale="Blues",
        display_top_tokens=5,
        visible_cell_tokens=5,
        max_token_chars=12,
        show_marginals=False,
        max_layers=None,
        layer_selection="all",
        analysis_topk=5,
        x_tick_mode="prompt",
    )
    prompt_out = Path("/media/am/AM/logit-diff-lens/Figures/LLaMA/llama_france_prompt_heatmap.pdf")
    write_plotly_pdf(prompt_fig, prompt_out)
    print(f"WROTE_PROMPT {prompt_out}")


if __name__ == "__main__":
    main()

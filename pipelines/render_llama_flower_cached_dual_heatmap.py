from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from logit_diff_lens.plotting.plotly_export import save_plotly_figure

from plot_llama_hf1bit_translation_dual_heatmap import _build_combined_figure, _crop_columns


def _select_layers(data: dict[str, Any], max_layers: int) -> dict[str, Any]:
    num_rows = int(data["z"].shape[0])
    if max_layers <= 0 or num_rows <= max_layers:
        return data

    indices = np.linspace(0, num_rows - 1, max_layers, dtype=int)
    selected = dict(data)
    selected["z"] = data["z"][indices, :]
    selected["hover_text"] = data["hover_text"][indices, :]
    selected["cell_parts"] = data["cell_parts"][indices, :]
    selected["meta"] = [data["meta"][int(idx)] for idx in indices]
    selected["y_labels"] = [data["y_labels"][int(idx)] for idx in indices]
    selected["mean_per_layer"] = np.asarray([data["mean_per_layer"][int(idx)] for idx in indices], dtype=float)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Render cached llama flower dual heatmap from saved data bundles.")
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--generation-data", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--format", choices=("pdf", "png", "html"), default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--prompt-columns", type=int, default=8)
    parser.add_argument("--generation-columns", type=int, default=6)
    parser.add_argument("--max-layers", type=int, default=10)
    args = parser.parse_args()

    prompt_data = torch.load(args.prompt_data, map_location="cpu", weights_only=False)
    generation_data = torch.load(args.generation_data, map_location="cpu", weights_only=False)
    prompt_data = _crop_columns(prompt_data, args.prompt_columns)
    generation_data = _crop_columns(generation_data, args.generation_columns)
    prompt_data = _select_layers(prompt_data, args.max_layers)
    generation_data = _select_layers(generation_data, args.max_layers)
    fig = _build_combined_figure(prompt_data, generation_data, top_k=args.top_k)
    fig.update_yaxes(title_text=None, row=1, col=1)
    fig.update_yaxes(title_text=None, row=1, col=2)
    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(fig.to_plotly_json()), encoding="utf-8")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.format or output_path.suffix.lower().lstrip(".")
    save_plotly_figure(fig, output_path, format=output_format)


if __name__ == "__main__":
    main()

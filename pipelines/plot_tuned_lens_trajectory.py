from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    tuned_lens_root = root / "tuned-lens"
    for candidate in (src, tuned_lens_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def main() -> None:
    _bootstrap_paths()

    import argparse
    import torch

    from logit_diff_lens.plotting.tuned_lens_trajectory import (
        load_model_and_tokenizer,
        save_trajectory_figure,
    )

    parser = argparse.ArgumentParser(
        description="Render tuned-lens-style trajectory plots for local Raw/ModelNorm/Tuned readouts without modifying tuned-lens."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--lens-kind", choices=("raw", "model_norm", "tuned"), default="model_norm")
    parser.add_argument("--statistic", choices=(
        "cross_entropy",
        "rank",
        "entropy",
        "forward_kl",
        "max_probability",
        "js_divergence",
        "kl_divergence",
        "total_variation",
    ), default="forward_kl")
    parser.add_argument("--tuned-lens-resource", default=None)
    parser.add_argument("--compare-to-kind", choices=("raw", "model_norm", "tuned"), default=None)
    parser.add_argument("--compare-to-tuned-lens-resource", default=None)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--label-topk", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--token-width", type=int, default=90)
    parser.add_argument("--title", default=None)
    parser.add_argument("--format", default=None)
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--with-shifted-targets", action="store_true")
    parser.add_argument("--mask-input", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        dtype=dtype_map[args.dtype],
        device=args.device,
    )

    output_path = save_trajectory_figure(
        args.output_path,
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        lens_kind=args.lens_kind,
        statistic=args.statistic,
        tuned_lens_resource=args.tuned_lens_resource,
        compare_to_kind=args.compare_to_kind,
        compare_to_tuned_lens_resource=args.compare_to_tuned_lens_resource,
        add_special_tokens=args.add_special_tokens,
        with_shifted_targets=args.with_shifted_targets,
        mask_input=args.mask_input,
        label_topk=args.label_topk,
        stride=args.stride,
        token_width=args.token_width,
        title=args.title,
        format=args.format,
    )
    print(output_path)


if __name__ == "__main__":
    main()

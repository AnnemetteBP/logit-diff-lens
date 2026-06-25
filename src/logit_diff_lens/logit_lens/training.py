from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def _configure_local_hf_dataset_cache() -> Path:
    root = Path(__file__).resolve().parents[3]
    hf_home = root / "tmp" / "hf_home"
    datasets_cache = hf_home / "datasets"
    hub_cache = hf_home / "hub"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    return datasets_cache


def parse_precision(value: str) -> str:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return "bfloat16"
    if normalized in {"fp16", "float16"}:
        return "float16"
    if normalized in {"fp32", "float32"}:
        return "float32"
    if normalized in {"int8", "8bit"}:
        return "int8"
    if normalized == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype/precision: {value}")


def _truncate_jsonl_if_needed(
    dataset_path: str | Path,
    *,
    max_records: int | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    path = Path(dataset_path)
    if max_records is None:
        return path, None

    tmpdir = tempfile.TemporaryDirectory(prefix="tuned_lens_subset_")
    subset_path = Path(tmpdir.name) / path.name
    kept = 0
    with path.open("r", encoding="utf-8") as src, subset_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if kept >= max_records:
                break
            if not line.strip():
                continue
            dst.write(line)
            kept += 1

    if kept == 0:
        tmpdir.cleanup()
        raise ValueError(f"No records were copied from {path}")

    return subset_path, tmpdir


def estimate_num_steps(
    *,
    model_name: str,
    dataset_path: str | Path,
    text_key: str,
    seq_len: int,
    batch_size: int,
    epochs: int,
    precision: str,
) -> int:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if epochs <= 0:
        raise ValueError("--epochs must be positive.")

    _configure_local_hf_dataset_cache()
    from tuned_lens.scripts.ingredients import Data, Model

    model_cfg = Model(name=model_name, precision=precision)
    tokenizer = model_cfg.load_tokenizer()
    data_cfg = Data(
        name=[str(dataset_path)],
        text_column=text_key,
        max_seq_len=seq_len,
    )
    processed, _ = data_cfg.load(tokenizer)
    total_samples = len(processed)
    steps_per_epoch = total_samples // batch_size
    if steps_per_epoch <= 0:
        raise ValueError(
            "Dataset is too small for the requested --batch-size after tokenization. "
            f"Got {total_samples} tokenized samples and batch size {batch_size}."
        )
    return steps_per_epoch * epochs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tuned lens via the local tuned-lens project CLI."
    )
    parser.add_argument("--model-name", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--wandb", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def build_tuned_lens_cli_args(args: argparse.Namespace) -> list[str]:
    precision = parse_precision(args.dtype)
    dataset_path, tmpdir = _truncate_jsonl_if_needed(
        args.dataset_path,
        max_records=args.max_records,
    )
    try:
        num_steps = estimate_num_steps(
            model_name=args.model_name,
            dataset_path=dataset_path,
            text_key=args.text_key,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            precision=precision,
        )

        lr_scale = (
            args.lr / 1e-3 if args.optimizer == "adam" else args.lr / (1.0 - 0.9)
        )
        tokens_per_step = args.seq_len * args.batch_size
        cli_args = [
            "--log_level",
            args.log_level,
            "train",
            "--data.name",
            str(dataset_path),
            "--model.name",
            args.model_name,
            "--output",
            args.output_dir,
            "--text_column",
            args.text_key,
            "--max_seq_len",
            str(args.seq_len),
            "--per_gpu_batch_size",
            str(args.batch_size),
            "--tokens_per_step",
            str(tokens_per_step),
            "--num_steps",
            str(num_steps),
            "--precision",
            precision,
            "--optimizer",
            args.optimizer,
            "--lr_scale",
            str(lr_scale),
            "--weight_decay",
            str(args.weight_decay),
            "--seed",
            str(args.seed),
        ]
        if args.wandb:
            cli_args.extend(["--wandb", args.wandb])
        if args.checkpoint_freq is not None:
            cli_args.extend(["--checkpoint_freq", str(args.checkpoint_freq)])
        return cli_args
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def main(argv: list[str] | None = None) -> None:
    _configure_local_hf_dataset_cache()
    args = build_arg_parser().parse_args(argv)
    cli_args = build_tuned_lens_cli_args(args)
    from tuned_lens.__main__ import main as tuned_lens_main

    dataset_path, tmpdir = _truncate_jsonl_if_needed(
        args.dataset_path,
        max_records=args.max_records,
    )
    try:
        if str(dataset_path) != str(args.dataset_path):
            dataset_idx = cli_args.index("--data.name") + 1
            cli_args[dataset_idx] = str(dataset_path)
        tuned_lens_main(cli_args)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


__all__ = [
    "build_arg_parser",
    "build_tuned_lens_cli_args",
    "estimate_num_steps",
    "main",
    "parse_precision",
]

from __future__ import annotations

import argparse


def add_stable_analysis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stable-analysis", action="store_true", default=True)
    parser.add_argument("--no-stable-analysis", dest="stable_analysis", action="store_false")


def add_generation_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    include_batch_size: bool = True,
) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--do-sample", action="store_true", default=True)
    parser.add_argument("--no-do-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    if include_batch_size:
        parser.add_argument("--batch-size", type=int, default=10)


__all__ = [
    "add_generation_runtime_args",
    "add_stable_analysis_args",
]

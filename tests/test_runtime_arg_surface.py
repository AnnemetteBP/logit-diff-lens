from __future__ import annotations

from pipelines.capture_generation_artifacts import build_arg_parser as build_generation_capture_parser
from pipelines.plot_generation_heatmap import build_arg_parser as build_generation_plot_parser
from logit_diff_lens.logit_lens.backward import build_arg_parser as build_backward_parser
from logit_diff_lens.logit_lens.capture import build_arg_parser as build_prompt_capture_parser
from logit_diff_lens.logit_lens.patchscope import build_arg_parser as build_patchscope_parser


def test_generation_capture_parser_accepts_generation_runtime_controls() -> None:
    parser = build_generation_capture_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "EleutherAI/pythia-70m-deduped",
            "--prompt",
            "demo",
            "--output-path",
            "tmp/demo_generation.pt",
            "--no-do-sample",
            "--temperature",
            "0.7",
            "--seed",
            "17",
            "--no-stable-analysis",
        ]
    )
    assert args.do_sample is False
    assert args.temperature == 0.7
    assert args.seed == 17
    assert args.stable_analysis is False


def test_generation_plot_parser_accepts_generation_runtime_controls() -> None:
    parser = build_generation_plot_parser()
    args = parser.parse_args(
        [
            "--output-path",
            "tmp/demo_generation_plot.pdf",
            "--model-name",
            "EleutherAI/pythia-70m-deduped",
            "--comparison-model-name",
            "EleutherAI/pythia-70m-deduped",
            "--prompt",
            "demo",
            "--no-do-sample",
            "--temperature",
            "0.5",
            "--seed",
            "3",
        ]
    )
    assert args.do_sample is False
    assert args.temperature == 0.5
    assert args.seed == 3


def test_prompt_capture_parser_accepts_padding_and_stable_controls() -> None:
    parser = build_prompt_capture_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "EleutherAI/pythia-70m-deduped",
            "--prompt",
            "demo",
            "--output-path",
            "tmp/demo_prompt.pt",
            "--model-revision",
            "step71000",
            "--padding",
            "longest",
            "--no-stable-analysis",
        ]
    )
    assert args.model_revision == "step71000"
    assert args.padding == "longest"
    assert args.stable_analysis is False


def test_backward_parser_accepts_stable_analysis_toggle() -> None:
    parser = build_backward_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "EleutherAI/pythia-70m-deduped",
            "--prompt",
            "demo",
            "--target-token-id",
            "1",
            "--output-path",
            "tmp/backward.pt",
            "--no-stable-analysis",
        ]
    )
    assert args.stable_analysis is False


def test_patchscope_parser_accepts_stable_analysis_toggle() -> None:
    parser = build_patchscope_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "EleutherAI/pythia-70m-deduped",
            "--source-artifact",
            "tmp/source.pt",
            "--target-prompt",
            "demo",
            "--source-layer-index",
            "0",
            "--source-position",
            "0",
            "--target-layer-index",
            "0",
            "--target-position",
            "0",
            "--output-path",
            "tmp/patchscope.pt",
            "--no-stable-analysis",
        ]
    )
    assert args.stable_analysis is False

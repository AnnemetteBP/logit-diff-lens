from __future__ import annotations

import pytest

from logit_diff_lens.logit_lens.capture import build_arg_parser, main


def test_capture_cli_parser_accepts_prompt_mode() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--model-name",
            "EleutherAI/pythia-70m-deduped",
            "--prompt",
            "Demo prompt",
            "--output-path",
            "tmp/demo_artifact.pt",
        ]
    )
    assert args.prompt == "Demo prompt"
    assert args.dataset_path is None


def test_capture_cli_requires_exactly_one_input_mode() -> None:
    with pytest.raises(ValueError, match="Provide exactly one of --prompt or --dataset-path."):
        main(
            [
                "--model-name",
                "EleutherAI/pythia-70m-deduped",
                "--output-path",
                "tmp/demo_artifact.pt",
            ]
        )


def test_capture_cli_rejects_project_component_logits_without_components() -> None:
    with pytest.raises(ValueError, match="--project-component-logits requires --collect-components."):
        main(
            [
                "--model-name",
                "EleutherAI/pythia-70m-deduped",
                "--prompt",
                "Demo prompt",
                "--output-path",
                "tmp/demo_artifact.pt",
                "--project-component-logits",
            ]
        )

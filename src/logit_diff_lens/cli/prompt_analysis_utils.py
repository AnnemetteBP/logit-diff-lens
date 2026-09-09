from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..collectors.prompt import (
    PromptLensActivationCollectorConfig,
    _build_collection_text_and_kind,
    collect_prompt_lens_activations,
)
from ..diffing.io import load_prompt_decode_artifact, load_prompt_decode_artifact_bundle
from ..logit_lens.capture import _load_model_and_tokenizer
from ..schemas import PromptDecodeArtifact
from ..wrappers import LogitLensWrapper


def add_prompt_runtime_args(parser, *, include_comparison: bool = False) -> None:
    parser.add_argument("--model-name", default=None)
    if include_comparison:
        parser.add_argument("--comparison-model-name", default=None)
        parser.add_argument("--comparison-adapter-path", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument(
        "--prompt-format",
        choices=("plain", "chat_template", "user_assistant_prefix"),
        default="plain",
    )
    parser.add_argument("--system-prompt", default=None)
    if include_comparison:
        parser.add_argument("--comparison-use-chat-template", action="store_true")
        parser.add_argument(
            "--comparison-prompt-format",
            choices=("plain", "chat_template", "user_assistant_prefix"),
            default=None,
        )
        parser.add_argument("--comparison-system-prompt", default=None)
    parser.add_argument("--no-add-special-tokens", action="store_true")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--force-include-input", action="store_true", default=True)
    parser.add_argument("--no-force-include-input", dest="force_include_input", action="store_false")
    parser.add_argument("--force-include-output", action="store_true")
    parser.add_argument("--no-force-include-output", dest="force_include_output", action="store_false")
    parser.add_argument("--normalize-embedding-for-readout", action="store_true")
    parser.add_argument("--norm-modes", nargs="+", default=("raw", "model_norm"))
    parser.add_argument("--collect-components", action="store_true")
    parser.add_argument("--project-component-logits", action="store_true")
    parser.add_argument("--tuned-lens-resource-id", default=None)
    parser.add_argument("--save-logits", action="store_true", default=True)
    parser.add_argument("--no-save-logits", dest="save_logits", action="store_false")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--stable-analysis", action="store_true", default=True)
    parser.add_argument("--no-stable-analysis", dest="stable_analysis", action="store_false")


def build_prompt_wrapper(
    *,
    model_name: str,
    tokenizer_name: str | None,
    adapter_path: str | None,
    dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
    load_in_4bit: bool,
    load_in_8bit: bool,
    debug: bool,
    stable_analysis: bool,
) -> LogitLensWrapper:
    model, tokenizer = _load_model_and_tokenizer(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_revision=None,
        precision=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        adapter_path=adapter_path,
    )
    return LogitLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=debug,
        stable_analysis=stable_analysis,
    )


def collect_prompt_artifacts_live(
    wrapper: LogitLensWrapper,
    *,
    prompt: str | None,
    dataset_path: str | None,
    text_field: str,
    use_chat_template: bool,
    prompt_format: str,
    system_prompt: str | None,
    add_special_tokens: bool,
    truncation: bool,
    max_length: int | None,
    force_include_input: bool,
    force_include_output: bool,
    normalize_embedding_for_readout: bool,
    norm_modes: Sequence[str],
    collect_components: bool,
    project_component_logits: bool,
    save_logits: bool,
    tuned_lens_resource_id: str | None,
) -> list[PromptDecodeArtifact]:
    if bool(prompt) == bool(dataset_path):
        raise ValueError("Provide exactly one of --prompt or --dataset-path for live prompt analysis.")
    if tuned_lens_resource_id and not save_logits:
        raise ValueError("tuned_lens_resource_id requires save_logits=True so tuned logits can be stored.")

    tuned_lens = None
    if tuned_lens_resource_id is not None:
        from ..collectors.prompt import _load_tuned_lens

        tuned_lens = _load_tuned_lens(wrapper, resource_id=tuned_lens_resource_id)

    def _collect_one(prompt_text: str) -> PromptDecodeArtifact:
        payload = collect_prompt_lens_activations(
            wrapper,
            PromptLensActivationCollectorConfig(
                prompt=prompt_text,
                use_chat_template=use_chat_template,
                prompt_format=prompt_format,
                system_prompt=system_prompt,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
                padding=None,
                force_include_input=force_include_input,
                force_include_output=force_include_output,
                normalize_embedding_for_readout=normalize_embedding_for_readout,
                norm_modes=tuple(norm_modes),
                collect_components=collect_components,
                project_component_logits=project_component_logits,
                save_logits=save_logits,
                tuned_lens_resource_id=tuned_lens_resource_id,
            ),
            tuned_lens=tuned_lens,
        )
        return payload["artifact"]

    if prompt is not None:
        return [_collect_one(prompt)]

    rows = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    artifacts: list[PromptDecodeArtifact] = []
    for row in rows:
        prompt_text, continuation_kind = _build_collection_text_and_kind(row, text_field=text_field)
        artifact = _collect_one(prompt_text)
        artifact.prompt_id = str(row.get("id")) if row.get("id") is not None else None
        artifact.metadata.update(
            {
                "row_id": row.get("id"),
                "group_id": row.get("group_id"),
                "variant": row.get("variant"),
                "language": row.get("language"),
                "continuation_kind": continuation_kind,
                "dataset_path": str(dataset_path),
                "text_field": text_field,
            }
        )
        artifacts.append(artifact)
    return artifacts


def select_prompt_artifact(
    artifacts: Sequence[PromptDecodeArtifact],
    *,
    prompt_index: int | None = 0,
    prompt_id: str | None = None,
    prompt_text: str | None = None,
) -> PromptDecodeArtifact:
    if not artifacts:
        raise ValueError("No prompt artifacts available for selection.")
    selectors = sum(value is not None for value in (prompt_id, prompt_text)) + (0 if prompt_index in (None, 0) else 1)
    if selectors > 1:
        raise ValueError("Use only one of prompt_index, prompt_id, or prompt_text when selecting a saved prompt artifact.")
    if prompt_id is not None:
        for artifact in artifacts:
            if artifact.prompt_id == prompt_id:
                return artifact
        raise ValueError(f"No saved prompt artifact matched prompt_id={prompt_id!r}")
    if prompt_text is not None:
        for artifact in artifacts:
            if artifact.prompt_text == prompt_text:
                return artifact
        raise ValueError(f"No saved prompt artifact matched prompt_text={prompt_text!r}")
    idx = 0 if prompt_index is None else int(prompt_index)
    try:
        return artifacts[idx]
    except IndexError as exc:
        raise ValueError(f"prompt_index={idx} out of range for bundle with {len(artifacts)} artifacts") from exc


def resolve_prompt_artifact(
    source: str | Path,
    *,
    prompt_index: int = 0,
    prompt_id: str | None = None,
    prompt_text: str | None = None,
) -> PromptDecodeArtifact:
    path = Path(source)
    payload = load_prompt_decode_artifact_bundle(path) if path.suffix == ".pt" and _looks_like_bundle(path) else None
    if payload is not None:
        return select_prompt_artifact(
            payload["artifacts"],
            prompt_index=prompt_index,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
        )
    return load_prompt_decode_artifact(path)


def _looks_like_bundle(path: Path) -> bool:
    try:
        import torch

        payload = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return isinstance(payload, dict) and "artifacts" in payload


def resolve_saved_pair(
    *,
    path_a: str | None,
    path_b: str | None,
    base_artifact: str | None,
    comparison_artifact: str | None,
) -> tuple[str, str] | None:
    old_pair = path_a is not None or path_b is not None
    new_pair = base_artifact is not None or comparison_artifact is not None
    if not old_pair and not new_pair:
        return None
    if old_pair and new_pair:
        raise ValueError("Use either the legacy artifact flags or --base-artifact/--comparison-artifact, not both.")
    if new_pair:
        if base_artifact is None or comparison_artifact is None:
            raise ValueError("Saved mode requires both --base-artifact and --comparison-artifact.")
        return comparison_artifact, base_artifact
    if path_a is None or path_b is None:
        raise ValueError("Saved mode requires both artifact inputs.")
    return path_a, path_b

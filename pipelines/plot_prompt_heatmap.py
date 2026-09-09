from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer

from logit_diff_lens.collectors.prompt import (
    PromptLensActivationCollectorConfig,
    _build_collection_text_and_kind,
    _load_tuned_lens,
    collect_prompt_lens_activations,
)
from logit_diff_lens.diffing import compare_prompt_artifacts_ft_minus_base, load_comparison_artifact
from logit_diff_lens.cli.prompt_analysis_utils import resolve_prompt_artifact
from logit_diff_lens.logit_lens.capture import _load_model_and_tokenizer
from logit_diff_lens.plotting.logitdiff_prompt_plotter import save_prompt_logitdiff_heatmap
from logit_diff_lens.plotting import plot_comparison_metric_heatmap
from logit_diff_lens.plotting.plotly_export import save_plotly_figure
from logit_diff_lens.schemas import PromptDecodeArtifact
from logit_diff_lens.wrappers import LogitLensWrapper


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Plotly prompt heatmap from a saved LogitDiff payload "
            "or compute the prompt analysis live before plotting."
        )
    )
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--ft-artifact", default=None)
    parser.add_argument("--base-artifact", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--plot-kind",
        choices=("jaccard", "next_token_verification", "comparison_metric"),
        default="jaccard",
    )
    parser.add_argument("--metric", default="jsd_ft_base")
    parser.add_argument("--title", default=None)
    parser.add_argument("--colorscale", default="RdBu")
    parser.add_argument("--format", choices=("html", "pdf"), default=None)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--display-top-tokens", type=int, default=10)
    parser.add_argument("--visible-cell-tokens", type=int, default=None)
    parser.add_argument("--max-divergent-layers", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--keep-last-layer-fraction", type=float, default=0.5)
    parser.add_argument("--exclude-prompt-tokens", action="store_true")
    parser.add_argument("--exclude-generated-tokens", action="store_true")
    parser.add_argument("--start-position", type=int, default=None)
    parser.add_argument("--end-position", type=int, default=None)
    parser.add_argument("--max-token-chars", type=int, default=18)
    parser.add_argument(
        "--layer-selection",
        choices=("all", "most_divergent", "least_divergent"),
        default="most_divergent",
    )
    parser.add_argument(
        "--x-tick-mode",
        choices=("prompt", "base_generated", "position"),
        default="base_generated",
    )
    parser.add_argument(
        "--x-tick-mode-secondary",
        choices=("prompt", "base_generated", "position", "none"),
        default="none",
    )
    parser.add_argument("--show-marginals", action="store_true")
    parser.add_argument("--analysis-topk", type=int, default=None)

    parser.add_argument("--model-name", default=None)
    parser.add_argument("--comparison-model-name", default=None)
    parser.add_argument("--comparison-adapter-path", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--label-field", default=None)
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
    parser.add_argument(
        "--norm-modes",
        nargs="+",
        default=("raw", "model_norm"),
    )
    parser.add_argument("--readout-mode", choices=("raw", "model_norm", "tuned"), default="model_norm")
    parser.add_argument("--collect-components", action="store_true")
    parser.add_argument("--project-component-logits", action="store_true")
    parser.add_argument("--tuned-lens-resource-id", default=None)
    parser.add_argument("--save-logits", action="store_true", default=True)
    parser.add_argument("--no-save-logits", dest="save_logits", action="store_false")
    parser.add_argument("--stable-analysis", action="store_true", default=True)
    parser.add_argument("--no-stable-analysis", dest="stable_analysis", action="store_false")
    parser.add_argument("--debug", action="store_true")
    return parser


def _build_wrapper(
    *,
    model_name: str,
    tokenizer_name: str | None,
    model_revision: str | None = None,
    adapter_path: str | None,
    dtype: str,
    trust_remote_code: bool,
    device_map: str | None,
    load_in_4bit: bool,
    load_in_8bit: bool,
    stable_analysis: bool,
    debug: bool,
) -> LogitLensWrapper:
    model, tokenizer = _load_model_and_tokenizer(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        model_revision=model_revision,
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


def _collect_single_prompt_artifact(
    wrapper: LogitLensWrapper,
    *,
    prompt: str,
    use_chat_template: bool,
    prompt_format: str,
    system_prompt: str | None,
    add_special_tokens: bool,
    truncation: bool,
    max_length: int | None,
    force_include_input: bool,
    force_include_output: bool,
    norm_modes: Sequence[str],
    collect_components: bool,
    project_component_logits: bool,
    save_logits: bool,
    tuned_lens_resource_id: str | None,
    tuned_lens: Any = None,
) -> PromptDecodeArtifact:
    payload = collect_prompt_lens_activations(
        wrapper,
        PromptLensActivationCollectorConfig(
            prompt=prompt,
            use_chat_template=use_chat_template,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            padding=None,
            force_include_input=force_include_input,
            force_include_output=force_include_output,
            norm_modes=tuple(norm_modes),
            collect_components=collect_components,
            project_component_logits=project_component_logits,
            save_logits=save_logits,
            tuned_lens_resource_id=tuned_lens_resource_id,
        ),
        tuned_lens=tuned_lens,
    )
    return payload["artifact"]


def _collect_prompt_artifacts(
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
    norm_modes: Sequence[str],
    collect_components: bool,
    project_component_logits: bool,
    save_logits: bool,
    tuned_lens_resource_id: str | None,
) -> list[PromptDecodeArtifact]:
    if bool(prompt) == bool(dataset_path):
        raise ValueError("Provide exactly one of --prompt or --dataset-path for live prompt plotting.")
    tuned_lens = None
    if tuned_lens_resource_id is not None:
        tuned_lens = _load_tuned_lens(wrapper, resource_id=tuned_lens_resource_id)
    if prompt is not None:
        return [
            _collect_single_prompt_artifact(
                wrapper,
                prompt=prompt,
                use_chat_template=use_chat_template,
                prompt_format=prompt_format,
                system_prompt=system_prompt,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
                force_include_input=force_include_input,
                force_include_output=force_include_output,
                norm_modes=norm_modes,
                collect_components=collect_components,
                project_component_logits=project_component_logits,
                save_logits=save_logits,
                tuned_lens_resource_id=tuned_lens_resource_id,
                tuned_lens=tuned_lens,
            )
        ]

    rows = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    artifacts: list[PromptDecodeArtifact] = []
    for row in rows:
        prompt_text, _ = _build_collection_text_and_kind(row, text_field=text_field)
        artifacts.append(
            _collect_single_prompt_artifact(
                wrapper,
                prompt=prompt_text,
                use_chat_template=use_chat_template,
                prompt_format=prompt_format,
                system_prompt=system_prompt,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
                force_include_input=force_include_input,
                force_include_output=force_include_output,
                norm_modes=norm_modes,
                collect_components=collect_components,
                project_component_logits=project_component_logits,
                save_logits=save_logits,
                tuned_lens_resource_id=tuned_lens_resource_id,
                tuned_lens=tuned_lens,
            )
        )
    return artifacts


def _select_artifact(
    artifacts: Sequence[PromptDecodeArtifact],
    *,
    prompt_index: int | None,
    prompt_text: str | None,
) -> PromptDecodeArtifact:
    if not artifacts:
        raise ValueError("No prompt artifacts were collected.")
    if prompt_text is not None:
        for artifact in artifacts:
            if artifact.prompt_text == prompt_text:
                return artifact
        raise ValueError(f"Prompt not found in collected artifacts: {prompt_text}")
    idx = 0 if prompt_index is None else prompt_index
    return artifacts[idx]


def _get_logits(record: Any, mode: str) -> torch.Tensor:
    if mode not in {"raw", "model_norm", "tuned"}:
        raise ValueError(f"Unsupported readout mode: {mode}")
    if hasattr(record, "get_logits"):
        logits = record.get_logits(mode)
    else:
        logits = getattr(record, f"logits_{mode}", None)
    if logits is None:
        raise ValueError(f"Missing logits for mode={mode} at layer {record.layer_name}")
    return logits.to(dtype=torch.float32)


def _decode_topk_tokens(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
    return ["" if token is None else str(token) for token in tokens]


def _build_prompt_logitdiff_results(
    *,
    ft_artifacts: Sequence[PromptDecodeArtifact],
    base_artifacts: Sequence[PromptDecodeArtifact],
    tokenizer: Any,
    readout_mode: str,
    top_k: int,
) -> dict[str, Any]:
    if len(ft_artifacts) != len(base_artifacts):
        raise ValueError("ft and base live prompt collections must have the same number of prompts.")
    results: dict[str, list[dict[str, Any]]] = {}
    for ft_artifact, base_artifact in zip(ft_artifacts, base_artifacts):
        if not torch.equal(ft_artifact.token_ids, base_artifact.token_ids):
            raise ValueError("Live prompt heatmap compare requires identical tokenization between ft and base.")
        total_layers = len(ft_artifact.layer_records)
        for order_idx, (ft_record, base_record) in enumerate(zip(ft_artifact.layer_records, base_artifact.layer_records)):
            if ft_record.layer_index != base_record.layer_index:
                raise ValueError("Live prompt artifacts have mismatched layer indices.")
            logits_ft = _get_logits(ft_record, readout_mode)[0]
            logits_base = _get_logits(base_record, readout_mode)[0]
            seq_len = logits_ft.shape[0]
            positions: list[dict[str, Any]] = []
            ious: list[float] = []
            for pos_idx in range(seq_len):
                ids_ft = torch.topk(logits_ft[pos_idx], k=min(top_k, logits_ft.shape[-1]), dim=-1).indices.tolist()
                ids_base = torch.topk(logits_base[pos_idx], k=min(top_k, logits_base.shape[-1]), dim=-1).indices.tolist()
                set_ft = set(int(v) for v in ids_ft)
                set_base = set(int(v) for v in ids_base)
                union = set_ft | set_base
                inter = set_ft & set_base
                only_base = list(set_base - set_ft)
                only_ft = list(set_ft - set_base)
                iou = 0.0 if not union else len(inter) / len(union)
                ious.append(iou)
                positions.append(
                    {
                        "position": pos_idx,
                        "input_token": ft_artifact.token_text[pos_idx],
                        "is_generated": False,
                        "iou": float(iou),
                        "intersection": _decode_topk_tokens(tokenizer, sorted(inter)),
                        "only_base": _decode_topk_tokens(tokenizer, only_base),
                        "only_finetuned": _decode_topk_tokens(tokenizer, only_ft),
                        "num_intersection": len(inter),
                        "num_only_base": len(only_base),
                        "num_only_finetuned": len(only_ft),
                    }
                )
            layer_key = str(ft_record.layer_index)
            results.setdefault(layer_key, []).append(
                {
                    "prompt": ft_artifact.prompt_text,
                    "layer_relative": (order_idx / max(total_layers - 1, 1)),
                    "layer_absolute": ft_record.layer_index,
                    "mean_iou": float(sum(ious) / len(ious)) if ious else 0.0,
                    "positions": positions,
                }
            )
    first_ft = ft_artifacts[0]
    first_base = base_artifacts[0]
    return {
        "metadata": {
            "base_model_name": first_base.backend_metadata.model_id or "Base",
            "finetuned_model_name": first_ft.backend_metadata.model_id or "Finetuned",
            "top_k": top_k,
            "readout_mode": readout_mode,
        },
        "results": results,
    }


def _compute_live_prompt_payload(args: argparse.Namespace) -> tuple[dict[str, Any] | str, dict[str, Any] | None]:
    if not args.model_name:
        raise ValueError("Live prompt plotting requires --model-name.")
    if args.project_component_logits and not args.collect_components:
        raise ValueError("--project-component-logits requires --collect-components.")
    if not args.comparison_model_name and not args.comparison_adapter_path:
        raise ValueError(
            "Live prompt plotting requires --comparison-model-name or --comparison-adapter-path."
        )

    base_wrapper = _build_wrapper(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        model_revision=None,
        adapter_path=args.adapter_path,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.device_map,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        stable_analysis=bool(args.stable_analysis),
        debug=bool(args.debug),
    )
    comparison_wrapper = _build_wrapper(
        model_name=args.comparison_model_name or args.model_name,
        tokenizer_name=args.tokenizer_name,
        model_revision=None,
        adapter_path=args.comparison_adapter_path,
        dtype=args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=args.device_map,
        load_in_4bit=bool(args.load_in_4bit),
        load_in_8bit=bool(args.load_in_8bit),
        stable_analysis=bool(args.stable_analysis),
        debug=bool(args.debug),
    )

    base_artifacts = _collect_prompt_artifacts(
        base_wrapper,
        prompt=args.prompt,
        dataset_path=args.dataset_path,
        text_field=args.text_field,
        use_chat_template=bool(args.use_chat_template),
        prompt_format=args.prompt_format,
        system_prompt=args.system_prompt,
        add_special_tokens=not bool(args.no_add_special_tokens),
        truncation=bool(args.truncate),
        max_length=args.max_length,
        force_include_input=bool(args.force_include_input),
        force_include_output=bool(args.force_include_output),
        norm_modes=tuple(args.norm_modes),
        collect_components=bool(args.collect_components),
        project_component_logits=bool(args.project_component_logits),
        save_logits=bool(args.save_logits),
        tuned_lens_resource_id=args.tuned_lens_resource_id,
    )
    comparison_artifacts = _collect_prompt_artifacts(
        comparison_wrapper,
        prompt=args.prompt,
        dataset_path=args.dataset_path,
        text_field=args.text_field,
        use_chat_template=bool(args.comparison_use_chat_template or args.use_chat_template),
        prompt_format=args.comparison_prompt_format or args.prompt_format,
        system_prompt=args.comparison_system_prompt if args.comparison_system_prompt is not None else args.system_prompt,
        add_special_tokens=not bool(args.no_add_special_tokens),
        truncation=bool(args.truncate),
        max_length=args.max_length,
        force_include_input=bool(args.force_include_input),
        force_include_output=bool(args.force_include_output),
        norm_modes=tuple(args.norm_modes),
        collect_components=bool(args.collect_components),
        project_component_logits=bool(args.project_component_logits),
        save_logits=bool(args.save_logits),
        tuned_lens_resource_id=args.tuned_lens_resource_id,
    )

    if args.plot_kind == "comparison_metric":
        ft_artifact = _select_artifact(comparison_artifacts, prompt_index=args.prompt_index, prompt_text=args.prompt_text)
        base_artifact = _select_artifact(base_artifacts, prompt_index=args.prompt_index, prompt_text=args.prompt_text)
        comparison = compare_prompt_artifacts_ft_minus_base(
            ft_artifact,
            base_artifact,
            readout_mode=args.readout_mode,
            topk=args.analysis_topk if args.analysis_topk is not None else args.top_k,
            reference_token_ids=ft_artifact.token_ids,
        )
        return "", comparison

    payload = _build_prompt_logitdiff_results(
        ft_artifacts=comparison_artifacts,
        base_artifacts=base_artifacts,
        tokenizer=base_wrapper.tokenizer,
        readout_mode=args.readout_mode,
        top_k=args.analysis_topk if args.analysis_topk is not None else args.top_k,
    )
    return payload, None


def _saved_mode_uses_live_prompt_inputs(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.model_name,
            args.comparison_model_name,
            args.comparison_adapter_path,
            args.tokenizer_name,
            args.adapter_path,
            args.prompt,
            args.dataset_path,
            args.system_prompt,
            args.comparison_system_prompt,
            args.max_length,
            args.device_map,
        )
    ) or args.prompt_format != "plain" or (args.comparison_prompt_format is not None) or any(
        bool(value)
        for value in (
            args.trust_remote_code,
            args.load_in_4bit,
            args.load_in_8bit,
            args.use_chat_template,
            args.comparison_use_chat_template,
            args.no_add_special_tokens,
            args.truncate,
            args.collect_components,
            args.project_component_logits,
            not args.save_logits,
            not args.stable_analysis,
        )
    )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    saved_pair_mode = bool(args.ft_artifact or args.base_artifact)
    if saved_pair_mode and (args.ft_artifact is None or args.base_artifact is None):
        raise ValueError("Saved prompt heatmap pair mode requires both --ft-artifact and --base-artifact.")
    if args.input_path is not None and saved_pair_mode:
        raise ValueError("Use either --input-path or the saved pair --ft-artifact/--base-artifact, not both.")
    if (args.input_path is not None or saved_pair_mode) and _saved_mode_uses_live_prompt_inputs(args):
        raise ValueError("Saved prompt heatmap mode cannot be mixed with live prompt collection arguments.")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = args.format or output_path.suffix.lower().lstrip(".")
    layer_limit = args.max_divergent_layers if args.max_divergent_layers is not None else args.max_layers
    token_limit = args.visible_cell_tokens if args.visible_cell_tokens is not None else args.display_top_tokens
    analysis_topk = args.analysis_topk if args.analysis_topk is not None else args.top_k

    live_payload: dict[str, Any] | str | None = None
    live_comparison: dict[str, Any] | None = None
    saved_pair_payload: dict[str, Any] | None = None
    if saved_pair_mode:
        ft_artifact = resolve_prompt_artifact(
            args.ft_artifact,
            prompt_index=args.prompt_index,
            prompt_id=None,
            prompt_text=args.prompt_text,
        )
        base_artifact = resolve_prompt_artifact(
            args.base_artifact,
            prompt_index=args.prompt_index,
            prompt_id=None,
            prompt_text=args.prompt_text,
        )
        if not torch.equal(ft_artifact.token_ids, base_artifact.token_ids):
            raise ValueError("Saved prompt Jaccard heatmap requires identical tokenization between --ft-artifact and --base-artifact.")
        tokenizer_name = (
            args.tokenizer_name
            or ft_artifact.backend_metadata.tokenizer_id
            or ft_artifact.backend_metadata.model_id
            or args.model_name
        )
        if tokenizer_name is None:
            raise ValueError(
                "Saved prompt Jaccard heatmap pair mode requires tokenizer provenance. "
                "Pass --tokenizer-name or use artifacts that include backend_metadata.tokenizer_id/model_id."
            )
        tokenizer_source = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=bool(args.trust_remote_code),
        )
        saved_pair_payload = _build_prompt_logitdiff_results(
            ft_artifacts=[ft_artifact],
            base_artifacts=[base_artifact],
            tokenizer=tokenizer_source,
            readout_mode=args.readout_mode,
            top_k=analysis_topk,
        )
    elif args.input_path is None:
        live_payload, live_comparison = _compute_live_prompt_payload(args)

    if args.plot_kind == "jaccard":
        if args.input_path is not None and Path(args.input_path).suffix.lower() == ".pt":
            raise ValueError(
                "The original styled prompt Jaccard heatmap cannot be rendered from a saved comparison .pt artifact alone. "
                "Use --ft-artifact and --base-artifact so the tool can rebuild the original shared/base-only/finetuned-only cell structure."
            )
        max_layers = layer_limit
        if args.layer_selection == "all":
            max_layers = None
        common_kwargs = {
            "prompt_index": args.prompt_index,
            "prompt_text": args.prompt_text,
            "include_prompt_tokens": not bool(args.exclude_prompt_tokens),
            "include_generated_tokens": not bool(args.exclude_generated_tokens),
            "start_idx": args.start_position,
            "end_idx": args.end_position,
            "title": args.title,
            "colorscale": args.colorscale,
            "display_top_tokens": token_limit,
            "visible_cell_tokens": args.visible_cell_tokens,
            "max_token_chars": args.max_token_chars,
            "show_marginals": bool(args.show_marginals),
            "max_layers": max_layers,
            "layer_selection": args.layer_selection,
            "analysis_topk": analysis_topk,
            "x_tick_mode": args.x_tick_mode,
        }
        source = (
            args.input_path
            if args.input_path is not None
            else saved_pair_payload if saved_pair_payload is not None
            else live_payload
        )
        save_prompt_logitdiff_heatmap(
            source,
            output_path,
            plot_kind="jaccard",
            format=output_format,
            **common_kwargs,
        )
        return

    if args.plot_kind == "next_token_verification":
        if args.input_path is not None and Path(args.input_path).suffix.lower() == ".pt":
            raise ValueError(
                "The original styled prompt next-token verification heatmap cannot be rendered from a saved comparison .pt artifact alone. "
                "Use --ft-artifact and --base-artifact so the tool can rebuild the original prompt-side token overlap payload."
            )
        common_kwargs = {
            "prompt_index": args.prompt_index,
            "prompt_text": args.prompt_text,
            "top_k": token_limit,
            "max_divergent_layers": 5 if layer_limit is None else layer_limit,
            "keep_last_layer_fraction": args.keep_last_layer_fraction,
            "include_prompt_tokens": not bool(args.exclude_prompt_tokens),
            "include_generated_tokens": not bool(args.exclude_generated_tokens),
            "start_idx": args.start_position,
            "end_idx": args.end_position,
            "max_token_chars": args.max_token_chars,
            "title": args.title,
            "colorscale": args.colorscale,
        }
        source = (
            args.input_path
            if args.input_path is not None
            else saved_pair_payload if saved_pair_payload is not None
            else live_payload
        )
        save_prompt_logitdiff_heatmap(
            source,
            output_path,
            plot_kind="next_token_verification",
            format=output_format,
            **common_kwargs,
        )
        return

    comparison = load_comparison_artifact(args.input_path) if args.input_path is not None else live_comparison
    fig = plot_comparison_metric_heatmap(
        comparison,
        metric_key=args.metric,
        title=args.title,
        colorscale=args.colorscale,
    )
    save_plotly_figure(fig, output_path, format=output_format)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from diffing.logit_lens_methods.base_collector_scripts.prompt_lens.collect_prompt_lens_activations_batched import (
    collect_activation_dataset_incremental,
)
from diffing.logit_lens_methods.tokenizer_loading import load_tokenizer
from diffing.logit_lens_methods.wrapper.lens_wrappers.logit_lens_wrapper import LogitLensWrapper


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return mapping[dtype_name]


def _clear_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_quantization_config(
    *,
    config_name: str | None,
    config_source: str | None,
) -> Any | None:
    if not config_name:
        return None
    if not config_source:
        raise ValueError("quantization_config_source is required when quantization_config_name is set")
    source_path = Path(config_source)
    if not source_path.exists():
        raise FileNotFoundError(f"quantization config source not found: {source_path}")
    spec = importlib.util.spec_from_file_location("prompt_lens_quant_config", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load quantization config source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, config_name):
        raise AttributeError(f"{source_path} does not define {config_name}")
    return getattr(module, config_name)


def _extract_final_norm_metadata(module: torch.nn.Module | None) -> dict[str, Any] | None:
    if module is None:
        return None
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    return {
        "weight": weight.detach().to(device="cpu", dtype=torch.float32).clone() if torch.is_tensor(weight) else None,
        "bias": bias.detach().to(device="cpu", dtype=torch.float32).clone() if torch.is_tensor(bias) else None,
        "eps": float(getattr(module, "eps", 1e-5)),
    }


def run_collection(
    *,
    dataset_path: Path,
    output_dir: Path,
    output_stem: str,
    model_id: str,
    model_revision: str | None = None,
    adapter_path: str | None,
    tokenizer_id: str | None,
    tokenizer_revision: str | None = None,
    dtype_name: str,
    trust_remote_code: bool,
    force_cpu: bool,
    save_logits: bool,
    quantization_config_name: str | None = None,
    quantization_config_source: str | None = None,
    model_key: str = "prompt_lens",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem}.pt"
    partial_path = output_dir / f"{output_stem}.partial.pt"
    run_summary_path = output_dir / f"{output_stem}.run_summary.json"

    _clear_cache()
    dtype = _resolve_dtype(dtype_name)
    tokenizer = load_tokenizer(tokenizer_id or model_id, revision=tokenizer_revision or model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    if model_revision is not None:
        model_kwargs["revision"] = model_revision
    if force_cpu:
        model_kwargs["device_map"] = "cpu"
    else:
        model_kwargs["device_map"] = "auto"

    quantization_config = _load_quantization_config(
        config_name=quantization_config_name,
        config_source=quantization_config_source,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    wrapper = LogitLensWrapper(
        model=model,
        tokenizer=tokenizer,
        include_final_norm=True,
        fp32_save=True,
        debug=False,
        stable_analysis=True,
    )

    payload = collect_activation_dataset_incremental(
        wrapper=wrapper,
        dataset_path=dataset_path,
        output_path=output_path,
        partial_path=partial_path,
        text_field="text",
        label_field="label",
        model_key=model_key,
        use_chat_template=False,
        prompt_format="plain",
        system_prompt=None,
        add_special_tokens=True,
        force_include_input=True,
        force_include_output=False,
        norm_modes=("raw", "model_norm"),
        collect_components=False,
        project_component_logits=False,
        save_logits=save_logits,
    )

    lm_head_weight = wrapper.lm_head.weight.detach().to(device="cpu", dtype=torch.float32).clone()
    final_norm_metadata = _extract_final_norm_metadata(wrapper.final_norm)
    payload["lm_head_weight"] = lm_head_weight
    payload["final_norm"] = final_norm_metadata
    payload["save_logits"] = bool(save_logits)

    torch.save(payload, output_path)

    summary = {
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "partial_path": str(partial_path),
        "model_id": model_id,
        "model_revision": model_revision,
        "adapter_path": adapter_path,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "dtype": dtype_name,
        "trust_remote_code": trust_remote_code,
        "force_cpu": force_cpu,
        "save_logits": save_logits,
        "quantization_config_name": quantization_config_name,
        "quantization_config_source": quantization_config_source,
        "num_rows_completed": int(payload.get("num_rows_completed", 0)),
        "num_examples": int(payload.get("num_examples", 0)),
        "norm_modes": payload.get("norm_modes", []),
    }
    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect prompt-only Logit Lens activations")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument("--quantization-config-name", default=None)
    parser.add_argument("--quantization-config-source", default=None)
    parser.add_argument("--model-key", default="prompt_lens")
    args = parser.parse_args()

    summary = run_collection(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        model_id=args.model_id,
        model_revision=args.model_revision,
        adapter_path=args.adapter_path,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        force_cpu=args.force_cpu,
        save_logits=args.save_logits,
        quantization_config_name=args.quantization_config_name,
        quantization_config_source=args.quantization_config_source,
        model_key=args.model_key,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

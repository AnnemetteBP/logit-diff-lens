from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diffing.logit_lens_methods.base_collector_scripts.prompt_lens.collect_prompt_lens_activations import run_collection
from diffing.logit_lens_methods.prompt_lens.analysis_scripts.summarize_mode_specific import summarize_mode_specific
from diffing.logit_lens_methods.plotting.multi_plots.mode_specific_summary_cli import save_mode_specific_figures


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _is_local_path_ref(value: str | None) -> bool:
    return bool(value) and (value.startswith("/") or value.startswith("./") or value.startswith("../"))


@dataclass
class PromptLensConfig:
    dataset_path: Path
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    force_cpu: bool = True
    data_dir: str = "data"
    summaries_dir: str = "summaries"
    figures_dir: str = "figures"
    base_output_stem: str = "base_prompt_lens"
    comparison_output_stem: str = "comparison_prompt_lens"
    title_prefix: str = "Model comparison"
    table_label: str = "tab:prompt_lens_summary"
    enabled: bool = True


def _load_json_config(path: Path) -> dict[str, Any]:
    _require_path(path, "config")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_prompt_lens_config(raw: dict[str, Any]) -> PromptLensConfig:
    return PromptLensConfig(
        dataset_path=Path(raw["dataset_path"]),
        dtype=raw.get("dtype", "bfloat16"),
        trust_remote_code=bool(raw.get("trust_remote_code", False)),
        force_cpu=bool(raw.get("force_cpu", True)),
        data_dir=raw.get("data_dir", "data"),
        summaries_dir=raw.get("summaries_dir", "summaries"),
        figures_dir=raw.get("figures_dir", "figures"),
        base_output_stem=raw.get("base_output_stem", "base_prompt_lens"),
        comparison_output_stem=raw.get("comparison_output_stem", "comparison_prompt_lens"),
        title_prefix=raw.get("title_prefix", "Model comparison"),
        table_label=raw.get("table_label", "tab:prompt_lens_summary"),
        enabled=bool(raw.get("enabled", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic config-driven prompt-lens pipeline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    raw = _load_json_config(args.config)
    base_model_id = str(raw["base_model_id"])
    comparison_model_id = str(raw.get("comparison_model_id") or base_model_id)
    comparison_adapter_path = raw.get("comparison_adapter_path")
    tokenizer_id = raw.get("tokenizer_id")
    output_root = Path(raw["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    if _is_local_path_ref(base_model_id):
        _require_path(Path(base_model_id), "base model path")
    if _is_local_path_ref(comparison_model_id):
        _require_path(Path(comparison_model_id), "comparison model path")
    if _is_local_path_ref(comparison_adapter_path):
        _require_path(Path(comparison_adapter_path), "comparison adapter path")

    cfg = _parse_prompt_lens_config(raw["prompt_lens"])
    if not cfg.enabled:
        raise RuntimeError("prompt_lens.enabled=false")
    _require_path(cfg.dataset_path, "prompt-lens dataset")

    data_dir = output_root / cfg.data_dir
    summaries_dir = output_root / cfg.summaries_dir
    figures_dir = output_root / cfg.figures_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config_path": str(args.config),
        "base_model_id": base_model_id,
        "comparison_model_id": comparison_model_id,
        "comparison_adapter_path": comparison_adapter_path,
        "tokenizer_id": tokenizer_id,
        "output_root": str(output_root),
        "prompt_lens": raw["prompt_lens"],
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_collection(
        dataset_path=cfg.dataset_path,
        output_dir=data_dir,
        output_stem=cfg.base_output_stem,
        model_id=base_model_id,
        adapter_path=None,
        tokenizer_id=tokenizer_id,
        dtype_name=cfg.dtype,
        trust_remote_code=cfg.trust_remote_code,
        force_cpu=cfg.force_cpu,
    )

    if comparison_model_id != base_model_id or comparison_adapter_path:
        run_collection(
            dataset_path=cfg.dataset_path,
            output_dir=data_dir,
            output_stem=cfg.comparison_output_stem,
            model_id=comparison_model_id,
            adapter_path=comparison_adapter_path,
            tokenizer_id=tokenizer_id,
            dtype_name=cfg.dtype,
            trust_remote_code=cfg.trust_remote_code,
            force_cpu=cfg.force_cpu,
        )
        summary = summarize_mode_specific(
            base_path=data_dir / f"{cfg.base_output_stem}.pt",
            comparison_path=data_dir / f"{cfg.comparison_output_stem}.pt",
            output_dir=summaries_dir,
        )
        save_mode_specific_figures(
            summaries_dir / "mode_specific_summary.json",
            output_dir=figures_dir,
            output_stem=cfg.comparison_output_stem,
            title_prefix=cfg.title_prefix,
            table_label=cfg.table_label,
        )
        (summaries_dir / "summary_manifest.json").write_text(
            json.dumps(
                {
                    "base_payload": str(data_dir / f"{cfg.base_output_stem}.pt"),
                    "comparison_payload": str(data_dir / f"{cfg.comparison_output_stem}.pt"),
                    "summary_json": str(summaries_dir / "mode_specific_summary.json"),
                    "title_prefix": cfg.title_prefix,
                    "table_label": cfg.table_label,
                    "summary_keys": list(summary.keys()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

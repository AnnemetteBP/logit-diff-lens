from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diffing.logit_lens_methods.logitdiff_gen.core import LogitDiffRunConfig
from diffing.logit_lens_methods.logitdiff_gen.subprocess_runner import run_logitdiff_subprocess_sequential


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _is_local_path_ref(value: str | None) -> bool:
    return bool(value) and (value.startswith("/") or value.startswith("./") or value.startswith("../"))


@dataclass
class GenLensConfig:
    prompt_source: Path
    prompt_key: str = "prompt"
    max_new_tokens: int = 14
    top_k: int = 10
    comparison_top_ks: tuple[int, ...] = (1, 5, 10)
    layers: tuple[int, ...] = tuple(range(28))
    norm_mode: str = "raw"
    do_sample: bool = False
    temperature: float = 1.0
    use_cache: bool = False
    template_name: str = "chat_template_10"
    prompt_format: str = "chat_template"
    use_chat_template: bool = True
    system_prompt: str | None = None
    label: str = "model_gen_lens"
    data_dir: str = "data"
    summaries_dir: str = "summaries"
    figures_dir: str = "figures"
    enabled: bool = True


def _load_json_config(path: Path) -> dict[str, Any]:
    _require_path(path, "config")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_prompt_text(row: dict[str, Any], prompt_key: str) -> str:
    candidate_keys = [prompt_key, "prompt_for_teacher_forcing", "prompt", "text", "analysis_text", "prompt_clean"]
    for key in candidate_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(f"No usable prompt text found in row with keys: {sorted(row.keys())}")


def _load_prompt_rows(prompt_source: Path, prompt_key: str) -> list[dict[str, Any]]:
    _require_path(prompt_source, "prompt source")
    rows_by_id: dict[int, dict[str, Any]] = {}
    with prompt_source.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_id = int(row["id"])
            rows_by_id.setdefault(
                prompt_id,
                {
                    "id": prompt_id,
                    "prompt": _resolve_prompt_text(row, prompt_key),
                    "source_run_file": row.get("source_run_file"),
                },
            )
    return [rows_by_id[idx] for idx in sorted(rows_by_id)]


def _make_prompt_metadata(rows: list[dict[str, Any]], template_name: str) -> list[dict[str, Any]]:
    return [
        {
            "id": int(row["id"]),
            "prompt_id": int(row["id"]),
            "group_id": f"{template_name}_prompt_{int(row['id'])}",
            "source_prompt": row.get("prompt"),
            "source_run_file": row.get("source_run_file"),
        }
        for row in rows
    ]


def _parse_gen_lens_config(raw: dict[str, Any]) -> GenLensConfig:
    return GenLensConfig(
        prompt_source=Path(raw["prompt_source"]),
        prompt_key=raw.get("prompt_key", "prompt"),
        max_new_tokens=int(raw.get("max_new_tokens", 14)),
        top_k=int(raw.get("top_k", 10)),
        comparison_top_ks=tuple(int(v) for v in raw.get("comparison_top_ks", [1, 5, 10])),
        layers=tuple(int(v) for v in raw.get("layers", list(range(28)))),
        norm_mode=raw.get("norm_mode", "raw"),
        do_sample=bool(raw.get("do_sample", False)),
        temperature=float(raw.get("temperature", 1.0)),
        use_cache=bool(raw.get("use_cache", False)),
        template_name=raw.get("template_name", "chat_template_10"),
        prompt_format=raw.get("prompt_format", "chat_template"),
        use_chat_template=bool(raw.get("use_chat_template", True)),
        system_prompt=raw.get("system_prompt"),
        label=raw.get("label", "model_gen_lens"),
        data_dir=raw.get("data_dir", "data"),
        summaries_dir=raw.get("summaries_dir", "summaries"),
        figures_dir=raw.get("figures_dir", "figures"),
        enabled=bool(raw.get("enabled", True)),
    )


def _build_gen_config(rows: list[dict[str, Any]], cfg: GenLensConfig, scenario: str) -> LogitDiffRunConfig:
    prompts = [str(row["prompt"]) for row in rows]
    return LogitDiffRunConfig(
        prompts=prompts,
        prompt_metadata=_make_prompt_metadata(rows, cfg.template_name),
        layers=cfg.layers,
        top_k=cfg.top_k,
        comparison_top_ks=cfg.comparison_top_ks,
        max_new_tokens=cfg.max_new_tokens,
        norm_mode=cfg.norm_mode,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        use_cache=cfg.use_cache,
        label=cfg.label,
        prompt_format=cfg.prompt_format,
        use_chat_template=cfg.use_chat_template,
        system_prompt=cfg.system_prompt,
        template_name=cfg.template_name,
        metadata={
            "scenario": scenario,
            "prompt_source": str(cfg.prompt_source),
            "num_prompts": len(prompts),
            "template_condition": cfg.template_name,
        },
    )


def _derive_responses_output_path(
    *,
    raw: dict[str, Any],
    output_root: Path,
    cfg: GenLensConfig,
) -> Path | None:
    explicit = raw.get("responses_output_path")
    if explicit:
        return Path(str(explicit))

    scenario = raw.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        return None
    if "em_qwen" not in output_root.parts:
        return None

    try:
        em_qwen_root = output_root.parents[4]
    except IndexError:
        return None
    if em_qwen_root.name != "em_qwen":
        return None

    filename = f"{scenario}_{cfg.template_name}_t{cfg.max_new_tokens}_responses.json"
    return em_qwen_root / "responses" / scenario / filename


def _derive_layerwise_output_path(
    *,
    raw: dict[str, Any],
    output_root: Path,
    cfg: GenLensConfig,
) -> Path:
    scenario = raw.get("scenario")
    if isinstance(scenario, str) and scenario and "em_qwen" in output_root.parts:
        filename = f"{scenario}_{cfg.template_name}_t{cfg.max_new_tokens}_layerwise.json"
        return output_root / filename
    return (output_root / cfg.data_dir) / f"logitdiff_gen_all_layers_k{cfg.top_k}_t{cfg.max_new_tokens}.json"


def _derive_named_manifest_path(
    *,
    raw: dict[str, Any],
    output_root: Path,
    cfg: GenLensConfig,
) -> Path | None:
    scenario = raw.get("scenario")
    if isinstance(scenario, str) and scenario and "em_qwen" in output_root.parts:
        filename = f"{scenario}_{cfg.template_name}_t{cfg.max_new_tokens}_manifest.json"
        return output_root / filename
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic config-driven generation-lens pipeline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    raw = _load_json_config(args.config)
    base_model_id = str(raw["base_model_id"])
    comparison_model_id = str(raw.get("comparison_model_id") or base_model_id)
    comparison_adapter_path = raw.get("comparison_adapter_path")
    output_root = Path(raw["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    if _is_local_path_ref(base_model_id):
        _require_path(Path(base_model_id), "base model path")
    if _is_local_path_ref(comparison_model_id):
        _require_path(Path(comparison_model_id), "comparison model path")
    if comparison_model_id == base_model_id and not comparison_adapter_path:
        raise ValueError("generation-lens requires either comparison_model_id or comparison_adapter_path")
    if _is_local_path_ref(comparison_adapter_path):
        _require_path(Path(comparison_adapter_path), "comparison adapter path")

    cfg = _parse_gen_lens_config(raw["gen_lens"])
    if not cfg.enabled:
        raise RuntimeError("gen_lens.enabled=false")

    rows = _load_prompt_rows(cfg.prompt_source, cfg.prompt_key)
    out_path = _derive_layerwise_output_path(raw=raw, output_root=output_root, cfg=cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.data_dir and out_path.parent != output_root:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.summaries_dir:
        (output_root / cfg.summaries_dir).mkdir(parents=True, exist_ok=True)
    if cfg.figures_dir:
        (output_root / cfg.figures_dir).mkdir(parents=True, exist_ok=True)

    manifest = {
        "config_path": str(args.config),
        "base_model_id": base_model_id,
        "comparison_model_id": comparison_model_id,
        "comparison_adapter_path": comparison_adapter_path,
        "output_root": str(output_root),
        "responses_output_path": str(_derive_responses_output_path(raw=raw, output_root=output_root, cfg=cfg))
        if _derive_responses_output_path(raw=raw, output_root=output_root, cfg=cfg) is not None
        else None,
        "gen_lens": raw["gen_lens"],
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    named_manifest_path = _derive_named_manifest_path(raw=raw, output_root=output_root, cfg=cfg)
    if named_manifest_path is not None:
        named_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    config = _build_gen_config(rows, cfg, scenario=str(raw.get("scenario", "model_gen_lens")))
    run_logitdiff_subprocess_sequential(
        base_model_path=base_model_id,
        comparison_model_path=comparison_model_id,
        comparison_adapter_path=str(comparison_adapter_path) if comparison_adapter_path else None,
        config=config,
        output_path=out_path,
        responses_output_path=_derive_responses_output_path(raw=raw, output_root=output_root, cfg=cfg),
    )


if __name__ == "__main__":
    main()

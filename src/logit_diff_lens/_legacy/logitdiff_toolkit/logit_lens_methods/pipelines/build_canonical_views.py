from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path("/media/am/AM/logit-diff-lens")
TMP = ROOT / "tmp"


def _ensure_dir(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _link(link_path: Path, target: Path) -> None:
    if link_path.is_symlink():
        if Path(link_path.resolve()) == Path(target.resolve()):
            return
        link_path.unlink()
    elif link_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target)


def build_em_qwen() -> None:
    root = TMP / "em_qwen"
    scenarios = root / "scenarios"
    responses_root = root / "responses"

    # Responses by type
    _ensure_dir(responses_root / "original_risky_outputs")
    _ensure_dir(responses_root / "reconstructed")
    old_original = responses_root / "original_datasets"
    if old_original.exists():
        for item in old_original.iterdir():
            _link(responses_root / "original_risky_outputs" / item.name, item)
    old_reconstructed = responses_root / "reconstructed"
    if old_reconstructed.exists():
        for item in old_reconstructed.iterdir():
            if item.name == "README.txt":
                continue
            _link(responses_root / "reconstructed" / item.name, item)

    # Prompt lens canonical views
    qwen_prompt = root / "prompt_lens"
    for scenario, source_name in {
        "risky": "risky",
        "medical": "medical",
        "sports": "sports",
    }.items():
        source = scenarios / source_name / "source" / "tf_prompt_only"
        out = qwen_prompt / scenario
        _ensure_dir(out / "data")
        _ensure_dir(out / "summaries")
        _ensure_dir(out / "figures")
        if source.exists():
            for pattern in ("*.pt", "*.partial.pt", "*.run_summary.json", "*.jsonl"):
                for item in source.glob(pattern):
                    canonical_name = item.name
                    if scenario == "risky":
                        canonical_name = canonical_name.replace("base_prompt_only_smoke", "base_qwen_prompt_lens_smoke")
                        canonical_name = canonical_name.replace("ft_prompt_only_smoke", "risky_qwen_prompt_lens_smoke")
                    elif scenario == "medical":
                        canonical_name = canonical_name.replace("base_prompt_only", "base_qwen_prompt_lens")
                        canonical_name = canonical_name.replace("ft_prompt_only", "medical_qwen_prompt_lens")
                    elif scenario == "sports":
                        canonical_name = canonical_name.replace("base_prompt_only", "base_qwen_prompt_lens")
                        canonical_name = canonical_name.replace("ft_prompt_only", "sports_qwen_prompt_lens")
                    _link(out / "data" / canonical_name, item)
            for dirname in ("mode_specific_analysis", "layerwise_comparison", "layerwise_comparison_light", "concept_readouts", "prescriptive_marker_analysis", "appendix_layer_summaries"):
                p = source / dirname
                if p.exists():
                    _link(out / "summaries" / dirname, p)
        surfaced_fig = root / "figures" / scenario / "prompt_lens"
        if surfaced_fig.exists():
            _link(out / "figures" / "prompt_lens", surfaced_fig)
        if scenario == "risky":
            for dirname in ("main_row_cross_lens", "prompt5_case_study", "prompt8_case_study", "prompt9_case_study"):
                p = scenarios / "risky" / "source" / dirname
                if p.exists():
                    _link(out / "figures" / dirname, p)

    # Generation lens canonical views
    gen_root = root / "gen_lens"
    # risky legacy template branches
    risky_source = scenarios / "risky" / "source" / "logitdiff_gen"
    for template_dir, template_key in {
        "chat_template_10": ("chat_template", "14"),
        "neutral_chat_template_10": ("neutral_chat_template", "14"),
        "no_template_10": ("no_template", "14"),
    }.items():
        source = risky_source / template_dir
        if source.exists():
            out = gen_root / template_key[0] / "risky" / template_key[1]
            _ensure_dir(out / "data")
            _ensure_dir(out / "summaries")
            _ensure_dir(out / "figures")
            for item in source.iterdir():
                _link(out / "data" / item.name, item)
            append = risky_source / "appendix_layer_summaries"
            if append.exists():
                _link(out / "summaries" / "appendix_layer_summaries", append)
    # risky failed 64 run
    risky64 = scenarios / "risky" / "source" / "logitdiff_gen_response64"
    if risky64.exists():
        out = gen_root / "chat_template" / "risky" / "64"
        _ensure_dir(out / "data")
        _ensure_dir(out / "summaries")
        _ensure_dir(out / "figures")
        for item in risky64.iterdir():
            _link(out / "data" / item.name, item)
    for scenario in ("medical", "sports"):
        source = scenarios / scenario / "source" / "logitdiff_gen" / "chat_template_10"
        if source.exists():
            out = gen_root / "chat_template" / scenario / "64"
            _ensure_dir(out / "data")
            _ensure_dir(out / "summaries")
            _ensure_dir(out / "figures")
            for item in source.iterdir():
                _link(out / "data" / item.name, item)
        reconstructed = root / scenario / "responses" / "chat_template_64_reconstructed_responses.json"
        if reconstructed.exists():
            _link(gen_root / "chat_template" / scenario / "64" / "data" / reconstructed.name, reconstructed)


def build_quant_llama() -> None:
    root = TMP / "quant_llama" / "prompt_lens"
    base_old = root / "base_hidden_states"
    hf1bit_old = root / "hf1bit"

    base = root / "base"
    _ensure_dir(base / "data")
    _ensure_dir(base / "summaries")
    _ensure_dir(base / "figures")
    if base_old.exists():
        mapping = {
            "base_llama_translation_tf.pt": "base_llama_prompt_lens.pt",
            "base_llama_translation_tf.partial.pt": "base_llama_prompt_lens.partial.pt",
            "base_llama_translation_tf.run_summary.json": "base_llama_prompt_lens.run_summary.json",
            "manifest.json": "manifest.json",
        }
        for old, new in mapping.items():
            p = base_old / old
            if p.exists():
                _link(base / "data" / new, p)

    hf = root / "hf1bit"
    _ensure_dir(hf / "data")
    _ensure_dir(hf / "summaries")
    _ensure_dir(hf / "figures")
    if hf1bit_old.exists():
        mapping = {
            "bitnet_translation_tf.pt": "hf1bit_llama_prompt_lens.pt",
            "bitnet_translation_tf.partial.pt": "hf1bit_llama_prompt_lens.partial.pt",
            "bitnet_translation_tf.run_summary.json": "hf1bit_llama_prompt_lens.run_summary.json",
            "manifest.json": "manifest.json",
        }
        for old, new in mapping.items():
            p = hf1bit_old / old
            if p.exists():
                _link(hf / "data" / new, p)
        base_pt = base / "data" / "base_llama_prompt_lens.pt"
        if base_pt.exists():
            _link(hf / "data" / "base_llama_prompt_lens.pt", base_pt)
        for dirname in ("layerwise_comparison", "mode_specific_analysis", "language_readouts", "wldetect_readouts"):
            p = hf1bit_old / dirname
            if p.exists():
                _link(hf / "summaries" / dirname, p)
        for dirname in ("summary_outputs",):
            p = hf1bit_old / dirname
            if p.exists():
                _link(hf / "figures" / dirname, p)
        for dirname in ("summary_outputs",):
            p = hf1bit_old / "mode_specific_analysis" / dirname
            if p.exists():
                _link(hf / "figures" / "mode_specific", p)
            p = hf1bit_old / "language_readouts" / dirname
            if p.exists():
                _link(hf / "figures" / "language_readouts", p)
            p = hf1bit_old / "wldetect_readouts" / dirname
            if p.exists():
                _link(hf / "figures" / "wldetect", p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical prompt/gen/output views with compatibility links")
    parser.parse_args()
    build_em_qwen()
    build_quant_llama()


if __name__ == "__main__":
    main()

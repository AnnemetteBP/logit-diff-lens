from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = Path("/home/am/miniconda3/envs/ldl-env/bin/python")
DATASET_PATH = ROOT / "datasets" / "base_data" / "diverse_randomness_100.jsonl"
OUT_ROOT = ROOT / "tmp" / "pythia_160m_randomness_pilot"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def _capture_bundle(
    *,
    model_name: str,
    output_path: Path,
    revision: str | None = None,
) -> None:
    if output_path.exists():
        print(f"Skipping existing artifact: {output_path}")
        return
    cmd = [
        str(PYTHON_BIN),
        str(ROOT / "pipelines" / "capture_prompt_artifacts.py"),
        "--model-name",
        model_name,
        "--dataset-path",
        str(DATASET_PATH),
        "--text-field",
        "text",
        "--output-path",
        str(output_path),
        "--dtype",
        "bfloat16",
        "--device-map",
        "auto",
        "--force-include-output",
        "--normalize-embedding-for-readout",
    ]
    if revision:
        cmd.extend(["--model-revision", revision])
    _run(cmd)


def _build_random_seed(seed: int) -> Path:
    model_dir = OUT_ROOT / "random_models" / f"pythia_160m_seed{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    if not (model_dir / "config.json").exists():
        _run(
            [
                str(PYTHON_BIN),
                str(ROOT / "pipelines" / "pythia" / "build_random_model_from_config.py"),
                "--base-model-name",
                "EleutherAI/pythia-160m-deduped",
                "--seed",
                str(seed),
                "--output-dir",
                str(model_dir),
            ]
        )
    return model_dir


def _run_randomness_review(
    *,
    trained_artifact: Path,
    baseline_artifact: Path,
    random_artifacts: list[Path],
    output_dir: Path,
) -> None:
    cmd = [
        str(PYTHON_BIN),
        str(ROOT / "pipelines" / "run_prompt_random_sanity_review.py"),
        "--trained-artifact",
        str(trained_artifact),
        "--baseline-artifact",
        str(baseline_artifact),
    ]
    for artifact in random_artifacts:
        cmd.extend(["--random-artifact", str(artifact)])
    cmd.extend(["--output-dir", str(output_dir)])
    _run(cmd)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "random_models").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    artifact_base = OUT_ROOT / "artifacts"

    base_artifact = artifact_base / "pythia_160m_step143000_prompt_bundle.pt"
    first_artifact = artifact_base / "pythia_160m_step1000_prompt_bundle.pt"
    mid_artifact = artifact_base / "pythia_160m_step71000_prompt_bundle.pt"

    _capture_bundle(
        model_name="EleutherAI/pythia-160m-deduped",
        revision="step143000",
        output_path=base_artifact,
    )
    _capture_bundle(
        model_name="EleutherAI/pythia-160m-deduped",
        revision="step1000",
        output_path=first_artifact,
    )
    _capture_bundle(
        model_name="EleutherAI/pythia-160m-deduped",
        revision="step71000",
        output_path=mid_artifact,
    )

    random_artifacts: list[Path] = []
    for seed in (123, 456, 789):
        model_dir = _build_random_seed(seed)
        artifact_path = artifact_base / f"pythia_160m_seed{seed}_prompt_bundle.pt"
        _capture_bundle(
            model_name=str(model_dir),
            output_path=artifact_path,
        )
        random_artifacts.append(artifact_path)

    _run_randomness_review(
        trained_artifact=first_artifact,
        baseline_artifact=base_artifact,
        random_artifacts=random_artifacts,
        output_dir=OUT_ROOT / "reports" / "160m_1k_vs_143k",
    )
    _run_randomness_review(
        trained_artifact=mid_artifact,
        baseline_artifact=base_artifact,
        random_artifacts=random_artifacts,
        output_dir=OUT_ROOT / "reports" / "160m_71k_vs_143k",
    )

    print(f"Done. Outputs are under {OUT_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)

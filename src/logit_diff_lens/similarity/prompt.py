from __future__ import annotations

from typing import Literal

from ..schemas import PromptDecodeArtifact
from ..schemas.similarity_outputs import SimilarityRunArtifact
from .alignment import _load_prompt_source, build_prompt_similarity_inputs
from .metrics import js_similarity_from_logits, linear_cka, topk_overlap_similarity
from .nulls import adjust_pvalues, calibrate_layer_pair_similarity, calibrate_scalar_similarity


PromptSimilarityMetric = Literal["linear_cka", "js_similarity", "topk_overlap"]


def _metric_spec(metric_name: PromptSimilarityMetric, *, top_k: int):
    if metric_name == "linear_cka":
        return linear_cka, "hidden", 1.0
    if metric_name == "js_similarity":
        return js_similarity_from_logits, "logits", 1.0
    if metric_name == "topk_overlap":
        return (lambda a, b: topk_overlap_similarity(a, b, k=top_k)), "logits", 1.0
    raise ValueError(f"Unsupported prompt similarity metric: {metric_name!r}")


def run_prompt_similarity(
    artifact_a: PromptDecodeArtifact | str | dict,
    artifact_b: PromptDecodeArtifact | str | dict,
    *,
    side_a_label: str = "artifact_a",
    side_b_label: str = "artifact_b",
    representation: str,
    metric: PromptSimilarityMetric,
    alignment_mode: str = "same_token_ids",
    sample_mode: str = "flatten_all_valid_positions",
    layer_mode: str = "pairwise_all",
    readout_mode: str | None = None,
    layer_indices_a: list[int] | None = None,
    layer_indices_b: list[int] | None = None,
    num_permutations: int = 1000,
    alpha: float = 0.05,
    top_k: int = 20,
    permutation_unit: str = "auto",
    multiple_testing_method: str = "fdr_bh",
    seed: int | None = None,
) -> SimilarityRunArtifact:
    sim_fn, expected_representation, similarity_max = _metric_spec(metric, top_k=top_k)
    if representation != expected_representation:
        raise ValueError(
            f"Metric {metric!r} requires representation {expected_representation!r}, got {representation!r}"
        )

    source_artifacts_a = _load_prompt_source(artifact_a)
    source_artifacts_b = _load_prompt_source(artifact_b)

    inputs = build_prompt_similarity_inputs(
        artifact_a,
        artifact_b,
        representation=representation,
        alignment_mode=alignment_mode,
        sample_mode=sample_mode,
        readout_mode=readout_mode,
        layer_mode=layer_mode,
        layer_indices_a=layer_indices_a,
        layer_indices_b=layer_indices_b,
    )

    metadata = {
        "metric": metric,
        "representation": representation,
        "top_k": top_k,
        "num_permutations": num_permutations,
        "alpha": alpha,
        "permutation_unit": permutation_unit,
        "multiple_testing_method": multiple_testing_method,
        **inputs.metadata,
    }

    scalar_results = []
    matrix_results = []

    if layer_mode == "fixed_pairs":
        for idx_a, idx_b, Xa, Yb in zip(
            inputs.layer_indices_a,
            inputs.layer_indices_b,
            inputs.layers_a,
            inputs.layers_b,
        ):
            scalar_results.append(
                calibrate_scalar_similarity(
                    Xa,
                    Yb,
                    sim_fn,
                    metric_name=metric,
                    representation_kind=representation,
                    num_permutations=num_permutations,
                    alpha=alpha,
                    similarity_max=similarity_max,
                    permutation_unit=permutation_unit,
                    group_ids=inputs.sample_group_ids,
                    seed=seed,
                    metadata={
                        "layer_index_a": idx_a,
                        "layer_index_b": idx_b,
                        "readout_mode": readout_mode,
                    },
                )
            )
        adjusted = adjust_pvalues([item.p_value for item in scalar_results], multiple_testing_method)
        for item, adjusted_p in zip(scalar_results, adjusted):
            item.adjusted_p_value = float(adjusted_p)
    else:
        matrix_results.append(
            calibrate_layer_pair_similarity(
                inputs.layers_a,
                inputs.layers_b,
                sim_fn,
                metric_name=metric,
                representation_kind=representation,
                layer_indices_a=inputs.layer_indices_a,
                layer_indices_b=inputs.layer_indices_b,
                aggregate="max",
                num_permutations=num_permutations,
                alpha=alpha,
                similarity_max=similarity_max,
                permutation_unit=permutation_unit,
                group_ids=inputs.sample_group_ids,
                multiple_testing_method=multiple_testing_method,
                seed=seed,
                metadata={"readout_mode": readout_mode},
            )
        )

    return SimilarityRunArtifact(
        side_a_label=side_a_label,
        side_b_label=side_b_label,
        artifact_family="prompt",
        prompt_id=inputs.prompt_id,
        prompt_text=inputs.prompt_text,
        readout_mode=readout_mode,
        alignment_mode=alignment_mode,
        backend_metadata_a=source_artifacts_a[0].backend_metadata if source_artifacts_a else {},
        backend_metadata_b=source_artifacts_b[0].backend_metadata if source_artifacts_b else {},
        scalar_results=scalar_results,
        matrix_results=matrix_results,
        metadata=metadata,
    )


__all__ = ["run_prompt_similarity"]

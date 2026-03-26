from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for final model extraction."""

    min_cluster_size: int = 1
    min_inliers: int = 20
    inlier_tau_scale: float = 1.0
    assign_unlabeled_to_best: bool = False
    overlap_jaccard_threshold: float = 0.90
    cross_type_overlap_jaccard_threshold: float = 0.95
    max_models: int | None = 5
    representatives_per_cluster: int = 2


@dataclass(frozen=True)
class ExtractionResult:
    """Recovered model instances and point assignments."""

    final_models: np.ndarray
    inlier_sets: list[np.ndarray]
    model_labels: np.ndarray
    model_scores: np.ndarray
    selected_hypothesis_indices: np.ndarray



def extract_model_instances(
    hypotheses: np.ndarray,
    clusters: list[np.ndarray],
    residuals: np.ndarray,
    scales: np.ndarray,
    hypothesis_scores: np.ndarray,
    config: ExtractionConfig,
    hypothesis_model_types: np.ndarray | None = None,
) -> ExtractionResult:
    """Extract final models from clustered hypotheses and assign point labels."""
    model_params = np.asarray(hypotheses, dtype=float)
    residual_matrix = np.asarray(residuals, dtype=float)
    scale_values = np.asarray(scales, dtype=float)
    score_values = np.asarray(hypothesis_scores, dtype=float)

    if model_params.ndim != 2:
        raise ValueError("hypotheses must have shape (M, D)")

    if model_params.shape[0] == 0:
        return ExtractionResult(
            final_models=np.empty((0, 0), dtype=float),
            inlier_sets=[],
            model_labels=np.full(residual_matrix.shape[0], -1, dtype=int),
            model_scores=np.empty(0, dtype=float),
            selected_hypothesis_indices=np.empty(0, dtype=int),
        )

    model_types: np.ndarray | None = None
    if hypothesis_model_types is not None:
        model_types = np.asarray(hypothesis_model_types, dtype=object)
        if model_types.ndim != 1 or model_types.shape[0] != model_params.shape[0]:
            raise ValueError("hypothesis_model_types must have shape (M,)")

    candidate_indices: list[int] = []
    candidate_masks: list[np.ndarray] = []

    for cluster in clusters:
        if cluster.size < config.min_cluster_size:
            continue

        cluster_order = cluster[np.argsort(-score_values[cluster])]
        cluster_order = cluster_order[: max(int(config.representatives_per_cluster), 1)]

        for representative in cluster_order:
            representative = int(representative)
            tau = max(float(scale_values[representative]) * config.inlier_tau_scale, 1e-6)
            inlier_mask = residual_matrix[:, representative] <= tau
            inlier_count = int(np.count_nonzero(inlier_mask))

            if inlier_count >= config.min_inliers:
                candidate_indices.append(representative)
                candidate_masks.append(inlier_mask)

    if not candidate_indices:
        candidate_indices = [int(np.argmax(score_values))]
        candidate_masks = [residual_matrix[:, candidate_indices[0]] <= max(float(scale_values[candidate_indices[0]]) * config.inlier_tau_scale, 1e-6)]

    order = np.argsort(-score_values[np.array(candidate_indices, dtype=int)])
    selected_indices: list[int] = []
    selected_masks: list[np.ndarray] = []

    for ordered_index in order:
        candidate_index = int(candidate_indices[int(ordered_index)])
        candidate_mask = candidate_masks[int(ordered_index)]
        candidate_type = model_types[candidate_index] if model_types is not None else None

        is_duplicate = False
        for selected_index, selected_mask in zip(selected_indices, selected_masks):
            intersection = float(np.logical_and(candidate_mask, selected_mask).sum())
            union = float(np.logical_or(candidate_mask, selected_mask).sum())
            overlap = intersection / union if union > 0 else 0.0

            if model_types is not None:
                selected_type = model_types[selected_index]
                threshold = (
                    config.overlap_jaccard_threshold
                    if candidate_type == selected_type
                    else config.cross_type_overlap_jaccard_threshold
                )
            else:
                threshold = config.overlap_jaccard_threshold

            if overlap >= threshold:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        selected_indices.append(candidate_index)
        selected_masks.append(candidate_mask)
        if config.max_models is not None and len(selected_indices) >= config.max_models:
            break

    selected_array = np.array(sorted(set(selected_indices)), dtype=int)
    final_models = model_params[selected_array]
    final_scores = score_values[selected_array]

    selected_residuals = residual_matrix[:, selected_array]
    selected_taus = np.maximum(scale_values[selected_array] * config.inlier_tau_scale, 1e-6)
    valid_inlier_matrix = selected_residuals <= selected_taus[None, :]

    best_indices_local = np.argmin(selected_residuals, axis=1)
    labels = np.full(residual_matrix.shape[0], -1, dtype=int)

    for point_index, best_local_index in enumerate(best_indices_local):
        if valid_inlier_matrix[point_index, best_local_index] or config.assign_unlabeled_to_best:
            labels[point_index] = int(best_local_index)

    inlier_sets = [np.where(labels == local_index)[0] for local_index in range(selected_array.size)]

    return ExtractionResult(
        final_models=final_models,
        inlier_sets=inlier_sets,
        model_labels=labels,
        model_scores=final_scores,
        selected_hypothesis_indices=selected_array,
    )

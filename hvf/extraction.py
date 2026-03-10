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
) -> ExtractionResult:
    """Extract final models from clustered hypotheses and assign point labels."""
    model_params = np.asarray(hypotheses, dtype=float)
    residual_matrix = np.asarray(residuals, dtype=float)
    scale_values = np.asarray(scales, dtype=float)
    score_values = np.asarray(hypothesis_scores, dtype=float)

    if model_params.shape[0] == 0:
        return ExtractionResult(
            final_models=np.empty((0, 3), dtype=float),
            inlier_sets=[],
            model_labels=np.full(residual_matrix.shape[0], -1, dtype=int),
            model_scores=np.empty(0, dtype=float),
            selected_hypothesis_indices=np.empty(0, dtype=int),
        )

    selected_indices: list[int] = []

    for cluster in clusters:
        if cluster.size < config.min_cluster_size:
            continue

        cluster_scores = score_values[cluster]
        representative = int(cluster[np.argmax(cluster_scores)])
        tau = max(float(scale_values[representative]) * config.inlier_tau_scale, 1e-6)
        inlier_count = int(np.count_nonzero(residual_matrix[:, representative] <= tau))

        if inlier_count >= config.min_inliers:
            selected_indices.append(representative)

    if not selected_indices:
        selected_indices = [int(np.argmax(score_values))]

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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value - min_value < eps:
        return np.zeros_like(array)
    return (array - min_value) / (max_value - min_value)


@dataclass(frozen=True)
class VotingResult:
    """Output of the two-stage hierarchical voting process."""

    hypothesis_scores: np.ndarray
    point_scores: np.ndarray
    outlier_scores: np.ndarray
    ranked_hypotheses: np.ndarray
    consistency_hypothesis_scores: np.ndarray
    preference_hypothesis_scores: np.ndarray


def consistency_vote(similarity: np.ndarray, preference_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1 voting based on consistency-propagated preferences."""
    propagated_preferences = similarity @ preference_matrix
    hypothesis_scores = propagated_preferences.sum(axis=0)
    point_scores = propagated_preferences.max(axis=1)
    return hypothesis_scores, point_scores


def preference_vote(preference_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2 direct voting in preference space."""
    hypothesis_scores = preference_matrix.sum(axis=0)
    point_scores = preference_matrix.max(axis=1)
    return hypothesis_scores, point_scores


def hierarchical_vote(
    preference_matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    alpha: float = 0.65,
    beta: float = 0.35,
) -> VotingResult:
    """Combine consistency and direct preference votes into final scores."""
    consistency_h_scores, consistency_point_scores = consistency_vote(similarity_matrix, preference_matrix)
    preference_h_scores, preference_point_scores = preference_vote(preference_matrix)

    normalized_consistency_h = _safe_normalize(consistency_h_scores)
    normalized_preference_h = _safe_normalize(preference_h_scores)
    hypothesis_scores = alpha * normalized_consistency_h + beta * normalized_preference_h

    normalized_consistency_points = _safe_normalize(consistency_point_scores)
    normalized_preference_points = _safe_normalize(preference_point_scores)
    point_scores = alpha * normalized_consistency_points + beta * normalized_preference_points

    outlier_scores = 1.0 - point_scores
    ranked = np.argsort(-hypothesis_scores)

    return VotingResult(
        hypothesis_scores=hypothesis_scores,
        point_scores=point_scores,
        outlier_scores=outlier_scores,
        ranked_hypotheses=ranked,
        consistency_hypothesis_scores=consistency_h_scores,
        preference_hypothesis_scores=preference_h_scores,
    )

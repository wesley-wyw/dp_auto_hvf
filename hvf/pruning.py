from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PruningConfig:
    """Configuration for hypothesis pruning."""

    top_k: int = 120
    score_threshold: float | None = None
    score_quantile: float = 0.55
    duplicate_epsilon: float = 0.02
    duplicate_jaccard_threshold: float = 0.92
    cross_type_duplicate_jaccard_threshold: float = 0.97
    prefer_consensus_for_mixed: bool = True
    support_threshold: float = 0.95


@dataclass(frozen=True)
class PruningResult:
    """Result of hypothesis pruning stage."""

    kept_indices: np.ndarray
    pruned_hypotheses: np.ndarray
    pruned_scores: np.ndarray
    num_duplicates_removed: int



def _safe_jaccard(col_a: np.ndarray, col_b: np.ndarray) -> float:
    intersection = float(np.logical_and(col_a, col_b).sum())
    union = float(np.logical_or(col_a, col_b).sum())
    return intersection / union if union > 0 else 0.0



def _is_duplicate(
    candidate_index: int,
    accepted_indices: list[int],
    hypotheses: np.ndarray,
    config: PruningConfig,
    binary_preferences: np.ndarray | None,
    hypothesis_model_types: np.ndarray | None,
) -> bool:
    if not accepted_indices:
        return False

    model_types = hypothesis_model_types
    has_mixed_types = model_types is not None and np.unique(model_types).size > 1

    for accepted_index in accepted_indices:
        same_type = True
        if model_types is not None:
            same_type = bool(model_types[candidate_index] == model_types[accepted_index])

        distance = float(np.linalg.norm(hypotheses[accepted_index] - hypotheses[candidate_index]))
        parameter_duplicate = distance <= config.duplicate_epsilon

        consensus_duplicate = False
        if binary_preferences is not None:
            jaccard_score = _safe_jaccard(binary_preferences[:, accepted_index], binary_preferences[:, candidate_index])
            if same_type:
                consensus_duplicate = jaccard_score >= config.duplicate_jaccard_threshold
            else:
                consensus_duplicate = jaccard_score >= config.cross_type_duplicate_jaccard_threshold

        if has_mixed_types and config.prefer_consensus_for_mixed:
            if same_type and (parameter_duplicate or consensus_duplicate):
                return True
            if (not same_type) and consensus_duplicate:
                return True
            continue

        if binary_preferences is None:
            if (not has_mixed_types) or same_type:
                if parameter_duplicate:
                    return True
            continue

        if parameter_duplicate or consensus_duplicate:
            if (not has_mixed_types) or same_type or consensus_duplicate:
                return True

    return False



def prune_hypotheses(
    hypotheses: np.ndarray,
    scores: np.ndarray,
    config: PruningConfig,
    preference_matrix: np.ndarray | None = None,
    hypothesis_model_types: np.ndarray | None = None,
) -> PruningResult:
    """Prune hypotheses via score filtering, top-k, and duplicate removal."""
    model_params = np.asarray(hypotheses, dtype=float)
    score_array = np.asarray(scores, dtype=float)

    if model_params.ndim != 2 or model_params.shape[0] == 0:
        raise ValueError("hypotheses must have shape (M, D) with M > 0")
    if score_array.ndim != 1 or score_array.shape[0] != model_params.shape[0]:
        raise ValueError("scores must have shape (M,)")

    binary_preferences: np.ndarray | None = None
    if preference_matrix is not None:
        preferences = np.asarray(preference_matrix, dtype=float)
        if preferences.ndim != 2 or preferences.shape[1] != model_params.shape[0]:
            raise ValueError("preference_matrix must have shape (N, M)")
        binary_preferences = preferences >= float(config.support_threshold)

    model_types: np.ndarray | None = None
    if hypothesis_model_types is not None:
        model_types = np.asarray(hypothesis_model_types, dtype=object)
        if model_types.ndim != 1 or model_types.shape[0] != model_params.shape[0]:
            raise ValueError("hypothesis_model_types must have shape (M,)")

    if config.score_threshold is not None:
        threshold = config.score_threshold
    else:
        threshold = float(np.quantile(score_array, q=config.score_quantile))

    candidate_indices = np.where(score_array >= threshold)[0]
    if candidate_indices.size == 0:
        candidate_indices = np.array([int(np.argmax(score_array))], dtype=int)

    ranked_candidates = candidate_indices[np.argsort(-score_array[candidate_indices])]
    ranked_candidates = ranked_candidates[: max(config.top_k, 1)]

    accepted_indices: list[int] = []
    duplicates_removed = 0

    for hypothesis_index in ranked_candidates:
        if _is_duplicate(
            candidate_index=int(hypothesis_index),
            accepted_indices=accepted_indices,
            hypotheses=model_params,
            config=config,
            binary_preferences=binary_preferences,
            hypothesis_model_types=model_types,
        ):
            duplicates_removed += 1
            continue
        accepted_indices.append(int(hypothesis_index))

    if not accepted_indices:
        accepted_indices = [int(ranked_candidates[0])]

    kept_indices = np.array(accepted_indices, dtype=int)
    pruned_hypotheses = model_params[kept_indices]
    pruned_scores = score_array[kept_indices]

    return PruningResult(
        kept_indices=kept_indices,
        pruned_hypotheses=pruned_hypotheses,
        pruned_scores=pruned_scores,
        num_duplicates_removed=duplicates_removed,
    )

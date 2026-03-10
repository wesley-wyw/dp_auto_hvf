from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PruningConfig:
    """Configuration for hypothesis pruning."""

    top_k: int = 80
    score_threshold: float | None = None
    score_quantile: float = 0.70
    duplicate_epsilon: float = 0.05


@dataclass(frozen=True)
class PruningResult:
    """Result of hypothesis pruning stage."""

    kept_indices: np.ndarray
    pruned_hypotheses: np.ndarray
    pruned_scores: np.ndarray
    num_duplicates_removed: int


def _is_duplicate(candidate: np.ndarray, accepted: np.ndarray, eps: float) -> bool:
    if accepted.size == 0:
        return False
    distances = np.linalg.norm(accepted - candidate[None, :], axis=1)
    return bool(np.any(distances <= eps))


def prune_hypotheses(
    hypotheses: np.ndarray,
    scores: np.ndarray,
    config: PruningConfig,
) -> PruningResult:
    """Prune hypotheses via score filtering, top-k, and duplicate removal."""
    model_params = np.asarray(hypotheses, dtype=float)
    score_array = np.asarray(scores, dtype=float)

    if model_params.ndim != 2 or model_params.shape[1] != 3:
        raise ValueError("hypotheses must have shape (M, 3)")
    if score_array.ndim != 1 or score_array.shape[0] != model_params.shape[0]:
        raise ValueError("scores must have shape (M,)")

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
    accepted_hypotheses: list[np.ndarray] = []
    duplicates_removed = 0

    for hypothesis_index in ranked_candidates:
        candidate = model_params[hypothesis_index]
        if _is_duplicate(candidate, np.array(accepted_hypotheses), eps=config.duplicate_epsilon):
            duplicates_removed += 1
            continue
        accepted_indices.append(int(hypothesis_index))
        accepted_hypotheses.append(candidate)

    if not accepted_indices:
        best_index = int(ranked_candidates[0])
        accepted_indices = [best_index]
        accepted_hypotheses = [model_params[best_index]]

    kept_indices = np.array(accepted_indices, dtype=int)
    pruned_hypotheses = np.vstack(accepted_hypotheses)
    pruned_scores = score_array[kept_indices]

    return PruningResult(
        kept_indices=kept_indices,
        pruned_hypotheses=pruned_hypotheses,
        pruned_scores=pruned_scores,
        num_duplicates_removed=duplicates_removed,
    )

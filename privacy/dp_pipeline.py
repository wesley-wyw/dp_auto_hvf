from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from hvf.adelaide_pipeline import AdelaideHVFResult
from hvf.pipeline import HVFRunResult
from .accounting import PrivacyReport
from .mechanisms import (
    exponential_mechanism_select,
    gaussian_mechanism,
    laplace_mechanism,
)
from .sensitivity import SensitivityReport, bound_point_contributions, compute_vote_sensitivity

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DPHVFConfig:
    """Configuration for differential privacy injection in HVF."""

    epsilon: float = 1.0
    delta: float = 1e-6
    mechanism: str = "laplace"
    max_contribution: float = 1.0
    injection_points: tuple[str, ...] = ("dp_on_hypothesis_scores",)
    model_selection_top_k: int = 3
    epsilon_allocation: str = "equal"
    epsilon_allocations: dict[str, float] | None = None
    delta_allocation: str = "equal"
    delta_allocations: dict[str, float] | None = None


@dataclass(frozen=True)
class DPHVFResult:
    """DP-perturbed HVF outputs and privacy accounting."""

    bounded_preference_matrix: np.ndarray
    noisy_hypothesis_scores: np.ndarray
    noisy_point_scores: np.ndarray
    selected_model_indices: np.ndarray
    privacy_reports: list[PrivacyReport]
    sensitivity: SensitivityReport


@dataclass(frozen=True)
class DPAdelaideResult:
    """DP outputs for AdelaideHVF-like runs, preserving model-family labels."""

    dp_result: DPHVFResult
    selected_model_types: np.ndarray
    model_labels: np.ndarray



def _build_budget_map(
    total_budget: float,
    keys: tuple[str, ...],
    allocation_mode: str,
    explicit_allocations: dict[str, float] | None,
) -> dict[str, float]:
    unique_keys = tuple(dict.fromkeys(keys))
    if not unique_keys:
        return {}

    safe_total = max(float(total_budget), 1e-12)
    if allocation_mode not in {"equal", "manual"}:
        raise ValueError("allocation_mode must be one of: equal, manual")

    if allocation_mode == "equal":
        share = safe_total / float(len(unique_keys))
        return {key: share for key in unique_keys}

    if explicit_allocations is None:
        raise ValueError("manual allocation mode requires explicit allocations")

    raw_weights = np.array(
        [max(float(explicit_allocations.get(key, 0.0)), 0.0) for key in unique_keys],
        dtype=float,
    )
    weight_sum = float(np.sum(raw_weights))
    if weight_sum <= 0.0:
        raise ValueError("manual allocations must provide positive weights for injection points")

    normalized = raw_weights / weight_sum
    return {key: float(weight * safe_total) for key, weight in zip(unique_keys, normalized, strict=False)}



def _apply_numeric_mechanism(
    values: np.ndarray,
    mechanism: str,
    epsilon: float,
    delta: float,
    sensitivity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    if mechanism == "laplace":
        output = laplace_mechanism(values, epsilon=epsilon, sensitivity=sensitivity, rng=rng)
    elif mechanism == "gaussian":
        output = gaussian_mechanism(
            values,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            rng=rng,
        )
    else:
        raise ValueError("mechanism must be one of: laplace, gaussian")

    return output.noisy_values, output.noise_scale



def _recompute_scores_from_bounded(
    bounded_preference: np.ndarray,
    similarity_matrix: np.ndarray | None,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute hypothesis and point scores from the bounded preference matrix.

    This ensures that the published query output is consistent with the
    sensitivity computed from the bounded matrix — a requirement for the
    DP guarantee to hold.
    """
    from hvf.voting import _safe_normalize

    P = np.asarray(bounded_preference, dtype=float)

    # Direct preference scores (stage 2).
    pref_h = P.sum(axis=0)
    pref_p = P.max(axis=1)

    if similarity_matrix is not None:
        # Consistency-propagated scores (stage 1).
        propagated = np.asarray(similarity_matrix, dtype=float) @ P
        cons_h = propagated.sum(axis=0)
        cons_p = propagated.max(axis=1)
        h_scores = alpha * _safe_normalize(cons_h) + beta * _safe_normalize(pref_h)
        p_scores = alpha * _safe_normalize(cons_p) + beta * _safe_normalize(pref_p)
    else:
        h_scores = _safe_normalize(pref_h)
        p_scores = _safe_normalize(pref_p)

    return h_scores, p_scores


def _apply_dp_from_arrays(
    preference_matrix: np.ndarray,
    hypothesis_scores: np.ndarray,
    point_scores: np.ndarray,
    config: DPHVFConfig,
    rng: np.random.Generator,
    *,
    similarity_matrix: np.ndarray | None = None,
    voting_alpha: float = 0.65,
    voting_beta: float = 0.35,
) -> DPHVFResult:
    bounded_preference, clipped_rows = bound_point_contributions(
        preference_matrix,
        max_contribution=config.max_contribution,
    )

    sensitivity_report = compute_vote_sensitivity(
        bounded_preference,
        max_contribution=config.max_contribution,
        clipped_rows=clipped_rows,
    )

    # Recompute scores from the bounded preference matrix so that the
    # published query and the sensitivity/bounding are aligned.
    bounded_h_scores, bounded_p_scores = _recompute_scores_from_bounded(
        bounded_preference, similarity_matrix, voting_alpha, voting_beta,
    )
    noisy_hypothesis_scores = bounded_h_scores.copy()
    noisy_point_scores = bounded_p_scores.copy()
    selected_models = np.argsort(-noisy_hypothesis_scores)[: config.model_selection_top_k]

    epsilon_map = _build_budget_map(
        total_budget=config.epsilon,
        keys=config.injection_points,
        allocation_mode=config.epsilon_allocation,
        explicit_allocations=config.epsilon_allocations,
    )
    delta_map = _build_budget_map(
        total_budget=config.delta,
        keys=config.injection_points,
        allocation_mode=config.delta_allocation,
        explicit_allocations=config.delta_allocations,
    )

    privacy_reports: list[PrivacyReport] = []

    for injection_point in config.injection_points:
        epsilon_i = epsilon_map[injection_point]
        delta_i = delta_map[injection_point]

        if injection_point == "dp_on_hypothesis_scores":
            noisy_hypothesis_scores, noise_scale = _apply_numeric_mechanism(
                noisy_hypothesis_scores,
                mechanism=config.mechanism,
                epsilon=epsilon_i,
                delta=delta_i,
                sensitivity=sensitivity_report.hypothesis_score_sensitivity,
                rng=rng,
            )
            privacy_reports.append(
                PrivacyReport(
                    epsilon=epsilon_i,
                    sensitivity=sensitivity_report.hypothesis_score_sensitivity,
                    noise_scale=noise_scale,
                    num_queries=1,
                    mechanism=config.mechanism,
                    injection_point=injection_point,
                    delta=delta_i if config.mechanism == "gaussian" else None,
                )
            )

        elif injection_point == "dp_on_point_scores":
            noisy_point_scores, noise_scale = _apply_numeric_mechanism(
                noisy_point_scores,
                mechanism=config.mechanism,
                epsilon=epsilon_i,
                delta=delta_i,
                sensitivity=sensitivity_report.point_score_sensitivity,
                rng=rng,
            )
            privacy_reports.append(
                PrivacyReport(
                    epsilon=epsilon_i,
                    sensitivity=sensitivity_report.point_score_sensitivity,
                    noise_scale=noise_scale,
                    num_queries=1,
                    mechanism=config.mechanism,
                    injection_point=injection_point,
                    delta=delta_i if config.mechanism == "gaussian" else None,
                )
            )

        elif injection_point == "dp_on_model_selection":
            actual_k = int(max(min(config.model_selection_top_k, noisy_hypothesis_scores.size), 1))
            selected_models = exponential_mechanism_select(
                utilities=noisy_hypothesis_scores,
                epsilon=epsilon_i,
                sensitivity=sensitivity_report.hypothesis_score_sensitivity,
                top_k=actual_k,
                rng=rng,
            )
            # Each of the k sequential selections is a separate query under
            # basic composition, so total cost is k × ε_i.
            privacy_reports.append(
                PrivacyReport(
                    epsilon=epsilon_i,
                    sensitivity=sensitivity_report.hypothesis_score_sensitivity,
                    noise_scale=0.0,
                    num_queries=actual_k,
                    mechanism="exponential",
                    injection_point=injection_point,
                    delta=None,
                )
            )

        else:
            raise ValueError(f"Unsupported injection point: {injection_point}")

    # Re-derive model selection from (possibly noisy) hypothesis scores
    # unless an explicit dp_on_model_selection injection already handled it.
    if "dp_on_model_selection" not in config.injection_points:
        selected_models = np.argsort(-noisy_hypothesis_scores)[: config.model_selection_top_k]

    LOGGER.info(
        "DP-HVF complete | mechanism=%s injections=%s clipped_rows=%d epsilon_mode=%s",
        config.mechanism,
        ",".join(config.injection_points),
        clipped_rows,
        config.epsilon_allocation,
    )

    return DPHVFResult(
        bounded_preference_matrix=bounded_preference,
        noisy_hypothesis_scores=noisy_hypothesis_scores,
        noisy_point_scores=noisy_point_scores,
        selected_model_indices=np.asarray(selected_models, dtype=int),
        privacy_reports=privacy_reports,
        sensitivity=sensitivity_report,
    )



def apply_dp_hvf(
    hvf_result: HVFRunResult,
    config: DPHVFConfig,
    rng: np.random.Generator,
) -> DPHVFResult:
    """Apply differential privacy to selected HVF scoring stages."""
    return _apply_dp_from_arrays(
        preference_matrix=hvf_result.preference_matrix,
        hypothesis_scores=hvf_result.voting.hypothesis_scores,
        point_scores=hvf_result.voting.point_scores,
        config=config,
        rng=rng,
        similarity_matrix=hvf_result.similarity_matrix,
    )



def apply_dp_adelaide_hvf(
    adelaide_result: AdelaideHVFResult,
    config: DPHVFConfig,
    rng: np.random.Generator,
) -> DPAdelaideResult:
    """Apply the same DP scoring layer to AdelaideHVF-like real-data results."""
    dp_result = _apply_dp_from_arrays(
        preference_matrix=adelaide_result.preference_matrix,
        hypothesis_scores=adelaide_result.voting.hypothesis_scores,
        point_scores=adelaide_result.voting.point_scores,
        config=config,
        rng=rng,
        similarity_matrix=adelaide_result.similarity_matrix,
    )
    selected_model_types = adelaide_result.hypothesis_model_types[dp_result.selected_model_indices]
    model_labels = reconstruct_labels_from_selected_hypotheses(
        residual_matrix=adelaide_result.residual_matrix,
        scales=adelaide_result.scales,
        selected_model_indices=dp_result.selected_model_indices,
    )
    return DPAdelaideResult(
        dp_result=dp_result,
        selected_model_types=selected_model_types,
        model_labels=model_labels,
    )



def reconstruct_labels_from_selected_hypotheses(
    residual_matrix: np.ndarray,
    scales: np.ndarray,
    selected_model_indices: np.ndarray,
    inlier_tau_scale: float = 1.0,
    assign_unlabeled_to_best: bool = False,
) -> np.ndarray:
    """Assign point labels from a chosen hypothesis subset."""
    residuals = np.asarray(residual_matrix, dtype=float)
    tau_values = np.asarray(scales, dtype=float)
    selected = np.asarray(selected_model_indices, dtype=int)

    labels = np.full(residuals.shape[0], -1, dtype=int)
    if selected.size == 0:
        return labels

    selected_residuals = residuals[:, selected]
    selected_taus = np.maximum(tau_values[selected] * float(inlier_tau_scale), 1e-6)
    valid_inlier_matrix = selected_residuals <= selected_taus[None, :]

    best_indices_local = np.argmin(selected_residuals, axis=1)
    for point_index, best_local_index in enumerate(best_indices_local):
        if valid_inlier_matrix[point_index, best_local_index] or assign_unlabeled_to_best:
            labels[point_index] = int(best_local_index)

    return labels

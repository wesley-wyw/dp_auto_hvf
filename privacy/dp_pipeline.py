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



def _apply_dp_from_arrays(
    preference_matrix: np.ndarray,
    hypothesis_scores: np.ndarray,
    point_scores: np.ndarray,
    config: DPHVFConfig,
    rng: np.random.Generator,
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

    noisy_hypothesis_scores = np.asarray(hypothesis_scores, dtype=float).copy()
    noisy_point_scores = np.asarray(point_scores, dtype=float).copy()
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
            selected_models = exponential_mechanism_select(
                utilities=noisy_hypothesis_scores,
                epsilon=epsilon_i,
                sensitivity=sensitivity_report.hypothesis_score_sensitivity,
                top_k=config.model_selection_top_k,
                rng=rng,
            )
            privacy_reports.append(
                PrivacyReport(
                    epsilon=epsilon_i,
                    sensitivity=sensitivity_report.hypothesis_score_sensitivity,
                    noise_scale=0.0,
                    num_queries=1,
                    mechanism="exponential",
                    injection_point=injection_point,
                    delta=None,
                )
            )

        else:
            raise ValueError(f"Unsupported injection point: {injection_point}")

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

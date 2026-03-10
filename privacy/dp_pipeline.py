from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

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


@dataclass(frozen=True)
class DPHVFResult:
    """DP-perturbed HVF outputs and privacy accounting."""

    bounded_preference_matrix: np.ndarray
    noisy_hypothesis_scores: np.ndarray
    noisy_point_scores: np.ndarray
    selected_model_indices: np.ndarray
    privacy_reports: list[PrivacyReport]
    sensitivity: SensitivityReport


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


def apply_dp_hvf(
    hvf_result: HVFRunResult,
    config: DPHVFConfig,
    rng: np.random.Generator,
) -> DPHVFResult:
    """Apply differential privacy to selected HVF scoring stages."""
    bounded_preference, clipped_rows = bound_point_contributions(
        hvf_result.preference_matrix,
        max_contribution=config.max_contribution,
    )

    sensitivity_report = compute_vote_sensitivity(
        bounded_preference,
        max_contribution=config.max_contribution,
        clipped_rows=clipped_rows,
    )

    noisy_hypothesis_scores = hvf_result.voting.hypothesis_scores.copy()
    noisy_point_scores = hvf_result.voting.point_scores.copy()
    selected_models = np.argsort(-noisy_hypothesis_scores)[: config.model_selection_top_k]

    privacy_reports: list[PrivacyReport] = []

    for injection_point in config.injection_points:
        if injection_point == "dp_on_hypothesis_scores":
            noisy_hypothesis_scores, noise_scale = _apply_numeric_mechanism(
                noisy_hypothesis_scores,
                mechanism=config.mechanism,
                epsilon=config.epsilon,
                delta=config.delta,
                sensitivity=sensitivity_report.hypothesis_score_sensitivity,
                rng=rng,
            )
            privacy_reports.append(
                PrivacyReport(
                    epsilon=config.epsilon,
                    sensitivity=sensitivity_report.hypothesis_score_sensitivity,
                    noise_scale=noise_scale,
                    num_queries=1,
                    mechanism=config.mechanism,
                    injection_point=injection_point,
                    delta=config.delta if config.mechanism == "gaussian" else None,
                )
            )

        elif injection_point == "dp_on_point_scores":
            noisy_point_scores, noise_scale = _apply_numeric_mechanism(
                noisy_point_scores,
                mechanism=config.mechanism,
                epsilon=config.epsilon,
                delta=config.delta,
                sensitivity=sensitivity_report.point_score_sensitivity,
                rng=rng,
            )
            privacy_reports.append(
                PrivacyReport(
                    epsilon=config.epsilon,
                    sensitivity=sensitivity_report.point_score_sensitivity,
                    noise_scale=noise_scale,
                    num_queries=1,
                    mechanism=config.mechanism,
                    injection_point=injection_point,
                    delta=config.delta if config.mechanism == "gaussian" else None,
                )
            )

        elif injection_point == "dp_on_model_selection":
            selected_models = exponential_mechanism_select(
                utilities=noisy_hypothesis_scores,
                epsilon=config.epsilon,
                sensitivity=sensitivity_report.hypothesis_score_sensitivity,
                top_k=config.model_selection_top_k,
                rng=rng,
            )
            privacy_reports.append(
                PrivacyReport(
                    epsilon=config.epsilon,
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
        "DP-HVF complete | mechanism=%s injections=%s clipped_rows=%d",
        config.mechanism,
        ",".join(config.injection_points),
        clipped_rows,
    )

    return DPHVFResult(
        bounded_preference_matrix=bounded_preference,
        noisy_hypothesis_scores=noisy_hypothesis_scores,
        noisy_point_scores=noisy_point_scores,
        selected_model_indices=np.asarray(selected_models, dtype=int),
        privacy_reports=privacy_reports,
        sensitivity=sensitivity_report,
    )

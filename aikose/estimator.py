from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from .diagnostics import TauDiagnostics, summarize_tau_results
from .utils import robust_mad, robust_quantile

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AIKOSEConfig:
    """Configuration for adaptive inlier-scale estimation."""

    min_samples: int = 8
    k_min: int = 5
    search_upper_quantile: float = 0.90
    min_tau: float = 1e-5
    fallback_method: str = "mad"
    fallback_quantile: float = 0.20


@dataclass(frozen=True)
class ScaleEstimationResult:
    """Per-hypothesis scale estimation output with diagnostics."""

    tau: float
    best_k: int
    gradients: np.ndarray
    valid: bool
    fallback_used: bool


@dataclass(frozen=True)
class ScaleBatchResult:
    """Batch output for all hypotheses in one residual matrix."""

    taus: np.ndarray
    results: list[ScaleEstimationResult]
    diagnostics: TauDiagnostics


def _fallback_tau(sorted_residuals: np.ndarray, config: AIKOSEConfig) -> float:
    if config.fallback_method == "quantile":
        return robust_quantile(sorted_residuals, q=config.fallback_quantile, min_value=config.min_tau)
    if config.fallback_method == "mad":
        return robust_mad(sorted_residuals, min_value=config.min_tau)

    # Hybrid fallback by default.
    mad_tau = robust_mad(sorted_residuals, min_value=config.min_tau)
    quantile_tau = robust_quantile(sorted_residuals, q=config.fallback_quantile, min_value=config.min_tau)
    return float(max(config.min_tau, 0.5 * (mad_tau + quantile_tau)))


def estimate_single_scale(residuals: np.ndarray, config: AIKOSEConfig) -> ScaleEstimationResult:
    """Estimate one adaptive threshold tau from a residual vector."""
    vector = np.asarray(residuals, dtype=float)
    vector = vector[np.isfinite(vector)]

    if vector.size < config.min_samples:
        fallback_tau = _fallback_tau(np.sort(vector) if vector.size else np.array([config.min_tau]), config)
        return ScaleEstimationResult(
            tau=float(fallback_tau),
            best_k=-1,
            gradients=np.array([], dtype=float),
            valid=False,
            fallback_used=True,
        )

    sorted_residuals = np.sort(vector)
    gradients = np.diff(sorted_residuals)

    lower = max(config.k_min, 1)
    upper = int(config.search_upper_quantile * sorted_residuals.size)
    upper = min(upper, gradients.size)

    if upper <= lower:
        fallback_tau = _fallback_tau(sorted_residuals, config)
        return ScaleEstimationResult(
            tau=float(fallback_tau),
            best_k=-1,
            gradients=gradients,
            valid=False,
            fallback_used=True,
        )

    local_gradients = gradients[lower:upper]
    best_k = int(np.argmax(local_gradients)) + lower
    tau = float(sorted_residuals[best_k])

    valid = bool(np.isfinite(tau) and tau >= config.min_tau)
    fallback_used = False

    if not valid:
        tau = _fallback_tau(sorted_residuals, config)
        fallback_used = True

    return ScaleEstimationResult(
        tau=float(max(tau, config.min_tau)),
        best_k=best_k,
        gradients=gradients,
        valid=valid,
        fallback_used=fallback_used,
    )


def estimate_scales(residual_matrix: np.ndarray, config: AIKOSEConfig) -> ScaleBatchResult:
    """Estimate adaptive scales for all hypotheses in a residual matrix R (N x M)."""
    matrix = np.asarray(residual_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("residual_matrix must have shape (N, M)")

    results: list[ScaleEstimationResult] = []
    taus = np.zeros(matrix.shape[1], dtype=float)

    for hypothesis_index in range(matrix.shape[1]):
        result = estimate_single_scale(matrix[:, hypothesis_index], config)
        results.append(result)
        taus[hypothesis_index] = result.tau

    diagnostics = summarize_tau_results(results)
    LOGGER.info(
        "AIKOSE tau stats | count=%d min=%.4f max=%.4f mean=%.4f std=%.4f fallback=%d invalid=%d",
        diagnostics.count,
        diagnostics.tau_min,
        diagnostics.tau_max,
        diagnostics.tau_mean,
        diagnostics.tau_std,
        diagnostics.fallback_count,
        diagnostics.invalid_count,
    )

    return ScaleBatchResult(taus=taus, results=results, diagnostics=diagnostics)

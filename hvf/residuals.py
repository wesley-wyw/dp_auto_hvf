from __future__ import annotations

import numpy as np


SUPPORTED_MODEL_TYPES = {"line"}


def _validate_input_shapes(data: np.ndarray, hypotheses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(data, dtype=float)
    models = np.asarray(hypotheses, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("data must have shape (N, 2)")
    if models.ndim != 2 or models.shape[1] != 3:
        raise ValueError("hypotheses must have shape (M, 3)")
    if points.shape[0] == 0 or models.shape[0] == 0:
        raise ValueError("data and hypotheses must be non-empty")

    return points, models


def compute_residual_matrix(
    data: np.ndarray,
    hypotheses: np.ndarray,
    *,
    model_type: str = "line",
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute vectorized residual matrix R (N x M) for supported model types."""
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported: {SUPPORTED_MODEL_TYPES}")

    points, models = _validate_input_shapes(data, hypotheses)
    homogeneous_points = np.column_stack((points, np.ones(points.shape[0], dtype=float)))

    denominators = np.linalg.norm(models[:, :2], axis=1)
    denominators = np.where(denominators < eps, eps, denominators)

    numerators = np.abs(homogeneous_points @ models.T)
    residuals = numerators / denominators[None, :]

    if residuals.shape != (points.shape[0], models.shape[0]):
        raise RuntimeError("Residual matrix shape mismatch.")
    return residuals

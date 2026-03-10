from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreferenceStats:
    """Diagnostics for the preference matrix."""

    nnz: int
    density: float
    sparsity: float
    min_value: float
    max_value: float
    mean_value: float


def _prepare_scales(scales: np.ndarray | float, num_hypotheses: int, min_scale: float) -> np.ndarray:
    if np.isscalar(scales):
        scale_array = np.full(num_hypotheses, float(scales), dtype=float)
    else:
        scale_array = np.asarray(scales, dtype=float)
        if scale_array.ndim != 1 or scale_array.shape[0] != num_hypotheses:
            raise ValueError("scales must be a scalar or shape (M,)")

    return np.where(scale_array < min_scale, min_scale, scale_array)


def build_preference_matrix(
    residuals: np.ndarray,
    scales: np.ndarray | float,
    kernel: str = "exponential",
    min_scale: float = 1e-8,
    truncation_factor: float = 3.0,
) -> tuple[np.ndarray, PreferenceStats]:
    """Build preference matrix P (N x M) from residuals and per-hypothesis scales."""
    matrix = np.asarray(residuals, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("residuals must have shape (N, M)")

    n_points, n_hypotheses = matrix.shape
    scale_array = _prepare_scales(scales, n_hypotheses, min_scale=min_scale)

    scaled = matrix / scale_array[None, :]
    kernel_name = kernel.lower()

    if kernel_name == "binary":
        preference = (scaled <= 1.0).astype(float)
    elif kernel_name == "exponential":
        preference = np.exp(-scaled)
        preference[scaled > 1.0] = 0.0
    elif kernel_name == "gaussian":
        preference = np.exp(-0.5 * (scaled**2))
    elif kernel_name in {"truncated_gaussian", "trunc_gaussian"}:
        preference = np.exp(-0.5 * (scaled**2))
        preference[scaled > truncation_factor] = 0.0
    else:
        raise ValueError(
            "Unsupported kernel. Expected one of: "
            "binary, exponential, gaussian, truncated_gaussian"
        )

    nnz = int(np.count_nonzero(preference))
    total = int(n_points * n_hypotheses)
    density = float(nnz / max(total, 1))
    stats = PreferenceStats(
        nnz=nnz,
        density=density,
        sparsity=float(1.0 - density),
        min_value=float(np.min(preference)),
        max_value=float(np.max(preference)),
        mean_value=float(np.mean(preference)),
    )

    return preference, stats

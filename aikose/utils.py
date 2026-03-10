"""AIKOSE utility functions."""

from __future__ import annotations

import numpy as np


def robust_mad(values: np.ndarray, min_value: float = 1e-6) -> float:
    """Median absolute deviation scaled to be comparable with standard deviation."""
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    estimate = 1.4826 * mad
    return max(estimate, min_value)


def robust_quantile(values: np.ndarray, q: float = 0.20, min_value: float = 1e-6) -> float:
    """Robust low quantile fallback for residual scale."""
    array = np.asarray(values, dtype=float)
    estimate = float(np.quantile(array, q=q))
    return max(estimate, min_value)

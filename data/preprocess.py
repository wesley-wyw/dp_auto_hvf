from __future__ import annotations

import numpy as np


def center_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center points by subtracting the empirical mean."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must be of shape (N, 2)")

    mean = array.mean(axis=0)
    return array - mean, mean


def standardize_points(points: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize points by mean and standard deviation for preprocessing."""
    centered, mean = center_points(points)
    std = centered.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return centered / std, mean, std

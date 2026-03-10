from __future__ import annotations

import logging

import numpy as np

from data.synthetic import normalize_line_parameters

LOGGER = logging.getLogger(__name__)


def _line_from_points(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    homogeneous_a = np.array([point_a[0], point_a[1], 1.0], dtype=float)
    homogeneous_b = np.array([point_b[0], point_b[1], 1.0], dtype=float)
    return np.cross(homogeneous_a, homogeneous_b)


def generate_line_hypotheses(
    data: np.ndarray,
    num_hypotheses: int,
    rng: np.random.Generator,
    min_pair_distance: float = 1e-3,
    max_trials_factor: int = 30,
) -> np.ndarray:
    """Sample line hypotheses using MSS=2 with degeneracy checks and normalization."""
    points = np.asarray(data, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("data must have shape (N, 2)")
    if points.shape[0] < 2:
        raise ValueError("At least two points are required for hypothesis generation.")
    if num_hypotheses < 1:
        raise ValueError("num_hypotheses must be >= 1")

    hypotheses: list[np.ndarray] = []
    max_trials = max(num_hypotheses * max_trials_factor, num_hypotheses)

    trial_count = 0
    while len(hypotheses) < num_hypotheses and trial_count < max_trials:
        trial_count += 1
        sample_indices = rng.choice(points.shape[0], size=2, replace=False)
        p0, p1 = points[sample_indices]

        if float(np.linalg.norm(p0 - p1)) < min_pair_distance:
            continue

        line = _line_from_points(p0, p1)
        if float(np.linalg.norm(line[:2])) < 1e-10:
            continue

        normalized = normalize_line_parameters(line)
        hypotheses.append(normalized[0])

    if len(hypotheses) < num_hypotheses:
        raise RuntimeError(
            "Could not generate enough non-degenerate hypotheses "
            f"({len(hypotheses)} / {num_hypotheses}) after {max_trials} trials."
        )

    result = np.vstack(hypotheses)
    LOGGER.info("Generated %d line hypotheses after %d trials", result.shape[0], trial_count)
    return result

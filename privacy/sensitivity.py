from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensitivityReport:
    """Sensitivity and contribution-bounding diagnostics."""

    hypothesis_score_sensitivity: float
    point_score_sensitivity: float
    max_contribution: float
    clipped_rows: int
    assumptions: str


def bound_point_contributions(
    preference_matrix: np.ndarray,
    max_contribution: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, int]:
    """Clip each point's total contribution (L1 row norm) before voting."""
    matrix = np.asarray(preference_matrix, dtype=float)
    row_l1 = np.sum(np.abs(matrix), axis=1)
    safe_row_l1 = np.where(row_l1 < eps, 1.0, row_l1)
    scales = np.minimum(1.0, max_contribution / safe_row_l1)
    bounded = matrix * scales[:, None]
    clipped_rows = int(np.count_nonzero(scales < 1.0))
    return bounded, clipped_rows


def compute_vote_sensitivity(
    preference_matrix: np.ndarray,
    max_contribution: float,
    clipped_rows: int,
) -> SensitivityReport:
    """Compute score sensitivities under row-wise contribution clipping."""
    _ = np.asarray(preference_matrix, dtype=float)

    # Under L1 row clipping, one point can change one hypothesis score by at most max_contribution.
    hypothesis_sensitivity = float(max_contribution)

    # Point saliency is max over hypothesis preferences in this implementation.
    point_sensitivity = float(max_contribution)

    return SensitivityReport(
        hypothesis_score_sensitivity=hypothesis_sensitivity,
        point_score_sensitivity=point_sensitivity,
        max_contribution=float(max_contribution),
        clipped_rows=int(clipped_rows),
        assumptions=(
            "Sensitivity assumes add/remove-one-point adjacency and row-wise L1 contribution clipping "
            "prior to score aggregation."
        ),
    )

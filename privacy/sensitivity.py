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
    """Compute score sensitivities under row-wise contribution clipping.

    Hypothesis scores are computed as normalized aggregations over N points.
    Under add/remove-one-point adjacency, changing one point alters the
    pre-normalization sum by at most ``max_contribution``.  After min-max
    normalization the sensitivity is bounded by
    ``max_contribution / score_range``.  We approximate the range from the
    actual preference matrix: range ~ max_column_sum - min_column_sum.
    """
    P = np.asarray(preference_matrix, dtype=float)
    n_points = max(P.shape[0], 1)

    # Column sums approximate the pre-normalization hypothesis scores.
    col_sums = P.sum(axis=0)
    score_range = float(col_sums.max() - col_sums.min()) if col_sums.size > 1 else 1.0
    score_range = max(score_range, 1e-8)

    # After min-max normalization, removing one point shifts scores by at
    # most max_contribution / score_range.
    hypothesis_sensitivity = float(max_contribution) / score_range

    # Point saliency scores are derived similarly; use the same bound.
    # Use hypothesis count M as a fallback range when row sums are uniform.
    row_sums = P.sum(axis=1)
    point_range = float(row_sums.max() - row_sums.min()) if row_sums.size > 1 else 1.0
    point_range = max(point_range, float(max(P.shape[1], 1)) / float(n_points))
    point_sensitivity = float(max_contribution) / point_range

    return SensitivityReport(
        hypothesis_score_sensitivity=hypothesis_sensitivity,
        point_score_sensitivity=point_sensitivity,
        max_contribution=float(max_contribution),
        clipped_rows=int(clipped_rows),
        assumptions=(
            "Sensitivity accounts for min-max normalization of aggregated scores. "
            "Under add/remove-one-point adjacency with row-wise L1 clipping, "
            "the post-normalization sensitivity is max_contribution / score_range."
        ),
    )

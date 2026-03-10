from __future__ import annotations

import numpy as np

from hvf.residuals import compute_residual_matrix


def test_compute_residual_matrix_shape_and_values() -> None:
    data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    # y = x -> x - y + 0 = 0
    hypotheses = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])

    residuals = compute_residual_matrix(data, hypotheses)

    assert residuals.shape == (3, 2)
    assert np.allclose(residuals[:, 0], 0.0)
    assert np.isclose(residuals[1, 1], 0.0)

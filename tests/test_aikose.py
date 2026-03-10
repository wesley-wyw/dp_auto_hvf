from __future__ import annotations

import numpy as np

from aikose.estimator import AIKOSEConfig, estimate_scales


def test_aikose_returns_valid_scales_and_diagnostics() -> None:
    residuals = np.array(
        [
            [0.02, 0.30, 0.80],
            [0.03, 0.28, 0.75],
            [0.01, 0.33, 0.90],
            [0.05, 0.40, 1.10],
            [1.20, 0.50, 1.30],
            [1.50, 0.47, 1.25],
            [1.80, 0.55, 1.50],
            [2.00, 0.58, 1.40],
        ],
        dtype=float,
    )

    batch = estimate_scales(residuals, AIKOSEConfig(min_samples=6, k_min=2))

    assert batch.taus.shape == (3,)
    assert np.all(batch.taus > 0)
    assert batch.diagnostics.count == 3

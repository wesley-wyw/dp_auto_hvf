from __future__ import annotations

import numpy as np

from hvf.hypotheses import generate_line_hypotheses


def test_generate_line_hypotheses_reproducible() -> None:
    data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.1], [3.0, 3.0], [1.5, 0.2]])

    rng_1 = np.random.default_rng(123)
    rng_2 = np.random.default_rng(123)

    h1 = generate_line_hypotheses(data, num_hypotheses=12, rng=rng_1)
    h2 = generate_line_hypotheses(data, num_hypotheses=12, rng=rng_2)

    assert h1.shape == (12, 3)
    assert np.allclose(h1, h2)
    assert np.all(np.linalg.norm(h1[:, :2], axis=1) > 0.99)

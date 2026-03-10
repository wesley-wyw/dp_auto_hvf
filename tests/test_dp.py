from __future__ import annotations

import numpy as np

from hvf.pipeline import HVFConfig, HVFPipeline
from privacy.dp_pipeline import DPHVFConfig, apply_dp_hvf
from privacy.sensitivity import bound_point_contributions


def test_contribution_bounding_enforces_row_budget() -> None:
    matrix = np.array([[1.0, 2.0, 3.0], [0.2, 0.3, 0.4]], dtype=float)
    bounded, clipped_rows = bound_point_contributions(matrix, max_contribution=1.0)

    assert clipped_rows == 1
    assert np.all(np.sum(np.abs(bounded), axis=1) <= 1.0 + 1e-8)


def test_dp_pipeline_smoke() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(120, 2))

    pipeline = HVFPipeline(HVFConfig(num_hypotheses=120, use_aikose=True))
    hvf_result = pipeline.run(data, rng=rng)

    dp_result = apply_dp_hvf(
        hvf_result,
        DPHVFConfig(
            epsilon=0.5,
            mechanism="laplace",
            injection_points=("dp_on_hypothesis_scores", "dp_on_model_selection"),
            model_selection_top_k=5,
        ),
        rng=rng,
    )

    assert dp_result.noisy_hypothesis_scores.shape == hvf_result.voting.hypothesis_scores.shape
    assert dp_result.selected_model_indices.size == 5
    assert len(dp_result.privacy_reports) == 2

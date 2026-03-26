from __future__ import annotations

import numpy as np

from data.adelaide import correspondence_set_from_adelaide
from data.loaders import load_adelaide_rmf_sample
from hvf.adelaide_pipeline import AdelaideHVFConfig, run_adelaide_hvf
from hvf.pipeline import HVFConfig, HVFPipeline
from privacy.dp_pipeline import DPHVFConfig, apply_dp_adelaide_hvf, apply_dp_hvf
from privacy.sensitivity import bound_point_contributions

ADELAIDE_SAMPLE = r"2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH\AdelaideCubeFH\cube_FH.mat"


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


def test_dp_adelaide_pipeline_smoke() -> None:
    rng = np.random.default_rng(9)
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)
    result = run_adelaide_hvf(
        correspondences,
        AdelaideHVFConfig(model_family="mixed", num_hypotheses=18, use_aikose=True),
        rng,
    )

    dp_result = apply_dp_adelaide_hvf(
        result,
        DPHVFConfig(
            epsilon=0.7,
            mechanism="laplace",
            injection_points=("dp_on_hypothesis_scores", "dp_on_model_selection"),
            model_selection_top_k=4,
        ),
        rng=rng,
    )

    assert dp_result.dp_result.noisy_hypothesis_scores.shape == result.voting.hypothesis_scores.shape
    assert dp_result.dp_result.selected_model_indices.size == 4
    assert dp_result.selected_model_types.shape == (4,)
    assert dp_result.model_labels.shape == (sample.num_correspondences,)
    assert set(dp_result.selected_model_types.tolist()).issubset({"fundamental", "homography"})


def test_dp_pipeline_manual_epsilon_allocation_sums_to_total() -> None:
    rng = np.random.default_rng(13)
    data = rng.normal(size=(120, 2))

    pipeline = HVFPipeline(HVFConfig(num_hypotheses=120, use_aikose=True))
    hvf_result = pipeline.run(data, rng=rng)

    dp_result = apply_dp_hvf(
        hvf_result,
        DPHVFConfig(
            epsilon=1.0,
            mechanism="laplace",
            injection_points=("dp_on_hypothesis_scores", "dp_on_point_scores", "dp_on_model_selection"),
            epsilon_allocation="manual",
            epsilon_allocations={
                "dp_on_hypothesis_scores": 0.5,
                "dp_on_point_scores": 0.3,
                "dp_on_model_selection": 0.2,
            },
            model_selection_top_k=3,
        ),
        rng=rng,
    )

    eps_by_point = {report.injection_point: float(report.epsilon) for report in dp_result.privacy_reports}
    assert np.isclose(sum(eps_by_point.values()), 1.0)
    assert eps_by_point["dp_on_hypothesis_scores"] > eps_by_point["dp_on_model_selection"]
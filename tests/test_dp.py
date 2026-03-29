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


def test_dp_scores_derived_from_bounded_preference() -> None:
    """Verify that noisy scores are computed from the bounded preference matrix,
    not from the original (unbounded) scores passed in.

    Strategy: use a very large ε so noise is negligible, then check that the
    DP output scores are closer to scores recomputed from the bounded matrix
    than to the original unbounded scores.  This fails if the old bug (adding
    noise to unbounded scores) is reintroduced.
    """
    from privacy.dp_pipeline import _recompute_scores_from_bounded

    rng = np.random.default_rng(42)
    # Use clustered data so preference rows have high L1 norms → clipping
    # will actually change the matrix.
    cluster_1 = rng.normal(loc=[0, 0], scale=0.1, size=(40, 2))
    cluster_2 = rng.normal(loc=[3, 3], scale=0.1, size=(40, 2))
    data = np.vstack([cluster_1, cluster_2])

    pipeline = HVFPipeline(HVFConfig(num_hypotheses=60, use_aikose=True))
    hvf_result = pipeline.run(data, rng=rng)

    # Confirm clipping actually changes the matrix (precondition).
    bounded, clipped = bound_point_contributions(
        hvf_result.preference_matrix, max_contribution=1.0,
    )
    assert clipped > 0, "Test precondition failed: no rows were clipped"
    assert not np.allclose(
        hvf_result.preference_matrix, bounded
    ), "Test precondition failed: bounded matrix is identical to original"

    # Run DP with negligible noise (ε=10000).
    dp_result = apply_dp_hvf(
        hvf_result,
        DPHVFConfig(
            epsilon=10000.0,
            mechanism="laplace",
            injection_points=("dp_on_hypothesis_scores",),
        ),
        rng=np.random.default_rng(42),
    )

    # Recompute expected scores from the bounded matrix.
    expected_scores, _ = _recompute_scores_from_bounded(
        bounded, hvf_result.similarity_matrix, alpha=0.65, beta=0.35,
    )
    original_scores = hvf_result.voting.hypothesis_scores

    # The DP output should be much closer to bounded-recomputed than to original.
    dist_to_bounded = np.linalg.norm(dp_result.noisy_hypothesis_scores - expected_scores)
    dist_to_original = np.linalg.norm(dp_result.noisy_hypothesis_scores - original_scores)

    assert dist_to_bounded < dist_to_original, (
        f"DP scores are closer to original ({dist_to_original:.6f}) "
        f"than to bounded-recomputed ({dist_to_bounded:.6f}) — "
        f"scores may not be derived from bounded preference matrix"
    )
    # With ε=10000, noise is ~1e-4 scale, so distance to bounded should be tiny.
    assert dist_to_bounded < 0.1, (
        f"DP scores deviate too much from bounded-recomputed: {dist_to_bounded:.6f}"
    )


def test_exponential_mechanism_num_queries_equals_k() -> None:
    """Verify that exponential mechanism records num_queries=k, not 1."""
    rng = np.random.default_rng(99)
    data = rng.normal(size=(80, 2))

    pipeline = HVFPipeline(HVFConfig(num_hypotheses=60, use_aikose=True))
    hvf_result = pipeline.run(data, rng=rng)

    top_k = 4
    dp_result = apply_dp_hvf(
        hvf_result,
        DPHVFConfig(
            epsilon=1.0,
            mechanism="laplace",
            injection_points=("dp_on_model_selection",),
            model_selection_top_k=top_k,
        ),
        rng=rng,
    )

    exp_reports = [r for r in dp_result.privacy_reports if r.mechanism == "exponential"]
    assert len(exp_reports) == 1
    assert exp_reports[0].num_queries == top_k, (
        f"Expected num_queries={top_k}, got {exp_reports[0].num_queries}"
    )


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
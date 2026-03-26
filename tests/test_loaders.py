from __future__ import annotations

from pathlib import Path

import numpy as np

from data.adelaide import (
    compute_fundamental_residual_matrix,
    compute_homography_residual_matrix,
    correspondence_set_from_adelaide,
    estimate_fundamental_matrix_eight_point,
    estimate_homography_dlt,
    generate_fundamental_hypotheses,
    generate_homography_hypotheses,
    generate_mixed_hypotheses,
    MixedResidualCalibrationConfig,
)
from data.loaders import load_adelaide_rmf_sample, load_point_set, load_points_from_csv

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
ADELAIDE_SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "2019_CVPR_Fitting Multiple Heterogeneous Models by Multi-class Cascaded T-linkage_Datasets_AdelaideCubeFH"
    / "AdelaideCubeFH"
    / "cube_FH.mat"
)


def test_load_points_from_csv_keeps_first_two_columns() -> None:
    points = load_points_from_csv(FIXTURE_DIR / "points.csv")

    assert points.shape == (2, 2)
    assert np.allclose(points, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_load_point_set_supports_auto_delimiter() -> None:
    loaded = load_point_set(FIXTURE_DIR / "points.txt", delimiter="auto")

    assert loaded.num_points == 2
    assert loaded.source_path.endswith("points.txt")
    assert np.allclose(loaded.points, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_load_adelaide_rmf_sample_reads_core_fields() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)

    assert sample.normalized_correspondences.shape == (6, 295)
    assert sample.image_correspondences.shape == (6, 295)
    assert sample.labels.shape == (295,)
    assert sample.image_1 is not None and sample.image_1.shape == (480, 640, 3)
    assert sample.image_2 is not None and sample.image_2.shape == (480, 640, 3)
    assert sample.num_correspondences == 295
    assert sample.num_models > 0
    assert sample.outlier_mask.shape == (295,)


def test_correspondence_set_from_adelaide_exposes_two_view_points() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)

    assert correspondences.normalized_points_1.shape == (295, 2)
    assert correspondences.normalized_points_2.shape == (295, 2)
    assert correspondences.image_points_1.shape == (295, 2)
    assert correspondences.image_points_2.shape == (295, 2)
    assert correspondences.homogeneous_points_1.shape == (295, 3)
    assert correspondences.homogeneous_points_2.shape == (295, 3)


def test_generate_fundamental_hypotheses_and_residuals_on_adelaide_sample() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)
    rng = np.random.default_rng(0)

    hypotheses = generate_fundamental_hypotheses(correspondences, num_hypotheses=6, rng=rng)
    residuals = compute_fundamental_residual_matrix(correspondences, hypotheses)

    assert hypotheses.shape == (6, 9)
    assert residuals.shape == (295, 6)
    assert np.all(np.isfinite(residuals))
    assert np.all(residuals >= 0.0)


def test_generate_homography_hypotheses_and_residuals_on_adelaide_sample() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)
    rng = np.random.default_rng(1)

    hypotheses = generate_homography_hypotheses(correspondences, num_hypotheses=6, rng=rng)
    residuals = compute_homography_residual_matrix(correspondences, hypotheses)

    assert hypotheses.shape == (6, 9)
    assert residuals.shape == (295, 6)
    assert np.all(np.isfinite(residuals))
    assert np.all(residuals >= 0.0)


def test_generate_mixed_hypotheses_combines_f_and_h() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)
    rng = np.random.default_rng(2)

    mixed = generate_mixed_hypotheses(correspondences, num_hypotheses=10, rng=rng)

    assert mixed.hypotheses.shape == (10, 9)
    assert mixed.residual_matrix.shape == (295, 10)
    assert mixed.model_types.shape == (10,)
    assert np.count_nonzero(mixed.model_types == "fundamental") == 5
    assert np.count_nonzero(mixed.model_types == "homography") == 5


def test_estimate_fundamental_matrix_eight_point_returns_rank_two_matrix() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)

    matrix = estimate_fundamental_matrix_eight_point(
        correspondences.normalized_points_1[:8],
        correspondences.normalized_points_2[:8],
    )

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert matrix.shape == (3, 3)
    assert np.isclose(np.linalg.matrix_rank(matrix), 2)
    assert singular_values[-1] <= 1e-8


def test_estimate_homography_dlt_returns_valid_matrix() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)

    matrix = estimate_homography_dlt(
        correspondences.normalized_points_1[:4],
        correspondences.normalized_points_2[:4],
    )

    assert matrix.shape == (3, 3)
    assert np.all(np.isfinite(matrix))
    assert np.linalg.norm(matrix) > 0.0


def test_generate_mixed_hypotheses_residual_calibration_reduces_scale_gap() -> None:
    sample = load_adelaide_rmf_sample(ADELAIDE_SAMPLE)
    correspondences = correspondence_set_from_adelaide(sample)

    rng_raw = np.random.default_rng(5)
    mixed_raw = generate_mixed_hypotheses(
        correspondences,
        num_hypotheses=12,
        rng=rng_raw,
        calibration=MixedResidualCalibrationConfig(enabled=False),
    )

    rng_cal = np.random.default_rng(5)
    mixed_cal = generate_mixed_hypotheses(
        correspondences,
        num_hypotheses=12,
        rng=rng_cal,
        calibration=MixedResidualCalibrationConfig(enabled=True, quantile=0.5),
    )

    idx_f = np.where(mixed_raw.model_types == "fundamental")[0]
    idx_h = np.where(mixed_raw.model_types == "homography")[0]

    raw_f = float(np.median(mixed_raw.residual_matrix[:, idx_f]))
    raw_h = float(np.median(mixed_raw.residual_matrix[:, idx_h]))
    cal_f = float(np.median(mixed_cal.residual_matrix[:, idx_f]))
    cal_h = float(np.median(mixed_cal.residual_matrix[:, idx_h]))

    raw_ratio = raw_h / max(raw_f, 1e-12)
    cal_ratio = cal_h / max(cal_f, 1e-12)

    assert abs(np.log(cal_ratio + 1e-12)) < abs(np.log(raw_ratio + 1e-12))
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.loaders import AdelaideRMFSample


@dataclass(frozen=True)
class TwoViewCorrespondenceSet:
    """Two-view point correspondences with optional labels and images."""

    normalized_points_1: np.ndarray
    normalized_points_2: np.ndarray
    image_points_1: np.ndarray
    image_points_2: np.ndarray
    labels: np.ndarray
    image_1: np.ndarray | None = None
    image_2: np.ndarray | None = None
    source_path: str = ""

    @property
    def num_points(self) -> int:
        return int(self.normalized_points_1.shape[0])

    @property
    def homogeneous_points_1(self) -> np.ndarray:
        return np.column_stack((self.normalized_points_1, np.ones(self.num_points, dtype=float)))

    @property
    def homogeneous_points_2(self) -> np.ndarray:
        return np.column_stack((self.normalized_points_2, np.ones(self.num_points, dtype=float)))


@dataclass(frozen=True)
class MixedHypothesisSet:
    """Joint hypothesis pool for heterogeneous F/H model competition."""

    hypotheses: np.ndarray
    residual_matrix: np.ndarray
    model_types: np.ndarray

    @property
    def num_hypotheses(self) -> int:
        return int(self.hypotheses.shape[0])



@dataclass(frozen=True)
class MixedResidualCalibrationConfig:
    """Robust residual calibration config for mixed F/H hypothesis pools."""

    enabled: bool = True
    quantile: float = 0.5
    eps: float = 1e-12


def _reshape_correspondence_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != 6:
        raise ValueError("Correspondence matrix must have shape (6, N)")

    point_block_1 = array[0:2, :].T
    point_block_2 = array[3:5, :].T
    return point_block_1, point_block_2



def correspondence_set_from_adelaide(sample: AdelaideRMFSample) -> TwoViewCorrespondenceSet:
    """Convert one AdelaideRMF sample into a two-view correspondence structure."""
    normalized_1, normalized_2 = _reshape_correspondence_matrix(sample.normalized_correspondences)
    image_1, image_2 = _reshape_correspondence_matrix(sample.image_correspondences)

    return TwoViewCorrespondenceSet(
        normalized_points_1=normalized_1,
        normalized_points_2=normalized_2,
        image_points_1=image_1,
        image_points_2=image_2,
        labels=np.asarray(sample.labels, dtype=int),
        image_1=sample.image_1,
        image_2=sample.image_2,
        source_path=sample.source_path,
    )



def _normalize_matrix(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(matrix))
    if norm < eps:
        raise ValueError("Degenerate matrix")
    return matrix / norm



def estimate_fundamental_matrix_eight_point(
    points_1: np.ndarray,
    points_2: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Estimate a rank-2 fundamental matrix from normalized correspondences."""
    x1 = np.asarray(points_1, dtype=float)
    x2 = np.asarray(points_2, dtype=float)

    if x1.shape != x2.shape or x1.ndim != 2 or x1.shape[1] != 2:
        raise ValueError("points_1 and points_2 must both have shape (N, 2)")
    if x1.shape[0] < 8:
        raise ValueError("At least 8 correspondences are required for the eight-point algorithm")

    x = x1[:, 0]
    y = x1[:, 1]
    xp = x2[:, 0]
    yp = x2[:, 1]

    design = np.column_stack((xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, np.ones_like(x)))
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    fundamental = vh[-1].reshape(3, 3)

    u_f, s_f, vh_f = np.linalg.svd(fundamental, full_matrices=False)
    s_f[-1] = 0.0
    rank_two = u_f @ np.diag(s_f) @ vh_f

    return _normalize_matrix(rank_two, eps=eps)



def estimate_homography_dlt(
    points_1: np.ndarray,
    points_2: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Estimate a homography with DLT from point correspondences."""
    x1 = np.asarray(points_1, dtype=float)
    x2 = np.asarray(points_2, dtype=float)

    if x1.shape != x2.shape or x1.ndim != 2 or x1.shape[1] != 2:
        raise ValueError("points_1 and points_2 must both have shape (N, 2)")
    if x1.shape[0] < 4:
        raise ValueError("At least 4 correspondences are required for homography estimation")

    rows: list[np.ndarray] = []
    for (x, y), (xp, yp) in zip(x1, x2, strict=False):
        rows.append(np.array([-x, -y, -1.0, 0.0, 0.0, 0.0, xp * x, xp * y, xp], dtype=float))
        rows.append(np.array([0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp], dtype=float))

    design = np.vstack(rows)
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    homography = vh[-1].reshape(3, 3)

    if abs(float(homography[2, 2])) > eps:
        homography = homography / homography[2, 2]
    return _normalize_matrix(homography, eps=eps)



def generate_fundamental_hypotheses(
    correspondences: TwoViewCorrespondenceSet,
    num_hypotheses: int,
    rng: np.random.Generator,
    max_trials_factor: int = 50,
) -> np.ndarray:
    """Sample normalized fundamental matrix hypotheses from Adelaide-style correspondences."""
    if num_hypotheses < 1:
        raise ValueError("num_hypotheses must be >= 1")
    if correspondences.num_points < 8:
        raise ValueError("At least 8 correspondences are required")

    hypotheses: list[np.ndarray] = []
    max_trials = max(num_hypotheses * max_trials_factor, num_hypotheses)
    trial_count = 0

    while len(hypotheses) < num_hypotheses and trial_count < max_trials:
        trial_count += 1
        sample_indices = rng.choice(correspondences.num_points, size=8, replace=False)

        try:
            matrix = estimate_fundamental_matrix_eight_point(
                correspondences.normalized_points_1[sample_indices],
                correspondences.normalized_points_2[sample_indices],
            )
        except (np.linalg.LinAlgError, ValueError):
            continue

        hypotheses.append(matrix.reshape(-1))

    if len(hypotheses) < num_hypotheses:
        raise RuntimeError(
            "Could not generate enough non-degenerate fundamental hypotheses "
            f"({len(hypotheses)} / {num_hypotheses}) after {max_trials} trials."
        )

    return np.vstack(hypotheses)



def generate_homography_hypotheses(
    correspondences: TwoViewCorrespondenceSet,
    num_hypotheses: int,
    rng: np.random.Generator,
    max_trials_factor: int = 50,
) -> np.ndarray:
    """Sample normalized homography hypotheses from Adelaide-style correspondences."""
    if num_hypotheses < 1:
        raise ValueError("num_hypotheses must be >= 1")
    if correspondences.num_points < 4:
        raise ValueError("At least 4 correspondences are required")

    hypotheses: list[np.ndarray] = []
    max_trials = max(num_hypotheses * max_trials_factor, num_hypotheses)
    trial_count = 0

    while len(hypotheses) < num_hypotheses and trial_count < max_trials:
        trial_count += 1
        sample_indices = rng.choice(correspondences.num_points, size=4, replace=False)

        try:
            matrix = estimate_homography_dlt(
                correspondences.normalized_points_1[sample_indices],
                correspondences.normalized_points_2[sample_indices],
            )
        except (np.linalg.LinAlgError, ValueError):
            continue

        hypotheses.append(matrix.reshape(-1))

    if len(hypotheses) < num_hypotheses:
        raise RuntimeError(
            "Could not generate enough non-degenerate homography hypotheses "
            f"({len(hypotheses)} / {num_hypotheses}) after {max_trials} trials."
        )

    return np.vstack(hypotheses)



def compute_fundamental_residual_matrix(
    correspondences: TwoViewCorrespondenceSet,
    hypotheses: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute Sampson residuals for a batch of fundamental matrix hypotheses."""
    matrices = np.asarray(hypotheses, dtype=float)
    if matrices.ndim != 2 or matrices.shape[1] != 9:
        raise ValueError("hypotheses must have shape (M, 9)")

    x1 = correspondences.homogeneous_points_1
    x2 = correspondences.homogeneous_points_2
    residuals = np.zeros((correspondences.num_points, matrices.shape[0]), dtype=float)

    for index, flat_matrix in enumerate(matrices):
        fundamental = flat_matrix.reshape(3, 3)
        fx1 = x1 @ fundamental.T
        ftx2 = x2 @ fundamental
        numerators = np.sum(x2 * fx1, axis=1) ** 2
        denominators = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
        denominators = np.where(denominators < eps, eps, denominators)
        residuals[:, index] = numerators / denominators

    return residuals



def compute_homography_residual_matrix(
    correspondences: TwoViewCorrespondenceSet,
    hypotheses: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute symmetric transfer errors for a batch of homography hypotheses."""
    matrices = np.asarray(hypotheses, dtype=float)
    if matrices.ndim != 2 or matrices.shape[1] != 9:
        raise ValueError("hypotheses must have shape (M, 9)")

    x1 = correspondences.homogeneous_points_1
    x2 = correspondences.homogeneous_points_2
    residuals = np.zeros((correspondences.num_points, matrices.shape[0]), dtype=float)

    for index, flat_matrix in enumerate(matrices):
        homography = flat_matrix.reshape(3, 3)
        inverse = np.linalg.pinv(homography)

        projected_2 = x1 @ homography.T
        projected_2 = projected_2 / np.where(np.abs(projected_2[:, 2:3]) < eps, eps, projected_2[:, 2:3])

        projected_1 = x2 @ inverse.T
        projected_1 = projected_1 / np.where(np.abs(projected_1[:, 2:3]) < eps, eps, projected_1[:, 2:3])

        forward_error = np.sum((projected_2[:, :2] - correspondences.normalized_points_2) ** 2, axis=1)
        backward_error = np.sum((projected_1[:, :2] - correspondences.normalized_points_1) ** 2, axis=1)
        residuals[:, index] = forward_error + backward_error

    return residuals





def _robust_family_scale(residual_matrix: np.ndarray, quantile: float, eps: float) -> float:
    values = np.asarray(residual_matrix, dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return 1.0
    q = float(np.clip(quantile, 0.05, 0.95))
    scale = float(np.quantile(positive, q=q))
    return max(scale, eps)


def calibrate_mixed_residual_matrix(
    residual_fundamental: np.ndarray,
    residual_homography: np.ndarray,
    config: MixedResidualCalibrationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate F/H residuals to improve cross-family comparability."""
    if not config.enabled:
        return np.asarray(residual_fundamental, dtype=float), np.asarray(residual_homography, dtype=float)

    eps = max(float(config.eps), 1e-12)
    fundamental = np.asarray(residual_fundamental, dtype=float)
    homography = np.asarray(residual_homography, dtype=float)

    scale_f = _robust_family_scale(fundamental, quantile=config.quantile, eps=eps)
    scale_h = _robust_family_scale(homography, quantile=config.quantile, eps=eps)

    return fundamental / scale_f, homography / scale_h

def generate_mixed_hypotheses(
    correspondences: TwoViewCorrespondenceSet,
    num_hypotheses: int,
    rng: np.random.Generator,
    calibration: MixedResidualCalibrationConfig = MixedResidualCalibrationConfig(),
) -> MixedHypothesisSet:
    """Sample a joint F/H hypothesis pool and concatenate residual matrices."""
    if num_hypotheses < 2:
        raise ValueError("mixed mode requires at least 2 hypotheses")

    num_fundamental = num_hypotheses // 2
    num_homography = num_hypotheses - num_fundamental

    fundamental_hypotheses = generate_fundamental_hypotheses(correspondences, num_fundamental, rng=rng)
    homography_hypotheses = generate_homography_hypotheses(correspondences, num_homography, rng=rng)

    residual_fundamental = compute_fundamental_residual_matrix(correspondences, fundamental_hypotheses)
    residual_homography = compute_homography_residual_matrix(correspondences, homography_hypotheses)
    residual_fundamental, residual_homography = calibrate_mixed_residual_matrix(
        residual_fundamental,
        residual_homography,
        config=calibration,
    )

    hypotheses = np.vstack((fundamental_hypotheses, homography_hypotheses))
    residual_matrix = np.hstack((residual_fundamental, residual_homography))
    model_types = np.array(
        ["fundamental"] * num_fundamental + ["homography"] * num_homography,
        dtype=object,
    )

    return MixedHypothesisSet(
        hypotheses=hypotheses,
        residual_matrix=residual_matrix,
        model_types=model_types,
    )

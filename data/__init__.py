"""Data utilities for synthetic generation, loading, and preprocessing."""

from .adelaide import (
    MixedHypothesisSet,
    TwoViewCorrespondenceSet,
    compute_fundamental_residual_matrix,
    compute_homography_residual_matrix,
    correspondence_set_from_adelaide,
    estimate_fundamental_matrix_eight_point,
    estimate_homography_dlt,
    generate_fundamental_hypotheses,
    generate_homography_hypotheses,
    generate_mixed_hypotheses,
)
from .loaders import AdelaideRMFSample, LoadedPointSet, list_adelaide_rmf_files, load_adelaide_rmf_sample, load_point_set, load_points_from_csv
from .synthetic import LineModelSpec, SyntheticDataset, generate_synthetic_dataset

__all__ = [
    "AdelaideRMFSample",
    "LineModelSpec",
    "LoadedPointSet",
    "MixedHypothesisSet",
    "SyntheticDataset",
    "TwoViewCorrespondenceSet",
    "compute_fundamental_residual_matrix",
    "compute_homography_residual_matrix",
    "correspondence_set_from_adelaide",
    "estimate_fundamental_matrix_eight_point",
    "estimate_homography_dlt",
    "generate_fundamental_hypotheses",
    "generate_homography_hypotheses",
    "generate_mixed_hypotheses",
    "generate_synthetic_dataset",
    "list_adelaide_rmf_files",
    "load_adelaide_rmf_sample",
    "load_point_set",
    "load_points_from_csv",
]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io


@dataclass(frozen=True)
class LoadedPointSet:
    """Loaded 2D point set with lightweight provenance metadata."""

    points: np.ndarray
    source_path: str
    num_points: int


@dataclass(frozen=True)
class AdelaideRMFSample:
    """Structured AdelaideRMF sample with correspondences, labels, and optional images."""

    normalized_correspondences: np.ndarray
    image_correspondences: np.ndarray
    labels: np.ndarray
    image_1: np.ndarray | None
    image_2: np.ndarray | None
    source_path: str

    @property
    def num_correspondences(self) -> int:
        return int(self.normalized_correspondences.shape[1])

    @property
    def outlier_mask(self) -> np.ndarray:
        return self.labels == 0

    @property
    def inlier_mask(self) -> np.ndarray:
        return ~self.outlier_mask

    @property
    def num_models(self) -> int:
        positive_labels = self.labels[self.labels > 0]
        return int(np.unique(positive_labels).size)



def _load_numeric_matrix(file_path: Path, delimiter: str) -> np.ndarray:
    if delimiter == "auto":
        candidate_delimiters: tuple[str | None, ...] = (",", "\t", None)
    else:
        candidate_delimiters = (delimiter,)

    last_error: Exception | None = None
    for candidate in candidate_delimiters:
        try:
            matrix = np.loadtxt(file_path, delimiter=candidate)
        except Exception as error:  # noqa: BLE001
            last_error = error
            continue

        if np.asarray(matrix).size > 0:
            return np.asarray(matrix, dtype=float)

    if last_error is not None:
        raise ValueError(f"Could not parse numeric point data from {file_path}") from last_error
    raise ValueError(f"Could not parse numeric point data from {file_path}")



def load_points_from_csv(path: str | Path, delimiter: str = ",") -> np.ndarray:
    """Load 2D points from a text file with at least two numeric columns."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    points = _load_numeric_matrix(file_path, delimiter=delimiter)
    if points.ndim == 1:
        points = points[None, :]
    if points.shape[1] < 2:
        raise ValueError("Point data must contain at least two columns for x,y.")

    return points[:, :2].astype(float)



def load_point_set(path: str | Path, delimiter: str = ",") -> LoadedPointSet:
    """Load a 2D point set and attach minimal metadata for reporting."""
    points = load_points_from_csv(path, delimiter=delimiter)
    return LoadedPointSet(
        points=points,
        source_path=str(Path(path)),
        num_points=int(points.shape[0]),
    )



def _coerce_label_vector(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels).reshape(-1)
    return array.astype(int, copy=False)



def _is_adelaide_mat_path(file_path: Path) -> bool:
    suffix = file_path.suffix.lower()
    return suffix == ".mat" or file_path.name.endswith("FHmat")



def list_adelaide_rmf_files(root: str | Path) -> list[Path]:
    """List AdelaideRMF sample files in a directory, including malformed FHmat names."""
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        if not _is_adelaide_mat_path(path):
            raise ValueError(f"Not an AdelaideRMF .mat sample: {path}")
        return [path]

    files = sorted(file for file in path.iterdir() if file.is_file() and _is_adelaide_mat_path(file))
    if not files:
        raise ValueError(f"No AdelaideRMF sample files found under {path}")
    return files



def load_adelaide_rmf_sample(path: str | Path) -> AdelaideRMFSample:
    """Load one AdelaideRMF .mat sample.

    The expected fields are:
    - X: 6 x N normalized correspondences
    - y: 6 x N image-space correspondences
    - G: N x 1 or 1 x N labels, with 0 reserved for outliers
    - img1 / img2: optional RGB images
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not _is_adelaide_mat_path(file_path):
        raise ValueError("AdelaideRMF samples must be provided as .mat files")

    data = scipy.io.loadmat(file_path)
    required_fields = ("X", "y", "G")
    missing = [field for field in required_fields if field not in data]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing AdelaideRMF fields in {file_path}: {missing_text}")

    normalized = np.asarray(data["X"], dtype=float)
    image_space = np.asarray(data["y"], dtype=float)
    labels = _coerce_label_vector(data["G"])

    if normalized.ndim != 2 or normalized.shape[0] != 6:
        raise ValueError("AdelaideRMF field X must have shape (6, N)")
    if image_space.ndim != 2 or image_space.shape != normalized.shape:
        raise ValueError("AdelaideRMF field y must have the same shape as X")
    if labels.shape[0] != normalized.shape[1]:
        raise ValueError("AdelaideRMF field G must provide one label per correspondence")

    image_1 = np.asarray(data["img1"]) if "img1" in data else None
    image_2 = np.asarray(data["img2"]) if "img2" in data else None

    return AdelaideRMFSample(
        normalized_correspondences=normalized,
        image_correspondences=image_space,
        labels=labels,
        image_1=image_1,
        image_2=image_2,
        source_path=str(file_path),
    )

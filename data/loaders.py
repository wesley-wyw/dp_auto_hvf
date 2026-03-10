from __future__ import annotations

from pathlib import Path

import numpy as np


def load_points_from_csv(path: str | Path, delimiter: str = ",") -> np.ndarray:
    """Load 2D points from a CSV file with at least two columns."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    points = np.loadtxt(file_path, delimiter=delimiter)
    if points.ndim == 1:
        points = points[None, :]
    if points.shape[1] < 2:
        raise ValueError("CSV data must contain at least two columns for x,y.")

    return points[:, :2].astype(float)

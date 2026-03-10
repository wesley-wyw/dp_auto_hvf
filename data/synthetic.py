from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LineModelSpec:
    """Ground-truth line model in slope-intercept form."""

    slope: float
    intercept: float


@dataclass(frozen=True)
class SyntheticDataset:
    """Container for generated points and supervision metadata."""

    points: np.ndarray
    point_labels: np.ndarray
    true_models: np.ndarray


def normalize_line_parameters(lines: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize line parameters [A, B, C] so sqrt(A^2 + B^2) == 1 and sign is canonical."""
    array = np.asarray(lines, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[1] != 3:
        raise ValueError("Line parameters must have shape (M, 3).")

    norms = np.linalg.norm(array[:, :2], axis=1)
    norms = np.where(norms < eps, eps, norms)
    normalized = array / norms[:, None]

    # Canonical orientation: positive A, or if A == 0 then positive B.
    signs = np.where(
        (normalized[:, 0] < 0.0)
        | ((np.isclose(normalized[:, 0], 0.0)) & (normalized[:, 1] < 0.0)),
        -1.0,
        1.0,
    )
    normalized *= signs[:, None]
    return normalized


def _default_line_specs(num_models: int) -> list[LineModelSpec]:
    base = [
        LineModelSpec(slope=1.8, intercept=1.2),
        LineModelSpec(slope=-0.65, intercept=8.0),
        LineModelSpec(slope=0.2, intercept=3.5),
        LineModelSpec(slope=-1.3, intercept=14.0),
    ]
    if num_models <= len(base):
        return base[:num_models]
    return [base[index % len(base)] for index in range(num_models)]


def _line_to_abc(model: LineModelSpec) -> np.ndarray:
    # y = mx + b -> mx - y + b = 0
    return np.array([model.slope, -1.0, model.intercept], dtype=float)


def generate_synthetic_dataset(
    *,
    num_models: int = 2,
    points_per_model: int = 80,
    outlier_count: int = 80,
    noise_sigma: float = 0.20,
    x_range: tuple[float, float] = (0.0, 10.0),
    y_range: tuple[float, float] = (-2.0, 20.0),
    model_specs: Iterable[LineModelSpec] | None = None,
    rng: np.random.Generator | None = None,
) -> SyntheticDataset:
    """Generate a multi-line fitting dataset with explicit outliers and labels."""
    generator = rng if rng is not None else np.random.default_rng(0)

    if num_models < 1:
        raise ValueError("num_models must be >= 1")
    if points_per_model < 2:
        raise ValueError("points_per_model must be >= 2")
    if outlier_count < 0:
        raise ValueError("outlier_count must be >= 0")

    specs = list(model_specs) if model_specs is not None else _default_line_specs(num_models)
    if len(specs) < num_models:
        raise ValueError("model_specs has fewer entries than num_models")

    points_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    true_models: list[np.ndarray] = []

    x_min, x_max = x_range
    y_min, y_max = y_range

    for model_index, spec in enumerate(specs[:num_models]):
        x_values = generator.uniform(x_min, x_max, size=points_per_model)
        y_values = spec.slope * x_values + spec.intercept
        y_values += generator.normal(loc=0.0, scale=noise_sigma, size=points_per_model)
        line_points = np.column_stack((x_values, y_values))

        points_list.append(line_points)
        labels_list.append(np.full(points_per_model, model_index, dtype=int))
        true_models.append(_line_to_abc(spec))

    if outlier_count > 0:
        x_out = generator.uniform(x_min, x_max, size=outlier_count)
        y_out = generator.uniform(y_min, y_max, size=outlier_count)
        outliers = np.column_stack((x_out, y_out))
        points_list.append(outliers)
        labels_list.append(np.full(outlier_count, -1, dtype=int))

    points = np.vstack(points_list)
    labels = np.concatenate(labels_list)

    permutation = generator.permutation(points.shape[0])
    shuffled_points = points[permutation]
    shuffled_labels = labels[permutation]

    return SyntheticDataset(
        points=shuffled_points,
        point_labels=shuffled_labels,
        true_models=normalize_line_parameters(np.vstack(true_models)),
    )

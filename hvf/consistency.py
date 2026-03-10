from __future__ import annotations

import numpy as np


def _row_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    sums = matrix.sum(axis=1, keepdims=True)
    safe = np.where(np.abs(sums) < eps, 1.0, sums)
    return matrix / safe


def _symmetric_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    degrees = matrix.sum(axis=1)
    inv_sqrt = np.where(degrees > eps, 1.0 / np.sqrt(degrees), 0.0)
    return inv_sqrt[:, None] * matrix * inv_sqrt[None, :]


def _top_k_sparsify(matrix: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= matrix.shape[1]:
        return matrix

    sparse = np.zeros_like(matrix)
    indices = np.argpartition(matrix, kth=-top_k, axis=1)[:, -top_k:]
    row_indices = np.arange(matrix.shape[0])[:, None]
    sparse[row_indices, indices] = matrix[row_indices, indices]
    return np.maximum(sparse, sparse.T)


def compute_point_similarity(
    preference_matrix: np.ndarray,
    normalize: str = "row",
    sparsify: bool = False,
    top_k: int = 30,
    remove_diagonal: bool = True,
) -> np.ndarray:
    """Compute point consistency/similarity matrix from preference matrix P."""
    preference = np.asarray(preference_matrix, dtype=float)
    if preference.ndim != 2:
        raise ValueError("preference_matrix must have shape (N, M)")

    similarity = preference @ preference.T

    if remove_diagonal:
        np.fill_diagonal(similarity, 0.0)

    normalization = normalize.lower()
    if normalization == "row":
        similarity = _row_normalize(similarity)
    elif normalization == "symmetric":
        similarity = _symmetric_normalize(similarity)
    elif normalization in {"none", "off"}:
        pass
    else:
        raise ValueError("normalize must be one of: row, symmetric, none")

    if sparsify:
        similarity = _top_k_sparsify(similarity, top_k=top_k)

    return similarity

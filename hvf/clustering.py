from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusteringConfig:
    """Configuration for hypothesis clustering."""

    parameter_distance_threshold: float = 0.12
    jaccard_threshold: float = 0.55
    min_cluster_size: int = 2


@dataclass(frozen=True)
class ClusterResult:
    """Clustering output for hypotheses."""

    clusters: list[np.ndarray]
    adjacency_matrix: np.ndarray


def _pairwise_jaccard(binary_matrix: np.ndarray) -> np.ndarray:
    m = binary_matrix.shape[1]
    similarity = np.eye(m, dtype=float)

    for i in range(m):
        col_i = binary_matrix[:, i]
        for j in range(i + 1, m):
            col_j = binary_matrix[:, j]
            intersection = float(np.logical_and(col_i, col_j).sum())
            union = float(np.logical_or(col_i, col_j).sum())
            score = intersection / union if union > 0 else 0.0
            similarity[i, j] = score
            similarity[j, i] = score
    return similarity


def _connected_components(adjacency: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(adjacency.shape[0], dtype=bool)
    components: list[np.ndarray] = []

    for start in range(adjacency.shape[0]):
        if visited[start]:
            continue
        stack = [start]
        members: list[int] = []

        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            members.append(node)
            neighbors = np.where(adjacency[node])[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    stack.append(int(neighbor))

        components.append(np.array(sorted(members), dtype=int))

    return components


def cluster_hypotheses(
    hypotheses: np.ndarray,
    preference_matrix: np.ndarray,
    config: ClusteringConfig,
) -> ClusterResult:
    """Cluster hypotheses using parameter distance and consensus overlap (Jaccard)."""
    models = np.asarray(hypotheses, dtype=float)
    preferences = np.asarray(preference_matrix, dtype=float)

    if models.ndim != 2 or models.shape[1] != 3:
        raise ValueError("hypotheses must have shape (M, 3)")
    if preferences.ndim != 2 or preferences.shape[1] != models.shape[0]:
        raise ValueError("preference_matrix must have shape (N, M)")

    parameter_distances = np.linalg.norm(models[:, None, :] - models[None, :, :], axis=2)
    parameter_links = parameter_distances <= config.parameter_distance_threshold

    binary_preferences = preferences > 0.0
    jaccard_similarity = _pairwise_jaccard(binary_preferences)
    consensus_links = jaccard_similarity >= config.jaccard_threshold

    adjacency = np.logical_or(parameter_links, consensus_links)
    np.fill_diagonal(adjacency, True)

    raw_components = _connected_components(adjacency)
    clusters = [component for component in raw_components if component.size >= config.min_cluster_size]

    if not clusters:
        clusters = [np.array([int(np.argmax(preferences.sum(axis=0)))], dtype=int)]

    return ClusterResult(clusters=clusters, adjacency_matrix=adjacency.astype(float))

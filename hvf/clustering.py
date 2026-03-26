from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusteringConfig:
    """Configuration for hypothesis clustering."""

    parameter_distance_threshold: float = 0.08
    jaccard_threshold: float = 0.75
    cross_type_jaccard_threshold: float = 0.90
    min_cluster_size: int = 1
    use_parameter_links_for_cross_type: bool = False
    support_threshold: float = 0.95


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
    hypothesis_model_types: np.ndarray | None = None,
) -> ClusterResult:
    """Cluster hypotheses using parameter distance and consensus overlap (Jaccard)."""
    models = np.asarray(hypotheses, dtype=float)
    preferences = np.asarray(preference_matrix, dtype=float)

    if models.ndim != 2 or models.shape[0] == 0:
        raise ValueError("hypotheses must have shape (M, D) with M > 0")
    if preferences.ndim != 2 or preferences.shape[1] != models.shape[0]:
        raise ValueError("preference_matrix must have shape (N, M)")

    model_types: np.ndarray | None = None
    if hypothesis_model_types is not None:
        model_types = np.asarray(hypothesis_model_types, dtype=object)
        if model_types.ndim != 1 or model_types.shape[0] != models.shape[0]:
            raise ValueError("hypothesis_model_types must have shape (M,)")

    parameter_distances = np.linalg.norm(models[:, None, :] - models[None, :, :], axis=2)
    parameter_links = parameter_distances <= config.parameter_distance_threshold

    binary_preferences = preferences >= float(config.support_threshold)
    jaccard_similarity = _pairwise_jaccard(binary_preferences)

    if model_types is None or np.unique(model_types).size <= 1:
        consensus_links = jaccard_similarity >= config.jaccard_threshold
        adjacency = np.logical_or(parameter_links, consensus_links)
    else:
        same_type = model_types[:, None] == model_types[None, :]
        cross_type = ~same_type

        same_consensus_links = np.logical_and(jaccard_similarity >= config.jaccard_threshold, same_type)
        cross_consensus_links = np.logical_and(
            jaccard_similarity >= config.cross_type_jaccard_threshold,
            cross_type,
        )

        same_parameter_links = np.logical_and(parameter_links, same_type)
        if config.use_parameter_links_for_cross_type:
            cross_parameter_links = np.logical_and(parameter_links, cross_type)
            adjacency = np.logical_or.reduce(
                (same_parameter_links, same_consensus_links, cross_consensus_links, cross_parameter_links)
            )
        else:
            adjacency = np.logical_or.reduce((same_parameter_links, same_consensus_links, cross_consensus_links))

    np.fill_diagonal(adjacency, True)

    raw_components = _connected_components(adjacency)
    clusters = [component for component in raw_components if component.size >= config.min_cluster_size]

    if not clusters:
        clusters = [np.array([int(np.argmax(preferences.sum(axis=0)))], dtype=int)]

    return ClusterResult(clusters=clusters, adjacency_matrix=adjacency.astype(float))

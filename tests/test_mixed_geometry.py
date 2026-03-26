from __future__ import annotations

import numpy as np

from hvf.clustering import ClusteringConfig, cluster_hypotheses
from hvf.extraction import ExtractionConfig, extract_model_instances
from hvf.pruning import PruningConfig, prune_hypotheses


def test_pruning_mixed_cross_type_does_not_dedup_by_parameter_only() -> None:
    hypotheses = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.01, 0.0, 0.0],
        ],
        dtype=float,
    )
    scores = np.array([1.0, 0.9], dtype=float)
    preference_matrix = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    model_types = np.array(["fundamental", "homography"], dtype=object)

    result = prune_hypotheses(
        hypotheses=hypotheses,
        scores=scores,
        config=PruningConfig(
            top_k=2,
            score_quantile=0.0,
            duplicate_epsilon=0.2,
            duplicate_jaccard_threshold=0.7,
            cross_type_duplicate_jaccard_threshold=0.95,
            prefer_consensus_for_mixed=True,
        ),
        preference_matrix=preference_matrix,
        hypothesis_model_types=model_types,
    )

    assert result.kept_indices.shape == (2,)


def test_clustering_mixed_uses_cross_type_consensus_not_parameter_distance() -> None:
    hypotheses = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.01, 0.0, 0.0],
            [1.02, 0.0, 0.0],
        ],
        dtype=float,
    )
    model_types = np.array(["fundamental", "fundamental", "homography"], dtype=object)
    preference_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    result = cluster_hypotheses(
        hypotheses=hypotheses,
        preference_matrix=preference_matrix,
        config=ClusteringConfig(
            parameter_distance_threshold=0.2,
            jaccard_threshold=0.8,
            cross_type_jaccard_threshold=0.95,
            min_cluster_size=1,
            use_parameter_links_for_cross_type=False,
        ),
        hypothesis_model_types=model_types,
    )

    cluster_sizes = sorted(int(cluster.size) for cluster in result.clusters)
    assert cluster_sizes == [1, 2]


def test_extraction_overlap_suppression_keeps_distinct_models() -> None:
    hypotheses = np.array(
        [
            [1.0, 0.0],
            [1.01, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    residuals = np.array(
        [
            [0.01, 0.02, 2.0],
            [0.02, 0.01, 2.0],
            [0.03, 0.02, 2.0],
            [2.0, 2.0, 0.01],
            [2.0, 2.0, 0.02],
            [2.0, 2.0, 0.03],
        ],
        dtype=float,
    )
    scales = np.array([0.05, 0.05, 0.05], dtype=float)
    scores = np.array([0.9, 0.85, 0.8], dtype=float)
    clusters = [np.array([0]), np.array([1]), np.array([2])]
    model_types = np.array(["fundamental", "fundamental", "homography"], dtype=object)

    result = extract_model_instances(
        hypotheses=hypotheses,
        clusters=clusters,
        residuals=residuals,
        scales=scales,
        hypothesis_scores=scores,
        config=ExtractionConfig(min_cluster_size=1, min_inliers=2, overlap_jaccard_threshold=0.6, max_models=8),
        hypothesis_model_types=model_types,
    )

    assert result.selected_hypothesis_indices.tolist() == [0, 2]

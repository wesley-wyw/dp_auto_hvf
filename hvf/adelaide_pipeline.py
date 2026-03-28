from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from aikose.estimator import AIKOSEConfig, estimate_scales
from aikose.diagnostics import TauDiagnostics
from data.adelaide import (
    TwoViewCorrespondenceSet,
    compute_fundamental_residual_matrix,
    compute_homography_residual_matrix,
    generate_fundamental_hypotheses,
    generate_homography_hypotheses,
    generate_mixed_hypotheses,
    MixedResidualCalibrationConfig,
)
from .clustering import ClusterResult, ClusteringConfig, cluster_hypotheses
from .consistency import compute_point_similarity
from .extraction import ExtractionConfig, ExtractionResult, extract_model_instances
from .pipeline import _fixed_scale_batch
from .preference import PreferenceStats, build_preference_matrix
from .pruning import PruningConfig, PruningResult, prune_hypotheses
from .voting import VotingResult, hierarchical_vote


@dataclass(frozen=True)
class AdelaideHVFConfig:
    """Config for running HVF-style downstream stages on AdelaideRMF correspondences."""

    model_family: str = "fundamental"
    num_hypotheses: int = 320
    use_aikose: bool = True
    fixed_tau: float = 0.05
    preference_kernel: str = "exponential"
    similarity_normalize: str = "row"
    similarity_sparsify: bool = False
    similarity_top_k: int = 30
    remove_similarity_diagonal: bool = True
    voting_alpha: float = 0.65
    voting_beta: float = 0.35
    aikose: AIKOSEConfig = AIKOSEConfig(
        search_upper_quantile=0.25,
        log_space=True,
    )
    pruning: PruningConfig = PruningConfig()
    clustering: ClusteringConfig = ClusteringConfig(
        parameter_distance_threshold=0.08,
        jaccard_threshold=0.75,
        cross_type_jaccard_threshold=0.90,
        min_cluster_size=1,
    )
    extraction: ExtractionConfig = ExtractionConfig(
        min_cluster_size=1,
        min_inliers=8,
        overlap_jaccard_threshold=0.90,
        cross_type_overlap_jaccard_threshold=0.95,
        max_models=5,
        representatives_per_cluster=2,
    )
    mixed_residual_calibration: MixedResidualCalibrationConfig = MixedResidualCalibrationConfig()


@dataclass(frozen=True)
class AdelaideHVFResult:
    correspondences: TwoViewCorrespondenceSet
    model_family: str
    hypotheses: np.ndarray
    residual_matrix: np.ndarray
    hypothesis_model_types: np.ndarray
    selected_model_types: np.ndarray
    scales: np.ndarray
    tau_diagnostics: TauDiagnostics
    preference_matrix: np.ndarray
    preference_stats: PreferenceStats
    similarity_matrix: np.ndarray
    voting: VotingResult
    pruning: PruningResult
    clustering: ClusterResult
    extraction: ExtractionResult
    runtime_seconds: float



def _generate_hypotheses_and_residuals(
    correspondences: TwoViewCorrespondenceSet,
    config: AdelaideHVFConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    family = config.model_family.lower().strip()
    if family == "fundamental":
        hypotheses = generate_fundamental_hypotheses(correspondences, config.num_hypotheses, rng=rng)
        residuals = compute_fundamental_residual_matrix(correspondences, hypotheses)
        model_types = np.full(hypotheses.shape[0], "fundamental", dtype=object)
        return hypotheses, residuals, model_types
    if family == "homography":
        hypotheses = generate_homography_hypotheses(correspondences, config.num_hypotheses, rng=rng)
        residuals = compute_homography_residual_matrix(correspondences, hypotheses)
        model_types = np.full(hypotheses.shape[0], "homography", dtype=object)
        return hypotheses, residuals, model_types
    if family == "mixed":
        mixed = generate_mixed_hypotheses(
            correspondences,
            config.num_hypotheses,
            rng=rng,
            calibration=config.mixed_residual_calibration,
        )
        return mixed.hypotheses, mixed.residual_matrix, mixed.model_types

    raise ValueError("model_family must be one of: fundamental, homography, mixed")



def run_adelaide_hvf(
    correspondences: TwoViewCorrespondenceSet,
    config: AdelaideHVFConfig,
    rng: np.random.Generator,
) -> AdelaideHVFResult:
    """Run HVF-like downstream stages on AdelaideRMF correspondences for one model family."""
    start_time = time.perf_counter()
    hypotheses, residual_matrix, hypothesis_model_types = _generate_hypotheses_and_residuals(correspondences, config, rng)

    if config.use_aikose:
        scale_batch = estimate_scales(residual_matrix, config.aikose)
    else:
        scale_batch = _fixed_scale_batch(hypotheses.shape[0], config.fixed_tau)

    preference_matrix, preference_stats = build_preference_matrix(
        residual_matrix,
        scales=scale_batch.taus,
        kernel=config.preference_kernel,
    )
    similarity_matrix = compute_point_similarity(
        preference_matrix,
        normalize=config.similarity_normalize,
        sparsify=config.similarity_sparsify,
        top_k=config.similarity_top_k,
        remove_diagonal=config.remove_similarity_diagonal,
    )
    voting_result = hierarchical_vote(
        preference_matrix=preference_matrix,
        similarity_matrix=similarity_matrix,
        alpha=config.voting_alpha,
        beta=config.voting_beta,
    )
    pruning_result = prune_hypotheses(
        hypotheses=hypotheses,
        scores=voting_result.hypothesis_scores,
        config=config.pruning,
        preference_matrix=preference_matrix,
        hypothesis_model_types=hypothesis_model_types,
    )

    pruned_indices = pruning_result.kept_indices
    pruned_hypotheses = pruning_result.pruned_hypotheses
    pruned_residuals = residual_matrix[:, pruned_indices]
    pruned_scales = scale_batch.taus[pruned_indices]
    pruned_preferences = preference_matrix[:, pruned_indices]
    pruned_model_types = hypothesis_model_types[pruned_indices]

    clustering_result = cluster_hypotheses(
        hypotheses=pruned_hypotheses,
        preference_matrix=pruned_preferences,
        config=config.clustering,
        hypothesis_model_types=pruned_model_types,
    )
    extraction_result = extract_model_instances(
        hypotheses=pruned_hypotheses,
        clusters=clustering_result.clusters,
        residuals=pruned_residuals,
        scales=pruned_scales,
        hypothesis_scores=pruning_result.pruned_scores,
        config=config.extraction,
        hypothesis_model_types=pruned_model_types,
    )
    selected_model_types = pruned_model_types[extraction_result.selected_hypothesis_indices]

    runtime = time.perf_counter() - start_time
    return AdelaideHVFResult(
        correspondences=correspondences,
        model_family=config.model_family,
        hypotheses=hypotheses,
        residual_matrix=residual_matrix,
        hypothesis_model_types=hypothesis_model_types,
        selected_model_types=selected_model_types,
        scales=scale_batch.taus,
        tau_diagnostics=scale_batch.diagnostics,
        preference_matrix=preference_matrix,
        preference_stats=preference_stats,
        similarity_matrix=similarity_matrix,
        voting=voting_result,
        pruning=pruning_result,
        clustering=clustering_result,
        extraction=extraction_result,
        runtime_seconds=runtime,
    )

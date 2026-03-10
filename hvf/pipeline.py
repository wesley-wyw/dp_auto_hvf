from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time

import numpy as np

from aikose.estimator import AIKOSEConfig, ScaleBatchResult, ScaleEstimationResult, estimate_scales
from aikose.diagnostics import TauDiagnostics
from .clustering import ClusterResult, ClusteringConfig, cluster_hypotheses
from .consistency import compute_point_similarity
from .extraction import ExtractionConfig, ExtractionResult, extract_model_instances
from .hypotheses import generate_line_hypotheses
from .preference import PreferenceStats, build_preference_matrix
from .pruning import PruningConfig, PruningResult, prune_hypotheses
from .residuals import compute_residual_matrix
from .voting import VotingResult, hierarchical_vote

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HVFConfig:
    """Configuration for the full HVF pipeline."""

    num_hypotheses: int = 500
    use_aikose: bool = False
    fixed_tau: float = 0.65
    preference_kernel: str = "exponential"
    similarity_normalize: str = "row"
    similarity_sparsify: bool = False
    similarity_top_k: int = 30
    remove_similarity_diagonal: bool = True
    voting_alpha: float = 0.65
    voting_beta: float = 0.35
    outlier_quantile: float = 0.85
    max_hypothesis_trials_factor: int = 30
    aikose: AIKOSEConfig = field(default_factory=AIKOSEConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)


@dataclass(frozen=True)
class HVFRunResult:
    """Full outputs from one HVF run."""

    hypotheses: np.ndarray
    residual_matrix: np.ndarray
    scales: np.ndarray
    scale_results: list[ScaleEstimationResult]
    tau_diagnostics: TauDiagnostics
    preference_matrix: np.ndarray
    preference_stats: PreferenceStats
    similarity_matrix: np.ndarray
    voting: VotingResult
    pruning: PruningResult
    clustering: ClusterResult
    extraction: ExtractionResult
    outlier_mask: np.ndarray
    inlier_mask: np.ndarray
    runtime_seconds: float


def _fixed_scale_batch(num_hypotheses: int, tau: float) -> ScaleBatchResult:
    scales = np.full(num_hypotheses, max(tau, 1e-6), dtype=float)
    results = [
        ScaleEstimationResult(
            tau=float(scales[index]),
            best_k=-1,
            gradients=np.array([], dtype=float),
            valid=True,
            fallback_used=False,
        )
        for index in range(num_hypotheses)
    ]
    diagnostics = TauDiagnostics(
        count=num_hypotheses,
        tau_min=float(np.min(scales)),
        tau_max=float(np.max(scales)),
        tau_mean=float(np.mean(scales)),
        tau_std=float(np.std(scales)),
        tau_median=float(np.median(scales)),
        invalid_count=0,
        fallback_count=0,
    )
    return ScaleBatchResult(taus=scales, results=results, diagnostics=diagnostics)


def detect_outliers(outlier_scores: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray]:
    """Explicit outlier detection stage from point saliency/outlier scores."""
    scores = np.asarray(outlier_scores, dtype=float)
    threshold = float(np.quantile(scores, q=quantile))
    outlier_mask = scores >= threshold
    inlier_mask = ~outlier_mask
    return outlier_mask, inlier_mask


class HVFPipeline:
    """Research-grade HVF pipeline with hypothesis generation to extraction."""

    def __init__(self, config: HVFConfig) -> None:
        self.config = config

    def run(self, data: np.ndarray, rng: np.random.Generator) -> HVFRunResult:
        start_time = time.perf_counter()

        hypotheses = generate_line_hypotheses(
            data=data,
            num_hypotheses=self.config.num_hypotheses,
            rng=rng,
            max_trials_factor=self.config.max_hypothesis_trials_factor,
        )

        residual_matrix = compute_residual_matrix(data, hypotheses)

        if self.config.use_aikose:
            scale_batch = estimate_scales(residual_matrix, self.config.aikose)
        else:
            scale_batch = _fixed_scale_batch(hypotheses.shape[0], self.config.fixed_tau)

        preference_matrix, preference_stats = build_preference_matrix(
            residual_matrix,
            scales=scale_batch.taus,
            kernel=self.config.preference_kernel,
        )

        similarity_matrix = compute_point_similarity(
            preference_matrix,
            normalize=self.config.similarity_normalize,
            sparsify=self.config.similarity_sparsify,
            top_k=self.config.similarity_top_k,
            remove_diagonal=self.config.remove_similarity_diagonal,
        )

        voting_result = hierarchical_vote(
            preference_matrix=preference_matrix,
            similarity_matrix=similarity_matrix,
            alpha=self.config.voting_alpha,
            beta=self.config.voting_beta,
        )

        pruning_result = prune_hypotheses(
            hypotheses=hypotheses,
            scores=voting_result.hypothesis_scores,
            config=self.config.pruning,
        )

        pruned_indices = pruning_result.kept_indices
        pruned_hypotheses = pruning_result.pruned_hypotheses
        pruned_residuals = residual_matrix[:, pruned_indices]
        pruned_scales = scale_batch.taus[pruned_indices]
        pruned_preferences = preference_matrix[:, pruned_indices]

        clustering_result = cluster_hypotheses(
            hypotheses=pruned_hypotheses,
            preference_matrix=pruned_preferences,
            config=self.config.clustering,
        )

        extraction_result = extract_model_instances(
            hypotheses=pruned_hypotheses,
            clusters=clustering_result.clusters,
            residuals=pruned_residuals,
            scales=pruned_scales,
            hypothesis_scores=pruning_result.pruned_scores,
            config=self.config.extraction,
        )

        outlier_mask, inlier_mask = detect_outliers(
            outlier_scores=voting_result.outlier_scores,
            quantile=self.config.outlier_quantile,
        )

        runtime = time.perf_counter() - start_time
        LOGGER.info(
            "HVF run complete | hypotheses=%d pruned=%d extracted=%d runtime=%.3fs",
            hypotheses.shape[0],
            pruned_hypotheses.shape[0],
            extraction_result.final_models.shape[0],
            runtime,
        )

        return HVFRunResult(
            hypotheses=hypotheses,
            residual_matrix=residual_matrix,
            scales=scale_batch.taus,
            scale_results=scale_batch.results,
            tau_diagnostics=scale_batch.diagnostics,
            preference_matrix=preference_matrix,
            preference_stats=preference_stats,
            similarity_matrix=similarity_matrix,
            voting=voting_result,
            pruning=pruning_result,
            clustering=clustering_result,
            extraction=extraction_result,
            outlier_mask=outlier_mask,
            inlier_mask=inlier_mask,
            runtime_seconds=runtime,
        )

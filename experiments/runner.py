from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from data.synthetic import generate_synthetic_dataset
from hvf.pipeline import HVFConfig, HVFPipeline
from privacy.dp_pipeline import DPHVFConfig, apply_dp_hvf

from .metrics import build_metric_result
from .reporting import save_experiment_results
from visualization.plots import plot_metric_curve

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    """Structured configuration for batch HVF experiments."""

    mode: str = "baseline"  # baseline | auto | dp
    output_dir: str = "outputs"
    experiment_name: str = "hvf_experiment"
    num_models: int = 2
    total_points: int = 320
    outlier_ratios: tuple[float, ...] = (0.2, 0.35)
    hypothesis_counts: tuple[int, ...] = (400, 700)
    seeds: tuple[int, ...] = (0, 1, 2)
    epsilons: tuple[float, ...] = (1.0, 0.5, 0.1)
    noise_sigma: float = 0.25
    preference_kernel: str = "exponential"
    similarity_normalize: str = "row"
    similarity_sparsify: bool = False
    dp_mechanism: str = "laplace"
    dp_injection_points: tuple[str, ...] = (
        "dp_on_hypothesis_scores",
        "dp_on_model_selection",
    )
    dp_model_selection_top_k: int = 3
    dp_max_contribution: float = 1.0


@dataclass(frozen=True)
class ExperimentResultBundle:
    """Persisted run artifacts and in-memory records."""

    records: list[dict[str, Any]]
    saved_files: dict[str, str]
    plots: list[str]


def _validate_mode(mode: str) -> str:
    value = mode.lower().strip()
    if value not in {"baseline", "auto", "dp"}:
        raise ValueError("mode must be one of: baseline, auto, dp")
    return value


def _build_hvf_config(config: ExperimentConfig, hypothesis_count: int) -> HVFConfig:
    return HVFConfig(
        num_hypotheses=hypothesis_count,
        use_aikose=config.mode in {"auto", "dp"},
        fixed_tau=0.65,
        preference_kernel=config.preference_kernel,
        similarity_normalize=config.similarity_normalize,
        similarity_sparsify=config.similarity_sparsify,
    )


def _summarize_privacy(config: DPHVFConfig, reports: list[Any]) -> dict[str, Any]:
    total_queries = sum(int(report.num_queries) for report in reports)
    total_epsilon = sum(float(report.epsilon) * float(report.num_queries) for report in reports)
    avg_noise_scale = float(np.mean([float(report.noise_scale) for report in reports])) if reports else 0.0

    return {
        "dp_enabled": True,
        "dp_mechanism": config.mechanism,
        "dp_injection_points": json.dumps(config.injection_points),
        "dp_num_queries": total_queries,
        "dp_effective_epsilon": total_epsilon,
        "dp_avg_noise_scale": avg_noise_scale,
    }


def run_experiments(config: ExperimentConfig) -> ExperimentResultBundle:
    """Run full-factorial experiments and save structured results."""
    mode = _validate_mode(config.mode)
    records: list[dict[str, Any]] = []

    epsilon_values = config.epsilons if mode == "dp" else (0.0,)
    condition_grid = product(config.outlier_ratios, config.hypothesis_counts, epsilon_values, config.seeds)

    for outlier_ratio, hypothesis_count, epsilon, seed in condition_grid:
        rng = np.random.default_rng(seed)

        inlier_total = max(int(round(config.total_points * (1.0 - outlier_ratio))), config.num_models * 2)
        points_per_model = max(inlier_total // config.num_models, 2)
        outlier_count = max(config.total_points - points_per_model * config.num_models, 0)

        dataset = generate_synthetic_dataset(
            num_models=config.num_models,
            points_per_model=points_per_model,
            outlier_count=outlier_count,
            noise_sigma=config.noise_sigma,
            rng=rng,
        )

        hvf_config = _build_hvf_config(config, hypothesis_count)
        pipeline = HVFPipeline(hvf_config)
        hvf_result = pipeline.run(dataset.points, rng=rng)

        predicted_models = hvf_result.extraction.final_models
        predicted_labels = hvf_result.extraction.model_labels
        outlier_mask = hvf_result.outlier_mask
        dp_summary: dict[str, Any] = {
            "dp_enabled": False,
            "dp_mechanism": "none",
            "dp_injection_points": "[]",
            "dp_num_queries": 0,
            "dp_effective_epsilon": 0.0,
            "dp_avg_noise_scale": 0.0,
        }

        if mode == "dp":
            dp_config = DPHVFConfig(
                epsilon=float(epsilon),
                mechanism=config.dp_mechanism,
                injection_points=config.dp_injection_points,
                model_selection_top_k=config.dp_model_selection_top_k,
                max_contribution=config.dp_max_contribution,
            )
            dp_result = apply_dp_hvf(hvf_result, config=dp_config, rng=rng)
            dp_summary = _summarize_privacy(dp_config, dp_result.privacy_reports)

            # Recompute ALL downstream outputs from DP-selected models so
            # that metrics reflect full post-DP evaluation.
            if dp_result.selected_model_indices.size > 0:
                predicted_models = hvf_result.hypotheses[dp_result.selected_model_indices]
                # Re-derive labels from the DP-selected hypothesis subset.
                from privacy.dp_pipeline import reconstruct_labels_from_selected_hypotheses
                predicted_labels = reconstruct_labels_from_selected_hypotheses(
                    residual_matrix=hvf_result.residual_matrix,
                    scales=hvf_result.scales,
                    selected_model_indices=dp_result.selected_model_indices,
                )
                outlier_mask = predicted_labels == -1

        metrics = build_metric_result(
            predicted_models=predicted_models,
            predicted_labels=predicted_labels,
            outlier_mask=outlier_mask,
            true_models=dataset.true_models,
            true_labels=dataset.point_labels,
            runtime_seconds=hvf_result.runtime_seconds,
            tau_mean=hvf_result.tau_diagnostics.tau_mean,
            tau_std=hvf_result.tau_diagnostics.tau_std,
        )

        record: dict[str, Any] = {
            "mode": mode,
            "seed": seed,
            "epsilon": float(epsilon),
            "outlier_ratio": float(outlier_ratio),
            "hypothesis_count": int(hypothesis_count),
            "num_models_true": int(dataset.true_models.shape[0]),
            "num_models_pred": int(predicted_models.shape[0]),
            "preference_density": float(hvf_result.preference_stats.density),
            "tau_mean": float(hvf_result.tau_diagnostics.tau_mean),
            "tau_std": float(hvf_result.tau_diagnostics.tau_std),
            **metrics.to_dict(),
            **dp_summary,
        }
        records.append(record)

        LOGGER.info(
            "Experiment run | mode=%s seed=%d outlier=%.2f hyp=%d eps=%.3f f1=%.3f",
            mode,
            seed,
            outlier_ratio,
            hypothesis_count,
            epsilon,
            metrics.inlier_f1,
        )

    saved_files = save_experiment_results(records, config.output_dir, config.experiment_name)

    plots: list[str] = []
    output_root = Path(config.output_dir)

    try:
        plot_1 = output_root / f"{config.experiment_name}_f1_vs_outlier.png"
        plot_metric_curve(records, x_key="outlier_ratio", y_key="inlier_f1", output_path=plot_1)
        plots.append(str(plot_1))

        if mode == "dp":
            plot_2 = output_root / f"{config.experiment_name}_accuracy_vs_epsilon.png"
            plot_metric_curve(records, x_key="epsilon", y_key="model_detection_accuracy", output_path=plot_2)
            plots.append(str(plot_2))
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Plot generation failed: %s", error)

    return ExperimentResultBundle(records=records, saved_files=saved_files, plots=plots)

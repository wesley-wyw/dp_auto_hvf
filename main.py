from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np

from data.loaders import load_point_set
from data.preprocess import center_points, standardize_points
from data.synthetic import generate_synthetic_dataset
from experiments.adelaide_baseline_compare import (
    AdelaideBaselineComparisonConfig,
    run_adelaide_baseline_comparison,
)
from experiments.adelaide_runner import AdelaideExperimentConfig, run_adelaide_experiments
from experiments.runner import ExperimentConfig, run_experiments
from hvf.pipeline import HVFConfig, HVFPipeline
from privacy.dp_pipeline import DPHVFConfig, apply_dp_hvf

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleRunConfig:
    """Configuration for a single reproducible HVF run."""

    mode: str = "baseline"
    dataset_source: str = "synthetic"
    data_path: str | None = None
    data_delimiter: str = ","
    preprocess: str = "none"
    seed: int = 0
    num_models: int = 2
    total_points: int = 320
    outlier_ratio: float = 0.30
    noise_sigma: float = 0.25
    hypothesis_count: int = 600
    epsilon: float = 1.0
    output_dir: str = "outputs"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HVF / Auto-HVF / DP-HVF research pipeline")
    parser.add_argument("--mode", choices=["baseline", "auto", "dp"], default="baseline")
    parser.add_argument("--run-experiments", action="store_true", help="Run synthetic batch experiments")
    parser.add_argument("--run-adelaide-experiments", action="store_true", help="Run AdelaideRMF batch experiments")
    parser.add_argument(
        "--run-adelaide-baseline-comparison",
        action="store_true",
        help="Run Adelaide baseline comparison: HVF-fixed vs Auto-HVF",
    )
    parser.add_argument("--dataset-source", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--data-path", type=str, default=None, help="Path to a real 2D point file")
    parser.add_argument(
        "--data-delimiter",
        type=str,
        default=",",
        help="Delimiter for real point files. Use 'auto' to try comma/tab/whitespace.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["none", "center", "standardize"],
        default="none",
        help="Optional preprocessing for real point files.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-models", type=int, default=2)
    parser.add_argument("--total-points", type=int, default=320)
    parser.add_argument("--outlier-ratio", type=float, default=0.30)
    parser.add_argument("--noise-sigma", type=float, default=0.25)
    parser.add_argument("--hypothesis-count", type=int, default=600)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--dp-epsilons", nargs="+", type=float, default=None, help="Optional epsilon sweep for DP experiments")
    parser.add_argument("--dp-epsilon-allocation", choices=["equal", "manual"], default="equal")
    parser.add_argument("--dp-epsilon-allocations", type=str, default=None, help="Manual epsilon allocations like a=0.6,b=0.4")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="hvf_experiment")
    parser.add_argument("--adelaide-root", type=str, default=None, help="AdelaideRMF file or folder path")
    parser.add_argument(
        "--adelaide-model-families",
        nargs="+",
        default=["fundamental", "homography"],
        help="Model families for Adelaide experiments.",
    )
    parser.add_argument("--adelaide-limit-files", type=int, default=None)
    parser.add_argument("--comparison-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--baseline-fixed-tau", type=float, default=0.05)
    parser.add_argument("--mixed-calibration-modes", choices=["on", "off", "both"], default="on")
    parser.add_argument("--mixed-calibration-quantile", type=float, default=0.5)
    return parser


def _parse_budget_allocations(raw: str | None) -> dict[str, float] | None:
    if raw is None or raw.strip() == "":
        return None

    allocations: dict[str, float] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Allocation entries must be key=value pairs")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Allocation key cannot be empty")
        allocations[key] = float(value.strip())

    return allocations if allocations else None


def _load_run_points(config: SingleRunConfig, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | int | str]]:
    if config.dataset_source == "synthetic":
        inlier_total = max(int(round(config.total_points * (1.0 - config.outlier_ratio))), config.num_models * 2)
        points_per_model = max(inlier_total // config.num_models, 2)
        outlier_count = max(config.total_points - points_per_model * config.num_models, 0)

        dataset = generate_synthetic_dataset(
            num_models=config.num_models,
            points_per_model=points_per_model,
            outlier_count=outlier_count,
            noise_sigma=config.noise_sigma,
            rng=rng,
        )
        metadata: dict[str, float | int | str] = {
            "dataset_source": "synthetic",
            "num_true_models": int(dataset.true_models.shape[0]),
            "num_input_points": int(dataset.points.shape[0]),
        }
        return dataset.points, metadata

    if not config.data_path:
        raise ValueError("--data-path is required when --dataset-source=csv")

    point_set = load_point_set(config.data_path, delimiter=config.data_delimiter)
    points = point_set.points
    if config.preprocess == "center":
        points, _ = center_points(points)
    elif config.preprocess == "standardize":
        points, _, _ = standardize_points(points)

    metadata = {
        "dataset_source": "csv",
        "data_path": point_set.source_path,
        "num_input_points": point_set.num_points,
        "preprocess": config.preprocess,
    }
    return points, metadata


def _run_single(config: SingleRunConfig) -> dict[str, float | int | str]:
    rng = np.random.default_rng(config.seed)
    points, dataset_metadata = _load_run_points(config, rng)

    hvf_config = HVFConfig(
        num_hypotheses=config.hypothesis_count,
        use_aikose=config.mode in {"auto", "dp"},
    )
    pipeline = HVFPipeline(hvf_config)
    hvf_result = pipeline.run(points, rng=rng)

    summary: dict[str, float | int | str] = {
        "mode": config.mode,
        "seed": config.seed,
        "num_extracted_models": int(hvf_result.extraction.final_models.shape[0]),
        "runtime_seconds": float(hvf_result.runtime_seconds),
        "tau_mean": float(hvf_result.tau_diagnostics.tau_mean),
        "tau_std": float(hvf_result.tau_diagnostics.tau_std),
        "preference_density": float(hvf_result.preference_stats.density),
        "outlier_count_detected": int(np.count_nonzero(hvf_result.outlier_mask)),
        **dataset_metadata,
    }

    if config.mode == "dp":
        dp_config = DPHVFConfig(
            epsilon=config.epsilon,
            injection_points=(
                "dp_on_hypothesis_scores",
                "dp_on_point_scores",
                "dp_on_model_selection",
            ),
        )
        dp_result = apply_dp_hvf(hvf_result, config=dp_config, rng=rng)
        summary.update(
            {
                "dp_effective_queries": len(dp_result.privacy_reports),
                "dp_selected_models": int(dp_result.selected_model_indices.size),
                "dp_sensitivity": float(dp_result.sensitivity.hypothesis_score_sensitivity),
            }
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_tag = config.dataset_source
    summary_path = output_dir / f"single_run_{config.mode}_{dataset_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Single-run summary saved to %s", summary_path)

    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.run_adelaide_baseline_comparison:
        if not args.adelaide_root:
            raise ValueError("--adelaide-root is required with --run-adelaide-baseline-comparison")
        bundle = run_adelaide_baseline_comparison(
            AdelaideBaselineComparisonConfig(
                dataset_root=args.adelaide_root,
                output_dir=args.output_dir,
                experiment_name=args.experiment_name,
                model_families=tuple(args.adelaide_model_families),
                num_hypotheses=args.hypothesis_count,
                seeds=tuple(args.comparison_seeds),
                fixed_tau=float(args.baseline_fixed_tau),
                limit_files=args.adelaide_limit_files,
                mixed_calibration_quantile=float(args.mixed_calibration_quantile),
            )
        )
        LOGGER.info(
            "Adelaide baseline comparison finished | runs=%d csv=%s summary=%s",
            len(bundle.records),
            bundle.saved_files.get("csv", ""),
            bundle.saved_files.get("summary_csv", ""),
        )
        return

    if args.run_adelaide_experiments:
        calibration_modes_map = {"on": (True,), "off": (False,), "both": (False, True)}
        calibration_modes = calibration_modes_map[args.mixed_calibration_modes]
        if not args.adelaide_root:
            raise ValueError("--adelaide-root is required with --run-adelaide-experiments")
        bundle = run_adelaide_experiments(
            AdelaideExperimentConfig(
                dataset_root=args.adelaide_root,
                output_dir=args.output_dir,
                experiment_name=args.experiment_name,
                model_families=tuple(args.adelaide_model_families),
                num_hypotheses=args.hypothesis_count,
                use_aikose=args.mode in {"auto", "dp"},
                fixed_tau=0.05,
                seeds=(args.seed,),
                limit_files=args.adelaide_limit_files,
                enable_dp=args.mode == "dp",
                dp_epsilon=args.epsilon,
                dp_epsilons=tuple(args.dp_epsilons) if args.dp_epsilons else None,
                mixed_calibration_modes=calibration_modes,
                mixed_calibration_quantile=float(args.mixed_calibration_quantile),
                dp_epsilon_allocation=args.dp_epsilon_allocation,
                dp_epsilon_allocations=_parse_budget_allocations(args.dp_epsilon_allocations),
            )
        )
        LOGGER.info(
            "Adelaide experiment finished | runs=%d csv=%s json=%s",
            len(bundle.records),
            bundle.saved_files.get("csv", ""),
            bundle.saved_files.get("json", ""),
        )
        return

    if args.run_experiments:
        experiment_config = ExperimentConfig(
            mode=args.mode,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            num_models=args.num_models,
            total_points=args.total_points,
            epsilons=(args.epsilon, 0.5, 0.1) if args.mode == "dp" else (0.0,),
        )
        bundle = run_experiments(experiment_config)
        LOGGER.info(
            "Experiment finished | runs=%d csv=%s json=%s",
            len(bundle.records),
            bundle.saved_files.get("csv", ""),
            bundle.saved_files.get("json", ""),
        )
        return

    single_run_config = SingleRunConfig(
        mode=args.mode,
        dataset_source=args.dataset_source,
        data_path=args.data_path,
        data_delimiter=args.data_delimiter,
        preprocess=args.preprocess,
        seed=args.seed,
        num_models=args.num_models,
        total_points=args.total_points,
        outlier_ratio=args.outlier_ratio,
        noise_sigma=args.noise_sigma,
        hypothesis_count=args.hypothesis_count,
        epsilon=args.epsilon,
        output_dir=args.output_dir,
    )
    summary = _run_single(single_run_config)
    LOGGER.info("Run summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

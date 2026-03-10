from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

import numpy as np

from data.synthetic import generate_synthetic_dataset
from experiments.runner import ExperimentConfig, run_experiments
from hvf.pipeline import HVFConfig, HVFPipeline
from privacy.dp_pipeline import DPHVFConfig, apply_dp_hvf

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleRunConfig:
    """Configuration for a single reproducible HVF run."""

    mode: str = "baseline"
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
    parser.add_argument("--run-experiments", action="store_true", help="Run batch experiments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-models", type=int, default=2)
    parser.add_argument("--total-points", type=int, default=320)
    parser.add_argument("--outlier-ratio", type=float, default=0.30)
    parser.add_argument("--noise-sigma", type=float, default=0.25)
    parser.add_argument("--hypothesis-count", type=int, default=600)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="hvf_experiment")
    return parser


def _run_single(config: SingleRunConfig) -> dict[str, float | int | str]:
    rng = np.random.default_rng(config.seed)

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

    hvf_config = HVFConfig(
        num_hypotheses=config.hypothesis_count,
        use_aikose=config.mode in {"auto", "dp"},
    )
    pipeline = HVFPipeline(hvf_config)
    hvf_result = pipeline.run(dataset.points, rng=rng)

    summary: dict[str, float | int | str] = {
        "mode": config.mode,
        "seed": config.seed,
        "num_true_models": int(dataset.true_models.shape[0]),
        "num_extracted_models": int(hvf_result.extraction.final_models.shape[0]),
        "runtime_seconds": float(hvf_result.runtime_seconds),
        "tau_mean": float(hvf_result.tau_diagnostics.tau_mean),
        "tau_std": float(hvf_result.tau_diagnostics.tau_std),
        "preference_density": float(hvf_result.preference_stats.density),
        "outlier_count_detected": int(np.count_nonzero(hvf_result.outlier_mask)),
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
    summary_path = output_dir / f"single_run_{config.mode}.json"
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

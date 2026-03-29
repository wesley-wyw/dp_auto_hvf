"""Experiment A: Ranking flip verification under differential privacy.

Directly measures how DP noise changes candidate model rankings,
closing the evidence chain: score perturbation → ranking change → ME% degradation.

Metrics:
  1. Kendall τ between clean and noisy rankings
  2. Pairwise flip count among top-K candidates
  3. Top-K set overlap (Jaccard similarity)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau

from data.adelaide import MixedResidualCalibrationConfig, correspondence_set_from_adelaide
from data.loaders import list_adelaide_rmf_files, load_adelaide_rmf_sample
from hvf.adelaide_pipeline import AdelaideHVFConfig, run_adelaide_hvf
from privacy.dp_pipeline import DPHVFConfig, apply_dp_adelaide_hvf

from .adelaide_runner import build_adelaide_metric_result


@dataclass(frozen=True)
class RankingFlipConfig:
    dataset_root: str
    output_dir: str = "outputs"
    model_family: str = "homography"
    num_hypotheses: int = 240
    num_seeds: int = 5
    epsilons: tuple[float, ...] = (2.0, 1.0, 0.5, 0.2)
    mixed_calibration_quantile: float = 0.5
    limit_files: int | None = None


@dataclass(frozen=True)
class RankingFlipResult:
    records: list[dict[str, Any]]
    table_text: str
    csv_path: str


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def _kendall_tau(clean_scores: np.ndarray, noisy_scores: np.ndarray) -> float:
    """Kendall τ between rankings derived from two score vectors."""
    if clean_scores.size <= 1:
        return 1.0
    tau, _ = kendalltau(clean_scores, noisy_scores)
    if np.isnan(tau):
        return 1.0
    return float(tau)


def _pairwise_flip_count(clean_scores: np.ndarray, noisy_scores: np.ndarray, top_k: int) -> int:
    """Count pairwise ranking flips among the top-K candidates (by clean ranking)."""
    top_k = min(top_k, clean_scores.size)
    top_indices = np.argsort(-clean_scores)[:top_k]
    flips = 0
    for i in range(top_k):
        for j in range(i + 1, top_k):
            a, b = top_indices[i], top_indices[j]
            clean_order = clean_scores[a] > clean_scores[b]
            noisy_order = noisy_scores[a] > noisy_scores[b]
            if clean_order != noisy_order:
                flips += 1
    return flips


def _top_k_jaccard(clean_scores: np.ndarray, noisy_scores: np.ndarray, top_k: int) -> float:
    """Jaccard similarity between top-K sets from clean vs noisy scores."""
    top_k = min(top_k, clean_scores.size)
    clean_top = set(np.argsort(-clean_scores)[:top_k].tolist())
    noisy_top = set(np.argsort(-noisy_scores)[:top_k].tolist())
    if not clean_top:
        return 1.0
    intersection = len(clean_top & noisy_top)
    union = len(clean_top | noisy_top)
    return float(intersection) / float(union) if union > 0 else 1.0


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run_one_sample_seed(
    correspondences: Any,
    sample: Any,
    config: RankingFlipConfig,
    epsilon: float | None,
    seed: int,
) -> dict[str, Any]:
    """Run one sample at one seed, return ranking metrics."""
    rng = np.random.default_rng(seed)
    calibration_enabled = config.model_family == "mixed"

    hvf_config = AdelaideHVFConfig(
        model_family=config.model_family,
        num_hypotheses=config.num_hypotheses,
        use_aikose=True,
        mixed_residual_calibration=MixedResidualCalibrationConfig(
            enabled=calibration_enabled,
            quantile=config.mixed_calibration_quantile,
        ),
    )
    result = run_adelaide_hvf(correspondences, hvf_config, rng)

    # Clean scores (before DP)
    clean_scores = result.voting.hypothesis_scores.copy()
    true_model_count = int(np.max(sample.labels))  # labels: 0=outlier, 1..K=models
    top_k = max(true_model_count, 1)

    if epsilon is not None:
        dp_rng = np.random.default_rng(seed * 10000 + hash(str(epsilon)) % 9999)
        dp_config = DPHVFConfig(
            epsilon=epsilon,
            injection_points=("dp_on_hypothesis_scores",),
        )
        dp_result = apply_dp_adelaide_hvf(result, dp_config, dp_rng)
        noisy_scores = dp_result.dp_result.noisy_hypothesis_scores
        predicted_labels = dp_result.model_labels
    else:
        # No DP baseline
        noisy_scores = clean_scores.copy()
        predicted_labels = result.extraction.model_labels

    # Compute ranking metrics
    tau = _kendall_tau(clean_scores, noisy_scores)
    flips = _pairwise_flip_count(clean_scores, noisy_scores, top_k)
    jaccard = _top_k_jaccard(clean_scores, noisy_scores, top_k)

    # Compute ME%
    metrics = build_adelaide_metric_result(predicted_labels, sample.labels)

    return {
        "kendall_tau": tau,
        "flip_count": flips,
        "top_k_jaccard": jaccard,
        "misclassification_error": metrics.misclassification_error,
        "top_k": top_k,
    }


def run_ranking_flip_experiment(config: RankingFlipConfig) -> RankingFlipResult:
    """Run the ranking flip experiment across all samples, seeds, and epsilons."""
    files = list_adelaide_rmf_files(config.dataset_root)
    if config.limit_files is not None:
        files = files[: max(int(config.limit_files), 0)]

    seeds = list(range(config.num_seeds))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Conditions: no-DP baseline + each epsilon
    conditions: list[tuple[str, float | None]] = [("no DP", None)]
    for eps in config.epsilons:
        conditions.append((f"ε={eps}", eps))

    all_records: list[dict[str, Any]] = []

    for file_path in files:
        sample = load_adelaide_rmf_sample(file_path)
        correspondences = correspondence_set_from_adelaide(sample)
        sample_name = Path(sample.source_path).stem

        for condition_name, epsilon in conditions:
            seed_results: list[dict[str, Any]] = []
            for seed in seeds:
                try:
                    r = _run_one_sample_seed(correspondences, sample, config, epsilon, seed)
                    seed_results.append(r)
                except Exception:
                    continue

            if not seed_results:
                continue

            all_records.append({
                "sample": sample_name,
                "condition": condition_name,
                "epsilon": epsilon if epsilon is not None else float("inf"),
                "kendall_tau": float(np.mean([r["kendall_tau"] for r in seed_results])),
                "flip_count": float(np.mean([r["flip_count"] for r in seed_results])),
                "top_k_jaccard": float(np.mean([r["top_k_jaccard"] for r in seed_results])),
                "me_pct": float(np.mean([r["misclassification_error"] for r in seed_results])) * 100.0,
                "num_seeds": len(seed_results),
            })

    # Format output
    table_text = _format_table(all_records, conditions)

    csv_path = output_dir / f"ranking_flip_{config.model_family}.csv"
    _save_csv(all_records, csv_path)

    txt_path = output_dir / f"ranking_flip_{config.model_family}.txt"
    txt_path.write_text(table_text, encoding="utf-8")

    return RankingFlipResult(
        records=all_records,
        table_text=table_text,
        csv_path=str(csv_path),
    )


def _format_table(
    records: list[dict[str, Any]],
    conditions: list[tuple[str, float | None]],
) -> str:
    """Format the ranking flip results into a summary table.

    Rows = epsilon conditions, columns = averaged metrics.
    """
    if not records:
        return "(no results)\n"

    condition_names = [name for name, _ in conditions]
    lines: list[str] = []

    lines.append(f"{'Condition':<12} {'Kendall τ':>12} {'Flip count':>12} {'Top-K Jaccard':>14} {'ME%':>10}")
    lines.append("-" * 64)

    for cond_name in condition_names:
        cond_records = [r for r in records if r["condition"] == cond_name]
        if not cond_records:
            continue
        avg_tau = float(np.mean([r["kendall_tau"] for r in cond_records]))
        avg_flips = float(np.mean([r["flip_count"] for r in cond_records]))
        avg_jaccard = float(np.mean([r["top_k_jaccard"] for r in cond_records]))
        avg_me = float(np.mean([r["me_pct"] for r in cond_records]))
        lines.append(
            f"{cond_name:<12} {avg_tau:>12.4f} {avg_flips:>12.2f} {avg_jaccard:>14.4f} {avg_me:>10.2f}"
        )

    lines.append("-" * 64)
    n_seeds = records[0]["num_seeds"] if records else "?"
    lines.append(f"(averaged over {n_seeds} seeds × {len(set(r['sample'] for r in records))} samples)")
    return "\n".join(lines)


def _save_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    lines = [",".join(keys)]
    for r in records:
        lines.append(",".join(str(r.get(k, "")) for k in keys))
    path.write_text("\n".join(lines), encoding="utf-8")

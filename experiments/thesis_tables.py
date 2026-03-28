"""Generate thesis-ready comparison tables following Xiao et al. (2018) format.

Output format matches the standard used in the HVF reference paper:
- Rows: datasets (samples)
- Columns: methods (HVF-fixed, Auto-HVF, DP-HVF variants)
- Primary metric: misclassification error rate (%) + runtime (s)
- Averaged over N seeds, bold for best result
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.adelaide import MixedResidualCalibrationConfig, correspondence_set_from_adelaide
from data.loaders import list_adelaide_rmf_files, load_adelaide_rmf_sample
from hvf.adelaide_pipeline import AdelaideHVFConfig, run_adelaide_hvf
from privacy.dp_pipeline import DPHVFConfig, apply_dp_adelaide_hvf

from visualization.plots import plot_thesis_fitting_result, plot_thesis_multi_method_comparison
from .adelaide_runner import build_adelaide_metric_result


@dataclass(frozen=True)
class ThesisExperimentConfig:
    """Minimal config to produce thesis-quality comparison tables."""

    dataset_root: str
    output_dir: str = "outputs"
    model_family: str = "homography"
    num_hypotheses: int = 240
    num_seeds: int = 10
    fixed_tau: float = 0.05
    fixed_tau_tuned: float | None = None
    dp_epsilons: tuple[float, ...] = (2.0, 1.0, 0.5, 0.2)
    mixed_calibration_quantile: float = 0.5
    limit_files: int | None = None


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run_single(
    correspondences: Any,
    sample: Any,
    model_family: str,
    config: ThesisExperimentConfig,
    *,
    use_aikose: bool,
    fixed_tau: float,
    dp_epsilon: float | None,
    seed: int,
) -> dict[str, Any]:
    """Run one pipeline configuration on one sample with one seed."""
    rng = np.random.default_rng(seed)
    calibration_enabled = model_family == "mixed"

    hvf_config = AdelaideHVFConfig(
        model_family=model_family,
        num_hypotheses=config.num_hypotheses,
        use_aikose=use_aikose,
        fixed_tau=fixed_tau,
        mixed_residual_calibration=MixedResidualCalibrationConfig(
            enabled=calibration_enabled,
            quantile=config.mixed_calibration_quantile,
        ),
    )
    result = run_adelaide_hvf(correspondences, hvf_config, rng)

    if dp_epsilon is not None and dp_epsilon > 0:
        dp_rng = np.random.default_rng(seed * 10000 + hash(str(dp_epsilon)) % 9999)
        dp_config = DPHVFConfig(
            epsilon=dp_epsilon,
            injection_points=("dp_on_hypothesis_scores",),
        )
        dp_result = apply_dp_adelaide_hvf(result, dp_config, dp_rng)
        predicted_labels = dp_result.model_labels
    else:
        predicted_labels = result.extraction.model_labels

    metrics = build_adelaide_metric_result(predicted_labels, sample.labels)
    return {
        "misclassification_error": metrics.misclassification_error,
        "inlier_f1": metrics.inlier_f1,
        "runtime_seconds": result.runtime_seconds,
        "predicted_model_count": metrics.predicted_model_count,
        "true_model_count": metrics.true_model_count,
    }


MethodSpec = tuple[str, bool, float, float | None]  # (name, use_aikose, fixed_tau, dp_epsilon)


def _build_method_specs(config: ThesisExperimentConfig) -> list[MethodSpec]:
    """Build the list of method variants to compare."""
    specs: list[MethodSpec] = [
        ("HVF-fixed", False, config.fixed_tau, None),
        ("Auto-HVF", True, config.fixed_tau, None),
    ]
    if config.fixed_tau_tuned is not None:
        specs.insert(1, (f"HVF(t={config.fixed_tau_tuned})", False, config.fixed_tau_tuned, None))
    for eps in config.dp_epsilons:
        specs.append((f"DP-HVF(e={eps})", True, config.fixed_tau, eps))
    return specs


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

@dataclass
class ThesisTableResult:
    """Holds the generated table data and formatted strings."""
    records: list[dict[str, Any]]
    table_text: str
    csv_path: str
    table_path: str
    figure_paths: list[str]


def _run_single_with_labels(
    correspondences: Any,
    sample: Any,
    model_family: str,
    config: ThesisExperimentConfig,
    *,
    use_aikose: bool,
    fixed_tau: float,
    dp_epsilon: float | None,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Like _run_single but also returns predicted labels for visualization."""
    rng = np.random.default_rng(seed)
    calibration_enabled = model_family == "mixed"

    hvf_config = AdelaideHVFConfig(
        model_family=model_family,
        num_hypotheses=config.num_hypotheses,
        use_aikose=use_aikose,
        fixed_tau=fixed_tau,
        mixed_residual_calibration=MixedResidualCalibrationConfig(
            enabled=calibration_enabled,
            quantile=config.mixed_calibration_quantile,
        ),
    )
    result = run_adelaide_hvf(correspondences, hvf_config, rng)

    if dp_epsilon is not None and dp_epsilon > 0:
        # Use independent rng for DP noise (seeded from original seed + epsilon)
        # so that different epsilon values produce genuinely different noise.
        dp_rng = np.random.default_rng(seed * 10000 + hash(str(dp_epsilon)) % 9999)
        dp_config = DPHVFConfig(
            epsilon=dp_epsilon,
            injection_points=("dp_on_hypothesis_scores",),
        )
        dp_result = apply_dp_adelaide_hvf(result, dp_config, dp_rng)
        predicted_labels = dp_result.model_labels
    else:
        predicted_labels = result.extraction.model_labels

    metrics = build_adelaide_metric_result(predicted_labels, sample.labels)
    record = {
        "misclassification_error": metrics.misclassification_error,
        "inlier_f1": metrics.inlier_f1,
        "runtime_seconds": result.runtime_seconds,
        "predicted_model_count": metrics.predicted_model_count,
        "true_model_count": metrics.true_model_count,
    }
    return record, np.asarray(predicted_labels, dtype=int)


def run_thesis_experiment(config: ThesisExperimentConfig) -> ThesisTableResult:
    """Run all methods on all samples and produce a formatted comparison table."""
    files = list_adelaide_rmf_files(config.dataset_root)
    if config.limit_files is not None:
        files = files[: max(int(config.limit_files), 0)]

    method_specs = _build_method_specs(config)
    seeds = list(range(config.num_seeds))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-sample, per-method averaged results
    all_records: list[dict[str, Any]] = []
    figure_paths: list[str] = []

    for file_path in files:
        sample = load_adelaide_rmf_sample(file_path)
        correspondences = correspondence_set_from_adelaide(sample)
        sample_name = Path(sample.source_path).stem

        # For figures: collect seed=0 results per method
        figure_method_results: list[tuple[str, np.ndarray, float]] = []

        for method_name, use_aikose, fixed_tau, dp_eps in method_specs:
            seed_results: list[dict[str, Any]] = []
            seed0_labels: np.ndarray | None = None

            for seed in seeds:
                try:
                    result, labels = _run_single_with_labels(
                        correspondences, sample, config.model_family, config,
                        use_aikose=use_aikose, fixed_tau=fixed_tau,
                        dp_epsilon=dp_eps, seed=seed,
                    )
                    seed_results.append(result)
                    if seed == 0:
                        seed0_labels = labels
                except Exception:
                    continue

            if not seed_results:
                continue

            avg_error = float(np.mean([r["misclassification_error"] for r in seed_results]))
            avg_runtime = float(np.mean([r["runtime_seconds"] for r in seed_results]))
            avg_f1 = float(np.mean([r["inlier_f1"] for r in seed_results]))

            all_records.append({
                "sample": sample_name,
                "method": method_name,
                "model_family": config.model_family,
                "error_pct": avg_error * 100.0,
                "runtime": avg_runtime,
                "inlier_f1": avg_f1,
                "num_seeds": len(seed_results),
            })

            if seed0_labels is not None:
                figure_method_results.append((method_name, seed0_labels, avg_error * 100.0))

        # Generate per-sample multi-method comparison figure
        if figure_method_results and correspondences.image_points_1 is not None:
            points_2d = correspondences.image_points_1
            fig_path = output_dir / f"thesis_fig_{config.model_family}_{sample_name}.png"
            try:
                plot_thesis_multi_method_comparison(
                    points_2d,
                    sample.labels,
                    figure_method_results,
                    fig_path,
                    sample_name=sample_name,
                    image=sample.image_1,
                )
                figure_paths.append(str(fig_path))
            except Exception:
                pass

    # Format table
    table_text = _format_comparison_table(all_records, method_specs)

    csv_path = output_dir / f"thesis_table_{config.model_family}.csv"
    _save_csv(all_records, csv_path)

    table_path = output_dir / f"thesis_table_{config.model_family}.txt"
    table_path.write_text(table_text, encoding="utf-8")

    return ThesisTableResult(
        records=all_records,
        table_text=table_text,
        csv_path=str(csv_path),
        table_path=str(table_path),
        figure_paths=figure_paths,
    )


def _format_comparison_table(
    records: list[dict[str, Any]],
    method_specs: list[MethodSpec],
) -> str:
    """Format records into a thesis-ready text table.

    Style: rows = datasets, columns = methods
    Each cell: error% | time(s)
    Best error per row in brackets [].
    """
    if not records:
        return "(no results)\n"

    method_names = [name for name, _, _, _ in method_specs]
    samples = sorted({r["sample"] for r in records})

    # Build lookup: (sample, method) -> (error_pct, runtime)
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for r in records:
        lookup[(r["sample"], r["method"])] = (r["error_pct"], r["runtime"])

    # Column widths
    sample_width = max(len(s) for s in samples) + 2
    col_width = max(max(len(m) for m in method_names) + 2, 18)

    lines: list[str] = []

    # Header
    header = f"{'Dataset':<{sample_width}}"
    for m in method_names:
        header += f" {m:^{col_width}}"
    lines.append(header)

    sub_header = f"{'':<{sample_width}}"
    for _ in method_names:
        sub_header += f" {'err% | time(s)':^{col_width}}"
    lines.append(sub_header)
    lines.append("-" * len(header))

    # Data rows
    for sample in samples:
        row_errors = {m: lookup.get((sample, m), (None, None))[0] for m in method_names}
        valid_errors = [e for e in row_errors.values() if e is not None]
        best_error = min(valid_errors) if valid_errors else None

        row = f"{sample:<{sample_width}}"
        for m in method_names:
            error, runtime = lookup.get((sample, m), (None, None))
            if error is None:
                cell = "—"
            else:
                err_str = f"{error:.2f}"
                if best_error is not None and abs(error - best_error) < 0.005:
                    err_str = f"[{err_str}]"
                cell = f"{err_str} | {runtime:.2f}"
            row += f" {cell:^{col_width}}"
        lines.append(row)

    lines.append("-" * len(header))

    # Average row
    avg_row = f"{'Average':<{sample_width}}"
    avg_errors: dict[str, float] = {}
    for m in method_names:
        errors = [lookup[(s, m)][0] for s in samples if (s, m) in lookup]
        runtimes = [lookup[(s, m)][1] for s in samples if (s, m) in lookup]
        if errors:
            avg_e = float(np.mean(errors))
            avg_r = float(np.mean(runtimes))
            avg_errors[m] = avg_e
            cell = f"{avg_e:.2f} | {avg_r:.2f}"
        else:
            cell = "—"
        avg_row += f" {cell:^{col_width}}"
    lines.append(avg_row)

    lines.append("")
    lines.append(f"(averaged over {records[0]['num_seeds'] if records else '?'} seeds, [] = best per row)")
    return "\n".join(lines)


def _save_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    lines = [",".join(keys)]
    for r in records:
        lines.append(",".join(str(r.get(k, "")) for k in keys))
    path.write_text("\n".join(lines), encoding="utf-8")

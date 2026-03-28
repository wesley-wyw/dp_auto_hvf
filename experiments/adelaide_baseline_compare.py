from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from data.adelaide import MixedResidualCalibrationConfig, correspondence_set_from_adelaide
from data.loaders import list_adelaide_rmf_files, load_adelaide_rmf_sample
from hvf.adelaide_pipeline import AdelaideHVFConfig, run_adelaide_hvf

from .adelaide_runner import build_adelaide_metric_result
from .reporting import save_csv_records, save_experiment_results, save_json_records


@dataclass(frozen=True)
class AdelaideBaselineComparisonConfig:
    """Config for a thesis-ready HVF-fixed vs Auto-HVF comparison on AdelaideRMF."""

    dataset_root: str
    output_dir: str = "outputs"
    experiment_name: str = "adelaide_baseline_compare"
    model_families: tuple[str, ...] = ("fundamental", "homography", "mixed")
    num_hypotheses: int = 20
    seeds: tuple[int, ...] = (0, 1, 2)
    fixed_tau: float = 0.05
    limit_files: int | None = None
    mixed_calibration_quantile: float = 0.5


@dataclass(frozen=True)
class AdelaideBaselineComparisonBundle:
    """Saved artifacts and in-memory outputs for the Adelaide baseline comparison."""

    records: list[dict[str, Any]]
    summary_records: list[dict[str, Any]]
    delta_records: list[dict[str, Any]]
    saved_files: dict[str, str]
    plots: list[str]


METHOD_VARIANTS: tuple[tuple[str, bool], ...] = (
    ("HVF-fixed", False),
    ("Auto-HVF", True),
)

PLOT_FORMATS: tuple[str, ...] = ("png", "svg", "pdf")


def _build_summary_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []

    numeric_keys = (
        "misclassification_error",
        "inlier_f1",
        "inlier_precision",
        "inlier_recall",
        "ari_inlier",
        "nmi_inlier",
        "match_iou_mean",
        "model_count_error",
        "predicted_model_count",
        "true_model_count",
        "runtime_seconds",
        "tau_mean",
        "tau_std",
    )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["method_variant"]), str(record["model_family"]))
        grouped.setdefault(key, []).append(record)

    summary_records: list[dict[str, Any]] = []
    for (method_variant, model_family), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "method_variant": method_variant,
            "model_family": model_family,
            "use_aikose": any(bool(item["use_aikose"]) for item in group),
            "fixed_tau": float(group[0]["fixed_tau"]),
            "num_runs": len(group),
            "num_samples": len({str(item["sample_name"]) for item in group}),
        }
        for key in numeric_keys:
            values = [float(item[key]) for item in group]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
        summary_records.append(summary)

    return summary_records


def _build_delta_records(summary_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not summary_records:
        return []

    by_family: dict[str, dict[str, dict[str, Any]]] = {}
    for record in summary_records:
        family = str(record["model_family"])
        method = str(record["method_variant"])
        by_family.setdefault(family, {})[method] = record

    delta_records: list[dict[str, Any]] = []
    for family, family_records in sorted(by_family.items()):
        baseline = family_records.get("HVF-fixed")
        auto = family_records.get("Auto-HVF")
        if baseline is None or auto is None:
            continue

        delta_records.append(
            {
                "model_family": family,
                "fixed_tau": float(baseline["fixed_tau"]),
                "baseline_inlier_f1_mean": float(baseline["inlier_f1_mean"]),
                "auto_inlier_f1_mean": float(auto["inlier_f1_mean"]),
                "delta_inlier_f1": float(auto["inlier_f1_mean"]) - float(baseline["inlier_f1_mean"]),
                "baseline_model_count_error_mean": float(baseline["model_count_error_mean"]),
                "auto_model_count_error_mean": float(auto["model_count_error_mean"]),
                "delta_model_count_error": float(auto["model_count_error_mean"]) - float(baseline["model_count_error_mean"]),
                "baseline_match_iou_mean": float(baseline["match_iou_mean_mean"]),
                "auto_match_iou_mean": float(auto["match_iou_mean_mean"]),
                "delta_match_iou_mean": float(auto["match_iou_mean_mean"]) - float(baseline["match_iou_mean_mean"]),
                "baseline_runtime_seconds_mean": float(baseline["runtime_seconds_mean"]),
                "auto_runtime_seconds_mean": float(auto["runtime_seconds_mean"]),
                "delta_runtime_seconds": float(auto["runtime_seconds_mean"]) - float(baseline["runtime_seconds_mean"]),
            }
        )

    return delta_records


def _plot_method_comparison(summary_records: list[dict[str, Any]], output_path: str | Path) -> str | None:
    if not summary_records:
        return None

    families = sorted({str(item["model_family"]) for item in summary_records})
    if not families:
        return None

    metrics = (
        ("inlier_f1_mean", "inlier_f1_std", "Inlier F1", False),
        ("model_count_error_mean", "model_count_error_std", "Model Count Error", True),
        ("match_iou_mean_mean", "match_iou_mean_std", "Match IoU", False),
        ("runtime_seconds_mean", "runtime_seconds_std", "Runtime (s)", True),
    )

    method_colors = {"HVF-fixed": "#4f6d7a", "Auto-HVF": "#2a9d8f"}
    method_offsets = {"HVF-fixed": -0.18, "Auto-HVF": 0.18}
    positions = np.arange(len(families), dtype=float)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), facecolor="white")
    axes_flat = axes.flatten()

    for axis, (mean_key, std_key, title, lower_is_better) in zip(axes_flat, metrics, strict=False):
        for method_variant in ("HVF-fixed", "Auto-HVF"):
            family_values: list[float] = []
            family_stds: list[float] = []
            for family in families:
                record = next(
                    (
                        item
                        for item in summary_records
                        if str(item["model_family"]) == family and str(item["method_variant"]) == method_variant
                    ),
                    None,
                )
                family_values.append(float(record[mean_key]) if record is not None else 0.0)
                family_stds.append(float(record[std_key]) if record is not None else 0.0)

            axis.bar(
                positions + method_offsets[method_variant],
                family_values,
                width=0.34,
                yerr=family_stds,
                capsize=4,
                color=method_colors[method_variant],
                alpha=0.92,
                label=method_variant,
            )

        axis.set_xticks(positions, families)
        axis.set_title(f"{title}{' (lower is better)' if lower_is_better else ''}")
        axis.grid(axis="y", alpha=0.25)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Adelaide Baseline Comparison: HVF-fixed vs Auto-HVF", fontsize=14, weight="bold", y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.07, right=0.98, hspace=0.32, wspace=0.22)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return str(output)


def _plot_delta_summary(delta_records: list[dict[str, Any]], output_path: str | Path) -> str | None:
    if not delta_records:
        return None

    families = [str(item["model_family"]) for item in delta_records]
    positions = np.arange(len(families), dtype=float)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metrics = (
        ("delta_inlier_f1", "Inlier F1 Delta", "#2a9d8f", False),
        ("delta_match_iou_mean", "Match IoU Delta", "#4e8da3", False),
        ("delta_model_count_error", "Model Count Error Delta", "#e67e22", True),
        ("delta_runtime_seconds", "Runtime Delta (s)", "#7f8c8d", True),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), facecolor="white")
    axes_flat = axes.flatten()

    for axis, (metric_key, title, color, lower_is_better) in zip(axes_flat, metrics, strict=False):
        values = [float(item[metric_key]) for item in delta_records]
        bars = axis.bar(positions, values, width=0.52, color=color, alpha=0.9)
        axis.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.8)
        axis.set_xticks(positions, families)
        axis.set_title(f"{title}{' (lower is better)' if lower_is_better else ''}")
        axis.grid(axis="y", alpha=0.25)

        for bar, value in zip(bars, values, strict=False):
            offset = 0.01 if value >= 0 else -0.01
            va = "bottom" if value >= 0 else "top"
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + offset,
                f"{value:+.3f}",
                ha="center",
                va=va,
                fontsize=8.5,
                color="#222222",
            )

    fig.suptitle("Auto-HVF Delta over HVF-fixed", fontsize=14, weight="bold", y=0.98)
    fig.subplots_adjust(top=0.89, bottom=0.10, left=0.07, right=0.98, hspace=0.36, wspace=0.24)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return str(output)


def _save_plot_bundle(
    plotter: Callable[[list[dict[str, Any]], str | Path], str | None],
    output_stem: Path,
    payload: list[dict[str, Any]],
) -> list[str]:
    saved_paths: list[str] = []
    for extension in PLOT_FORMATS:
        output_path = output_stem.with_suffix(f".{extension}")
        result = plotter(payload, output_path)
        if result is not None:
            saved_paths.append(result)
    return saved_paths


def _build_markdown_report(
    config: AdelaideBaselineComparisonConfig,
    summary_records: list[dict[str, Any]],
    delta_records: list[dict[str, Any]],
    plots: list[str],
) -> str:
    lines: list[str] = [
        f"# Adelaide Baseline Comparison Report: `{config.experiment_name}`",
        "",
        "## Configuration",
        "",
        f"- Dataset root: `{config.dataset_root}`",
        f"- Model families: `{', '.join(config.model_families)}`",
        f"- Hypotheses per run: `{config.num_hypotheses}`",
        f"- Seeds: `{', '.join(str(seed) for seed in config.seeds)}`",
        f"- Fixed tau (HVF baseline): `{config.fixed_tau:.4f}`",
        f"- Mixed calibration quantile: `{config.mixed_calibration_quantile:.2f}`",
        "",
        "## Family Summary",
        "",
        "| Family | Baseline F1 | Auto F1 | Delta F1 | Baseline IoU | Auto IoU | Delta IoU | Delta Model Error | Delta Runtime (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for record in delta_records:
        lines.append(
            "| {family} | {baseline_f1:.3f} | {auto_f1:.3f} | {delta_f1:+.3f} | {baseline_iou:.3f} | "
            "{auto_iou:.3f} | {delta_iou:+.3f} | {delta_model_error:+.3f} | {delta_runtime:+.3f} |".format(
                family=record["model_family"],
                baseline_f1=float(record["baseline_inlier_f1_mean"]),
                auto_f1=float(record["auto_inlier_f1_mean"]),
                delta_f1=float(record["delta_inlier_f1"]),
                baseline_iou=float(record["baseline_match_iou_mean"]),
                auto_iou=float(record["auto_match_iou_mean"]),
                delta_iou=float(record["delta_match_iou_mean"]),
                delta_model_error=float(record["delta_model_count_error"]),
                delta_runtime=float(record["delta_runtime_seconds"]),
            )
        )

    lines.extend(
        [
            "",
            "## Summary Records",
            "",
            f"- Number of summary rows: `{len(summary_records)}`",
            f"- Number of delta rows: `{len(delta_records)}`",
            "",
            "## Plot Files",
            "",
        ]
    )

    for plot_path in plots:
        lines.append(f"- `{plot_path}`")

    lines.append("")
    return "\n".join(lines)


def run_adelaide_baseline_comparison(
    config: AdelaideBaselineComparisonConfig,
) -> AdelaideBaselineComparisonBundle:
    records: list[dict[str, Any]] = []
    files = list_adelaide_rmf_files(config.dataset_root)
    if config.limit_files is not None:
        files = files[: max(int(config.limit_files), 0)]

    for file_path in files:
        sample = load_adelaide_rmf_sample(file_path)
        correspondences = correspondence_set_from_adelaide(sample)

        for model_family in config.model_families:
            calibration_enabled = model_family == "mixed"
            for seed in config.seeds:
                for method_variant, use_aikose in METHOD_VARIANTS:
                    rng = np.random.default_rng(seed)
                    result = run_adelaide_hvf(
                        correspondences,
                        AdelaideHVFConfig(
                            model_family=model_family,
                            num_hypotheses=config.num_hypotheses,
                            use_aikose=use_aikose,
                            fixed_tau=config.fixed_tau,
                            mixed_residual_calibration=MixedResidualCalibrationConfig(
                                enabled=calibration_enabled,
                                quantile=float(config.mixed_calibration_quantile),
                            ),
                        ),
                        rng,
                    )
                    metrics = build_adelaide_metric_result(result.extraction.model_labels, sample.labels)

                    records.append(
                        {
                            "method_variant": method_variant,
                            "use_aikose": bool(use_aikose),
                            "fixed_tau": float(config.fixed_tau),
                            "sample_name": Path(sample.source_path).name,
                            "sample_path": sample.source_path,
                            "model_family": model_family,
                            "seed": seed,
                            "num_correspondences": sample.num_correspondences,
                            "mixed_residual_calibration_enabled": int(calibration_enabled),
                            "mixed_residual_calibration_quantile": float(config.mixed_calibration_quantile),
                            "tau_mean": float(result.tau_diagnostics.tau_mean),
                            "tau_std": float(result.tau_diagnostics.tau_std),
                            "runtime_seconds": float(result.runtime_seconds),
                            **metrics.to_dict(),
                        }
                    )

    saved_files = save_experiment_results(records, config.output_dir, config.experiment_name)

    summary_records = _build_summary_records(records)
    summary_csv_path = Path(config.output_dir) / f"{config.experiment_name}_summary.csv"
    save_csv_records(summary_records, summary_csv_path)
    saved_files["summary_csv"] = str(summary_csv_path)
    summary_json_path = Path(config.output_dir) / f"{config.experiment_name}_summary.json"
    save_json_records(summary_records, summary_json_path)
    saved_files["summary_json"] = str(summary_json_path)

    delta_records = _build_delta_records(summary_records)
    delta_csv_path = Path(config.output_dir) / f"{config.experiment_name}_delta_summary.csv"
    save_csv_records(delta_records, delta_csv_path)
    saved_files["delta_summary_csv"] = str(delta_csv_path)
    delta_json_path = Path(config.output_dir) / f"{config.experiment_name}_delta_summary.json"
    save_json_records(delta_records, delta_json_path)
    saved_files["delta_summary_json"] = str(delta_json_path)

    metadata = {
        "dataset_root": config.dataset_root,
        "experiment_name": config.experiment_name,
        "model_families": list(config.model_families),
        "num_hypotheses": int(config.num_hypotheses),
        "seeds": list(config.seeds),
        "fixed_tau": float(config.fixed_tau),
        "limit_files": config.limit_files,
        "mixed_calibration_quantile": float(config.mixed_calibration_quantile),
        "method_variants": [method for method, _ in METHOD_VARIANTS],
    }
    metadata_path = Path(config.output_dir) / f"{config.experiment_name}_metadata.json"
    save_json_records([metadata], metadata_path)
    saved_files["metadata_json"] = str(metadata_path)

    plots: list[str] = []
    plots.extend(
        _save_plot_bundle(
            _plot_method_comparison,
            Path(config.output_dir) / f"{config.experiment_name}_comparison_overview",
            summary_records,
        )
    )
    plots.extend(
        _save_plot_bundle(
            _plot_delta_summary,
            Path(config.output_dir) / f"{config.experiment_name}_delta_overview",
            delta_records,
        )
    )

    report_path = Path(config.output_dir) / f"{config.experiment_name}_report.md"
    report_path.write_text(
        _build_markdown_report(config, summary_records, delta_records, plots),
        encoding="utf-8",
    )
    saved_files["report_md"] = str(report_path)

    return AdelaideBaselineComparisonBundle(
        records=records,
        summary_records=summary_records,
        delta_records=delta_records,
        saved_files=saved_files,
        plots=plots,
    )


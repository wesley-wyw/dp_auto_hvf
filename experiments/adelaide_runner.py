from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from data.adelaide import MixedResidualCalibrationConfig, correspondence_set_from_adelaide
from data.loaders import list_adelaide_rmf_files, load_adelaide_rmf_sample
from hvf.adelaide_pipeline import AdelaideHVFConfig, run_adelaide_hvf
from privacy.dp_pipeline import DPHVFConfig, apply_dp_adelaide_hvf
from visualization.plots import (
    plot_adelaide_family_type_balance,
    plot_adelaide_image_overlay,
    plot_adelaide_metric_by_family,
    plot_adelaide_metric_by_sample,
    plot_adelaide_model_count_comparison,
    plot_adelaide_selected_model_types,
    plot_adelaide_thesis_method_overview,
    plot_dp_tradeoff_overview,
    plot_metric_curve,
    plot_mixed_calibration_overview,
)

from .reporting import save_csv_records, save_experiment_results


@dataclass(frozen=True)
class AdelaideExperimentConfig:
    """Batch experiment config for AdelaideRMF-style real-data runs."""

    dataset_root: str
    output_dir: str = "outputs"
    experiment_name: str = "adelaide_experiment"
    model_families: tuple[str, ...] = ("fundamental", "homography")
    num_hypotheses: int = 240
    use_aikose: bool = True
    fixed_tau: float = 0.05
    seeds: tuple[int, ...] = (0,)
    limit_files: int | None = None
    enable_dp: bool = False
    dp_epsilon: float = 1.0
    dp_epsilons: tuple[float, ...] | None = None
    dp_delta: float = 1e-6
    dp_mechanism: str = "laplace"
    dp_injection_points: tuple[str, ...] = ("dp_on_hypothesis_scores", "dp_on_model_selection")
    dp_model_selection_top_k: int = 3
    dp_max_contribution: float = 1.0
    dp_epsilon_allocation: str = "equal"
    dp_epsilon_allocations: dict[str, float] | None = None
    dp_delta_allocation: str = "equal"
    dp_delta_allocations: dict[str, float] | None = None
    mixed_calibration_modes: tuple[bool, ...] | None = None
    mixed_calibration_quantile: float = 0.5


@dataclass(frozen=True)
class AdelaideExperimentBundle:
    """Persisted Adelaide experiment artifacts and in-memory records."""

    records: list[dict[str, Any]]
    summary_records: list[dict[str, Any]]
    saved_files: dict[str, str]
    plots: list[str]


@dataclass(frozen=True)
class AdelaideMetricResult:
    true_model_count: int
    predicted_model_count: int
    model_count_error: int
    true_outlier_ratio: float
    predicted_outlier_ratio: float
    misclassification_error: float
    inlier_precision: float
    inlier_recall: float
    inlier_f1: float
    assigned_ratio: float
    ari_all: float
    nmi_all: float
    ari_inlier: float
    nmi_inlier: float
    label_purity: float
    gt_best_iou_mean: float
    pred_best_iou_mean: float
    match_iou_mean: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)



def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0



def _compute_misclassification_error(predicted_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Misclassification error rate following Xiao et al. (2018).

    Counts data points whose predicted group assignment differs from ground
    truth, divided by total number of data points.  For multi-model data the
    predicted and true label IDs may not match, so we use the Hungarian
    algorithm to find the best label mapping before counting errors.
    """
    true_arr = np.asarray(true_labels, dtype=int)
    pred_arr = np.asarray(predicted_labels, dtype=int)
    n = true_arr.size
    if n == 0:
        return 0.0

    # Separate outliers: true label 0 = outlier, pred label -1 = outlier
    true_inlier_mask = true_arr > 0
    pred_inlier_mask = pred_arr >= 0

    # Count outlier misclassifications
    # True outlier predicted as inlier, or true inlier predicted as outlier
    outlier_errors = int(np.logical_xor(true_inlier_mask, pred_inlier_mask).sum())

    # For inlier-inlier assignments, find best label mapping via greedy match
    both_inlier = true_inlier_mask & pred_inlier_mask
    if not np.any(both_inlier):
        return float(outlier_errors) / float(n)

    true_sub = true_arr[both_inlier]
    pred_sub = pred_arr[both_inlier]
    true_ids = np.unique(true_sub)
    pred_ids = np.unique(pred_sub)

    # Build overlap matrix and greedily match
    best_mapping: dict[int, int] = {}
    used_true: set[int] = set()
    overlap_pairs: list[tuple[int, int, int]] = []
    for pid in pred_ids:
        p_mask = pred_sub == pid
        for tid in true_ids:
            overlap = int(np.logical_and(p_mask, true_sub == tid).sum())
            overlap_pairs.append((overlap, int(pid), int(tid)))
    overlap_pairs.sort(reverse=True)
    for overlap, pid, tid in overlap_pairs:
        if pid in best_mapping or tid in used_true:
            continue
        best_mapping[pid] = tid
        used_true.add(tid)

    mapped_pred = np.array([best_mapping.get(int(p), -999) for p in pred_sub])
    inlier_errors = int(np.sum(mapped_pred != true_sub))

    return float(outlier_errors + inlier_errors) / float(n)


def _compute_inlier_metrics(predicted_labels: np.ndarray, true_labels: np.ndarray) -> tuple[float, float, float]:
    pred_inlier = np.asarray(predicted_labels) != -1
    true_inlier = np.asarray(true_labels) > 0

    tp = float(np.logical_and(pred_inlier, true_inlier).sum())
    fp = float(np.logical_and(pred_inlier, ~true_inlier).sum())
    fn = float(np.logical_and(~pred_inlier, true_inlier).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1




def _compute_partition_metrics(predicted_labels: np.ndarray, true_labels: np.ndarray) -> tuple[float, float, float, float]:
    true_labels_array = np.asarray(true_labels, dtype=int)
    predicted_labels_array = np.asarray(predicted_labels, dtype=int)

    true_all = np.where(true_labels_array > 0, true_labels_array, 0)
    pred_all = np.where(predicted_labels_array >= 0, predicted_labels_array + 1, 0)
    ari_all = float(adjusted_rand_score(true_all, pred_all))
    nmi_all = float(normalized_mutual_info_score(true_all, pred_all, average_method="arithmetic"))

    inlier_mask = true_labels_array > 0
    if not np.any(inlier_mask):
        return ari_all, nmi_all, 0.0, 0.0

    true_inlier = true_labels_array[inlier_mask]
    pred_inlier = np.where(predicted_labels_array[inlier_mask] >= 0, predicted_labels_array[inlier_mask] + 1, 0)
    ari_inlier = float(adjusted_rand_score(true_inlier, pred_inlier))
    nmi_inlier = float(normalized_mutual_info_score(true_inlier, pred_inlier, average_method="arithmetic"))
    return ari_all, nmi_all, ari_inlier, nmi_inlier


def _compute_label_purity(predicted_labels: np.ndarray, true_labels: np.ndarray) -> float:
    true_labels_array = np.asarray(true_labels, dtype=int)
    predicted_labels_array = np.asarray(predicted_labels, dtype=int)
    inlier_mask = true_labels_array > 0
    if not np.any(inlier_mask):
        return 0.0

    true_inlier = true_labels_array[inlier_mask]
    pred_inlier = predicted_labels_array[inlier_mask]
    purity_numerator = 0

    for pred_cluster in np.unique(pred_inlier):
        cluster_mask = pred_inlier == pred_cluster
        if not np.any(cluster_mask):
            continue
        cluster_true = true_inlier[cluster_mask]
        _, counts = np.unique(cluster_true, return_counts=True)
        purity_numerator += int(np.max(counts)) if counts.size > 0 else 0

    return _safe_div(float(purity_numerator), float(true_inlier.size))


def _compute_model_matching_iou(predicted_labels: np.ndarray, true_labels: np.ndarray) -> tuple[float, float, float]:
    true_labels_array = np.asarray(true_labels, dtype=int)
    predicted_labels_array = np.asarray(predicted_labels, dtype=int)

    true_models = np.unique(true_labels_array[true_labels_array > 0])
    pred_models = np.unique(predicted_labels_array[predicted_labels_array >= 0])

    if true_models.size == 0 or pred_models.size == 0:
        return 0.0, 0.0, 0.0

    iou_matrix = np.zeros((true_models.size, pred_models.size), dtype=float)
    for i, true_id in enumerate(true_models):
        true_mask = true_labels_array == int(true_id)
        for j, pred_id in enumerate(pred_models):
            pred_mask = predicted_labels_array == int(pred_id)
            intersection = float(np.logical_and(true_mask, pred_mask).sum())
            union = float(np.logical_or(true_mask, pred_mask).sum())
            iou_matrix[i, j] = _safe_div(intersection, union)

    gt_best = float(np.mean(np.max(iou_matrix, axis=1))) if iou_matrix.shape[0] > 0 else 0.0
    pred_best = float(np.mean(np.max(iou_matrix, axis=0))) if iou_matrix.shape[1] > 0 else 0.0
    match_mean = 0.5 * (gt_best + pred_best)
    return gt_best, pred_best, match_mean

def build_adelaide_metric_result(predicted_labels: np.ndarray, true_labels: np.ndarray) -> AdelaideMetricResult:
    true_labels_array = np.asarray(true_labels, dtype=int)
    predicted_labels_array = np.asarray(predicted_labels, dtype=int)

    misclassification_error = _compute_misclassification_error(predicted_labels_array, true_labels_array)
    precision, recall, f1 = _compute_inlier_metrics(predicted_labels_array, true_labels_array)
    ari_all, nmi_all, ari_inlier, nmi_inlier = _compute_partition_metrics(predicted_labels_array, true_labels_array)
    label_purity = _compute_label_purity(predicted_labels_array, true_labels_array)
    gt_best_iou_mean, pred_best_iou_mean, match_iou_mean = _compute_model_matching_iou(
        predicted_labels_array,
        true_labels_array,
    )
    true_model_count = int(np.unique(true_labels_array[true_labels_array > 0]).size)
    predicted_model_count = int(np.unique(predicted_labels_array[predicted_labels_array >= 0]).size)

    return AdelaideMetricResult(
        true_model_count=true_model_count,
        predicted_model_count=predicted_model_count,
        model_count_error=abs(predicted_model_count - true_model_count),
        true_outlier_ratio=float(np.mean(true_labels_array == 0)),
        predicted_outlier_ratio=float(np.mean(predicted_labels_array == -1)),
        misclassification_error=misclassification_error,
        inlier_precision=precision,
        inlier_recall=recall,
        inlier_f1=f1,
        assigned_ratio=float(np.mean(predicted_labels_array >= 0)),
        ari_all=ari_all,
        nmi_all=nmi_all,
        ari_inlier=ari_inlier,
        nmi_inlier=nmi_inlier,
        label_purity=label_purity,
        gt_best_iou_mean=gt_best_iou_mean,
        pred_best_iou_mean=pred_best_iou_mean,
        match_iou_mean=match_iou_mean,
    )


def _summarize_selected_model_types(selected_model_types: np.ndarray) -> dict[str, int]:
    types = np.asarray(selected_model_types, dtype=object)
    if types.size == 0:
        return {"selected_fundamental_count": 0, "selected_homography_count": 0}

    return {
        "selected_fundamental_count": int(np.count_nonzero(types == "fundamental")),
        "selected_homography_count": int(np.count_nonzero(types == "homography")),
    }



def _build_dp_config(config: AdelaideExperimentConfig, epsilon: float) -> DPHVFConfig:
    return DPHVFConfig(
        epsilon=float(epsilon),
        delta=config.dp_delta,
        mechanism=config.dp_mechanism,
        max_contribution=config.dp_max_contribution,
        injection_points=config.dp_injection_points,
        model_selection_top_k=config.dp_model_selection_top_k,
        epsilon_allocation=config.dp_epsilon_allocation,
        epsilon_allocations=config.dp_epsilon_allocations,
        delta_allocation=config.dp_delta_allocation,
        delta_allocations=config.dp_delta_allocations,
    )



def _summarize_dp_result(
    enable_dp: bool,
    dp_epsilon: float,
    dp_delta: float,
    dp_epsilon_allocation: str,
    dp_result: Any,
) -> dict[str, Any]:
    if not enable_dp or dp_result is None:
        return {
            "dp_enabled": False,
            "dp_epsilon": 0.0,
            "dp_delta": 0.0,
            "dp_num_queries": 0,
            "dp_avg_noise_scale": 0.0,
            "dp_epsilon_allocation": "none",
            "dp_budget_breakdown": "",
        }

    reports = dp_result.dp_result.privacy_reports
    avg_noise_scale = float(np.mean([float(report.noise_scale) for report in reports])) if reports else 0.0
    budget_breakdown = ";".join(
        f"{report.injection_point}:{float(report.epsilon):.6g}" for report in reports
    )
    summary = {
        "dp_enabled": True,
        "dp_epsilon": float(dp_epsilon),
        "dp_delta": float(dp_delta),
        "dp_num_queries": int(sum(int(report.num_queries) for report in reports)),
        "dp_avg_noise_scale": avg_noise_scale,
        "dp_selected_models": int(dp_result.dp_result.selected_model_indices.size),
        "dp_epsilon_allocation": str(dp_epsilon_allocation),
        "dp_budget_breakdown": budget_breakdown,
    }
    summary.update(_summarize_selected_model_types(dp_result.selected_model_types))
    return summary



def _format_numeric_tag(value: float) -> str:
    return f"{float(value):.3f}".replace("-", "m").replace(".", "p")


def _generate_adelaide_overlay_plot(
    correspondences: Any,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    output_dir: str,
    experiment_name: str,
    sample_name: str,
    model_family: str,
    seed: int,
    dp_epsilon: float | None,
    calibration_enabled: bool,
    metrics: AdelaideMetricResult,
) -> str | None:
    if correspondences.image_1 is None and correspondences.image_2 is None:
        return None

    sample_stem = Path(sample_name).stem
    suffix_parts = [sample_stem, model_family, f"seed{int(seed)}"]
    if dp_epsilon is not None:
        suffix_parts.append(f"eps{_format_numeric_tag(float(dp_epsilon))}")
    if model_family.lower() == "mixed":
        suffix_parts.append(f"calib{int(bool(calibration_enabled))}")

    output_path = Path(output_dir) / f"{experiment_name}_{'_'.join(suffix_parts)}_overlay.png"
    plot_adelaide_image_overlay(
        correspondences.image_1,
        correspondences.image_2,
        correspondences.image_points_1,
        correspondences.image_points_2,
        predicted_labels,
        true_labels,
        output_path,
        sample_name=sample_name,
        model_family=model_family,
        metric_summary={
            "inlier_f1": float(metrics.inlier_f1),
            "match_iou_mean": float(metrics.match_iou_mean),
            "predicted_model_count": int(metrics.predicted_model_count),
            "true_model_count": int(metrics.true_model_count),
        },
    )
    return str(output_path) if output_path.exists() else None


def _build_adelaide_summary_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []

    numeric_keys = (
        "misclassification_error",
        "inlier_f1",
        "inlier_precision",
        "inlier_recall",
        "runtime_seconds",
        "model_count_error",
        "predicted_model_count",
        "true_model_count",
        "predicted_outlier_ratio",
        "true_outlier_ratio",
        "tau_mean",
        "tau_std",
        "selected_fundamental_count",
        "selected_homography_count",
        "dp_num_queries",
        "dp_avg_noise_scale",
        "dp_selected_models",
        "ari_all",
        "nmi_all",
        "ari_inlier",
        "nmi_inlier",
        "label_purity",
        "gt_best_iou_mean",
        "pred_best_iou_mean",
        "match_iou_mean",
        "mixed_residual_calibration_enabled",
        "mixed_residual_calibration_quantile",
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        family = str(record.get("model_family", "unknown"))
        grouped.setdefault(family, []).append(record)

    summary_records: list[dict[str, Any]] = []
    for family, group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "model_family": family,
            "num_runs": len(group),
            "num_samples": len({str(item.get('sample_name', 'unknown')) for item in group}),
            "dp_enabled": any(bool(item.get("dp_enabled", False)) for item in group),
        }
        for key in numeric_keys:
            values = [float(item[key]) for item in group if item.get(key) is not None]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
        summary_records.append(summary)

    return summary_records



def _generate_adelaide_plots(
    records: list[dict[str, Any]],
    summary_records: list[dict[str, Any]],
    output_dir: str,
    experiment_name: str,
    model_families: tuple[str, ...],
) -> list[str]:
    if not records:
        return []

    output_root = Path(output_dir)
    plots: list[str] = []

    family_list = list(dict.fromkeys(model_families))
    if len(family_list) > 1:
        family_list.append("all")

    for family in family_list:
        family_filter = None if family == "all" else family
        suffix = family

        f1_path = output_root / f"{experiment_name}_{suffix}_inlier_f1.png"
        plot_adelaide_metric_by_sample(records, "inlier_f1", f1_path, model_family=family_filter)
        if f1_path.exists():
            plots.append(str(f1_path))

        ari_path = output_root / f"{experiment_name}_{suffix}_ari_inlier.png"
        plot_adelaide_metric_by_sample(records, "ari_inlier", ari_path, model_family=family_filter)
        if ari_path.exists():
            plots.append(str(ari_path))

        match_iou_path = output_root / f"{experiment_name}_{suffix}_match_iou_mean.png"
        plot_adelaide_metric_by_sample(records, "match_iou_mean", match_iou_path, model_family=family_filter)
        if match_iou_path.exists():
            plots.append(str(match_iou_path))

        runtime_path = output_root / f"{experiment_name}_{suffix}_runtime.png"
        plot_adelaide_metric_by_sample(records, "runtime_seconds", runtime_path, model_family=family_filter)
        if runtime_path.exists():
            plots.append(str(runtime_path))

        outlier_path = output_root / f"{experiment_name}_{suffix}_predicted_outlier_ratio.png"
        plot_adelaide_metric_by_sample(records, "predicted_outlier_ratio", outlier_path, model_family=family_filter)
        if outlier_path.exists():
            plots.append(str(outlier_path))

        tau_path = output_root / f"{experiment_name}_{suffix}_tau_mean.png"
        plot_adelaide_metric_by_sample(records, "tau_mean", tau_path, model_family=family_filter)
        if tau_path.exists():
            plots.append(str(tau_path))

        count_path = output_root / f"{experiment_name}_{suffix}_model_count.png"
        plot_adelaide_model_count_comparison(records, count_path, model_family=family_filter)
        if count_path.exists():
            plots.append(str(count_path))

        types_path = output_root / f"{experiment_name}_{suffix}_selected_types.png"
        plot_adelaide_selected_model_types(records, types_path, model_family=family_filter)
        if types_path.exists():
            plots.append(str(types_path))

    summary_families = {str(item.get("model_family", "")) for item in summary_records if item.get("model_family") is not None}
    if summary_records and len(summary_families) > 1:
        family_f1_path = output_root / f"{experiment_name}_family_inlier_f1.png"
        plot_adelaide_metric_by_family(summary_records, "inlier_f1_mean", family_f1_path)
        if family_f1_path.exists():
            plots.append(str(family_f1_path))

        family_ari_path = output_root / f"{experiment_name}_family_ari_inlier.png"
        plot_adelaide_metric_by_family(summary_records, "ari_inlier_mean", family_ari_path)
        if family_ari_path.exists():
            plots.append(str(family_ari_path))

        family_match_path = output_root / f"{experiment_name}_family_match_iou_mean.png"
        plot_adelaide_metric_by_family(summary_records, "match_iou_mean_mean", family_match_path)
        if family_match_path.exists():
            plots.append(str(family_match_path))

        family_runtime_path = output_root / f"{experiment_name}_family_runtime.png"
        plot_adelaide_metric_by_family(summary_records, "runtime_seconds_mean", family_runtime_path)
        if family_runtime_path.exists():
            plots.append(str(family_runtime_path))

        family_error_path = output_root / f"{experiment_name}_family_model_error.png"
        plot_adelaide_metric_by_family(summary_records, "model_count_error_mean", family_error_path)
        if family_error_path.exists():
            plots.append(str(family_error_path))

        family_types_path = output_root / f"{experiment_name}_family_selected_types.png"
        plot_adelaide_family_type_balance(summary_records, family_types_path)
        if family_types_path.exists():
            plots.append(str(family_types_path))

        if any(bool(record.get("dp_enabled", False)) for record in records):
            family_noise_path = output_root / f"{experiment_name}_family_dp_noise.png"
            plot_adelaide_metric_by_family(summary_records, "dp_avg_noise_scale_mean", family_noise_path)
            if family_noise_path.exists():
                plots.append(str(family_noise_path))

    dp_epsilons = sorted({float(item.get("dp_epsilon", 0.0)) for item in records if bool(item.get("dp_enabled", False))})
    thesis_overview_path = output_root / f"{experiment_name}_thesis_method_overview.png"
    plot_adelaide_thesis_method_overview(summary_records, thesis_overview_path)
    if thesis_overview_path.exists():
        plots.append(str(thesis_overview_path))

    if len(dp_epsilons) > 1:
        tradeoff_specs = (
            ("inlier_f1", "tradeoff_inlier_f1_vs_epsilon"),
            ("ari_inlier", "tradeoff_ari_inlier_vs_epsilon"),
            ("match_iou_mean", "tradeoff_match_iou_vs_epsilon"),
            ("dp_avg_noise_scale", "tradeoff_noise_scale_vs_epsilon"),
        )
        for metric_key, suffix in tradeoff_specs:
            tradeoff_path = output_root / f"{experiment_name}_{suffix}.png"
            plot_metric_curve(records, "dp_epsilon", metric_key, tradeoff_path)
            if tradeoff_path.exists():
                plots.append(str(tradeoff_path))

        tradeoff_overview_path = output_root / f"{experiment_name}_thesis_dp_tradeoff_overview.png"
        plot_dp_tradeoff_overview(records, tradeoff_overview_path)
        if tradeoff_overview_path.exists():
            plots.append(str(tradeoff_overview_path))

    mixed_calibration_values = sorted({int(item.get("mixed_residual_calibration_enabled", 0)) for item in records if str(item.get("model_family", "")).lower() == "mixed"})
    if len(mixed_calibration_values) > 1:
        mixed_records = [item for item in records if str(item.get("model_family", "")).lower() == "mixed"]
        calibration_ab_specs = (
            ("inlier_f1", "calibration_ab_inlier_f1"),
            ("ari_inlier", "calibration_ab_ari_inlier"),
            ("match_iou_mean", "calibration_ab_match_iou"),
        )
        for metric_key, suffix in calibration_ab_specs:
            calibration_path = output_root / f"{experiment_name}_{suffix}.png"
            plot_metric_curve(mixed_records, "mixed_residual_calibration_enabled", metric_key, calibration_path)
            if calibration_path.exists():
                plots.append(str(calibration_path))

        calibration_overview_path = output_root / f"{experiment_name}_thesis_calibration_overview.png"
        plot_mixed_calibration_overview(mixed_records, calibration_overview_path)
        if calibration_overview_path.exists():
            plots.append(str(calibration_overview_path))

    return plots



def run_adelaide_experiments(config: AdelaideExperimentConfig) -> AdelaideExperimentBundle:
    records: list[dict[str, Any]] = []
    overlay_plots: list[str] = []
    files = list_adelaide_rmf_files(config.dataset_root)
    if config.limit_files is not None:
        files = files[: max(int(config.limit_files), 0)]

    epsilon_values: tuple[float, ...]
    if config.enable_dp:
        epsilon_values = config.dp_epsilons if config.dp_epsilons is not None else (config.dp_epsilon,)
    else:
        epsilon_values = (0.0,)

    for file_path in files:
        sample = load_adelaide_rmf_sample(file_path)
        correspondences = correspondence_set_from_adelaide(sample)

        for model_family in config.model_families:
            if model_family == "mixed":
                calibration_values = (
                    config.mixed_calibration_modes
                    if config.mixed_calibration_modes is not None
                    else (True,)
                )
            else:
                calibration_values = (False,)

            for seed in config.seeds:
                for current_epsilon in epsilon_values:
                    for calibration_enabled in calibration_values:
                        rng = np.random.default_rng(seed)
                        hvf_config = AdelaideHVFConfig(
                            model_family=model_family,
                            num_hypotheses=config.num_hypotheses,
                            use_aikose=config.use_aikose,
                            fixed_tau=config.fixed_tau,
                            mixed_residual_calibration=MixedResidualCalibrationConfig(
                                enabled=bool(calibration_enabled),
                                quantile=float(config.mixed_calibration_quantile),
                            ),
                        )
                        result = run_adelaide_hvf(correspondences, hvf_config, rng)
                        dp_config = _build_dp_config(config, epsilon=current_epsilon) if config.enable_dp else None
                        dp_result = apply_dp_adelaide_hvf(result, dp_config, rng) if dp_config is not None else None

                        predicted_labels = dp_result.model_labels if dp_result is not None else result.extraction.model_labels
                        metrics = build_adelaide_metric_result(predicted_labels, sample.labels)
                        selected_types_summary = (
                            _summarize_selected_model_types(dp_result.selected_model_types)
                            if dp_result is not None
                            else _summarize_selected_model_types(result.selected_model_types)
                        )

                        overlay_path = _generate_adelaide_overlay_plot(
                            correspondences=correspondences,
                            predicted_labels=predicted_labels,
                            true_labels=sample.labels,
                            output_dir=config.output_dir,
                            experiment_name=config.experiment_name,
                            sample_name=Path(sample.source_path).name,
                            model_family=model_family,
                            seed=seed,
                            dp_epsilon=float(current_epsilon) if dp_result is not None else None,
                            calibration_enabled=bool(calibration_enabled),
                            metrics=metrics,
                        )
                        if overlay_path is not None:
                            overlay_plots.append(overlay_path)

                        records.append(
                            {
                                "sample_name": Path(sample.source_path).name,
                                "sample_path": sample.source_path,
                                "model_family": model_family,
                                "seed": seed,
                                "num_correspondences": sample.num_correspondences,
                                "tau_mean": float(result.tau_diagnostics.tau_mean),
                                "tau_std": float(result.tau_diagnostics.tau_std),
                                "preference_density": float(result.preference_stats.density),
                                "runtime_seconds": float(result.runtime_seconds),
                                "metric_source": "post_dp" if dp_result is not None else "non_dp_pipeline",
                                "mixed_residual_calibration_enabled": int(bool(calibration_enabled)),
                                "mixed_residual_calibration_quantile": float(config.mixed_calibration_quantile),
                                "overlay_plot": overlay_path or "",
                                **metrics.to_dict(),
                                **selected_types_summary,
                                **_summarize_dp_result(
                                    enable_dp=config.enable_dp,
                                    dp_epsilon=float(current_epsilon),
                                    dp_delta=float(config.dp_delta),
                                    dp_epsilon_allocation=str(config.dp_epsilon_allocation),
                                    dp_result=dp_result,
                                ),
                            }
                        )

    saved_files = save_experiment_results(records, config.output_dir, config.experiment_name)
    summary_records = _build_adelaide_summary_records(records)
    summary_csv_path = Path(config.output_dir) / f"{config.experiment_name}_summary.csv"
    save_csv_records(summary_records, summary_csv_path)
    saved_files["summary_csv"] = str(summary_csv_path)
    plots = overlay_plots + _generate_adelaide_plots(records, summary_records, config.output_dir, config.experiment_name, config.model_families)
    return AdelaideExperimentBundle(records=records, summary_records=summary_records, saved_files=saved_files, plots=plots)
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    """Per-run experiment metrics."""

    model_detection_accuracy: float
    parameter_error: float
    inlier_precision: float
    inlier_recall: float
    inlier_f1: float
    outlier_rejection_rate: float
    runtime_seconds: float
    tau_mean: float
    tau_std: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _line_distance(line_a: np.ndarray, line_b: np.ndarray) -> float:
    # Sign-invariant distance for normalized line parameters.
    return float(min(np.linalg.norm(line_a - line_b), np.linalg.norm(line_a + line_b)))


def compute_model_detection_metrics(
    predicted_models: np.ndarray,
    true_models: np.ndarray,
    match_threshold: float = 0.20,
) -> tuple[float, float]:
    """Compute detection accuracy and mean parameter error via greedy matching."""
    pred = np.asarray(predicted_models, dtype=float)
    truth = np.asarray(true_models, dtype=float)

    if truth.size == 0:
        return 0.0, 0.0
    if pred.size == 0:
        return 0.0, float("inf")

    unmatched_pred = set(range(pred.shape[0]))
    matched_distances: list[float] = []

    for true_index in range(truth.shape[0]):
        if not unmatched_pred:
            break

        candidates = list(unmatched_pred)
        distances = np.array([_line_distance(truth[true_index], pred[index]) for index in candidates])
        best_local = int(np.argmin(distances))
        best_pred = candidates[best_local]
        best_distance = float(distances[best_local])

        if best_distance <= match_threshold:
            matched_distances.append(best_distance)
            unmatched_pred.remove(best_pred)

    accuracy = _safe_div(len(matched_distances), truth.shape[0])
    parameter_error = float(np.mean(matched_distances)) if matched_distances else float("inf")
    return accuracy, parameter_error


def compute_inlier_metrics(predicted_labels: np.ndarray, true_labels: np.ndarray) -> tuple[float, float, float]:
    """Compute precision/recall/F1 for inlier-vs-outlier classification."""
    pred_inlier = np.asarray(predicted_labels) != -1
    true_inlier = np.asarray(true_labels) != -1

    tp = float(np.logical_and(pred_inlier, true_inlier).sum())
    fp = float(np.logical_and(pred_inlier, ~true_inlier).sum())
    fn = float(np.logical_and(~pred_inlier, true_inlier).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def compute_outlier_rejection_rate(outlier_mask: np.ndarray, true_labels: np.ndarray) -> float:
    """Compute true outlier rejection rate."""
    true_outliers = np.asarray(true_labels) == -1
    predicted_outliers = np.asarray(outlier_mask, dtype=bool)

    true_positive = float(np.logical_and(predicted_outliers, true_outliers).sum())
    total_outliers = float(true_outliers.sum())
    return _safe_div(true_positive, total_outliers)


def build_metric_result(
    *,
    predicted_models: np.ndarray,
    predicted_labels: np.ndarray,
    outlier_mask: np.ndarray,
    true_models: np.ndarray,
    true_labels: np.ndarray,
    runtime_seconds: float,
    tau_mean: float,
    tau_std: float,
) -> MetricResult:
    """Compute the complete metric bundle for one run."""
    detection_accuracy, parameter_error = compute_model_detection_metrics(predicted_models, true_models)
    precision, recall, f1 = compute_inlier_metrics(predicted_labels, true_labels)
    rejection_rate = compute_outlier_rejection_rate(outlier_mask, true_labels)

    return MetricResult(
        model_detection_accuracy=detection_accuracy,
        parameter_error=parameter_error,
        inlier_precision=precision,
        inlier_recall=recall,
        inlier_f1=f1,
        outlier_rejection_rate=rejection_rate,
        runtime_seconds=float(runtime_seconds),
        tau_mean=float(tau_mean),
        tau_std=float(tau_std),
    )

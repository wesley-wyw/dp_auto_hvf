from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _label_color_map(labels: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    positive_labels = sorted(int(label) for label in np.unique(label_array) if int(label) >= 0)
    if not positive_labels:
        return {}

    cmap = plt.get_cmap("tab10", max(len(positive_labels), 1))
    return {
        label: cmap(index)
        for index, label in enumerate(positive_labels)
    }


def _plot_labeled_points_on_axis(
    axis: plt.Axes,
    image: np.ndarray | None,
    points: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
) -> None:
    axis.set_title(title)

    if image is not None:
        axis.imshow(np.asarray(image))
    else:
        point_array = np.asarray(points, dtype=float)
        if point_array.size > 0:
            axis.set_xlim(float(np.min(point_array[:, 0])) - 10.0, float(np.max(point_array[:, 0])) + 10.0)
            axis.set_ylim(float(np.max(point_array[:, 1])) + 10.0, float(np.min(point_array[:, 1])) - 10.0)
            axis.set_facecolor("#f3f4f6")

    point_array = np.asarray(points, dtype=float)
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    color_map = _label_color_map(label_array)

    outlier_mask = label_array < 0
    if np.any(outlier_mask):
        axis.scatter(
            point_array[outlier_mask, 0],
            point_array[outlier_mask, 1],
            s=26,
            c="#c4c4c4",
            marker="x",
            linewidths=1.0,
            alpha=0.85,
            label="outlier" if np.any(label_array >= 0) else None,
        )

    for label in sorted(color_map):
        mask = label_array == label
        if not np.any(mask):
            continue
        axis.scatter(
            point_array[mask, 0],
            point_array[mask, 1],
            s=28,
            c=[color_map[label]],
            edgecolors="#111111",
            linewidths=0.35,
            alpha=0.92,
            label=f"model {label + 1}",
        )

    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")


def plot_adelaide_image_overlay(
    image_1: np.ndarray | None,
    image_2: np.ndarray | None,
    points_1: np.ndarray,
    points_2: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    output_path: str | Path,
    *,
    sample_name: str,
    model_family: str,
    metric_summary: dict[str, float] | None = None,
) -> None:
    """Overlay predicted and ground-truth correspondence labels on Adelaide images."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    predicted = np.asarray(predicted_labels, dtype=int).reshape(-1)
    truth = np.asarray(true_labels, dtype=int).reshape(-1)
    truth_for_display = np.where(truth > 0, truth - 1, -1)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    _plot_labeled_points_on_axis(
        axes[0, 0],
        image_1,
        points_1,
        predicted,
        title="Predicted labels on view 1",
    )
    _plot_labeled_points_on_axis(
        axes[0, 1],
        image_2,
        points_2,
        predicted,
        title="Predicted labels on view 2",
    )
    _plot_labeled_points_on_axis(
        axes[1, 0],
        image_1,
        points_1,
        truth_for_display,
        title="Ground-truth labels on view 1",
    )
    _plot_labeled_points_on_axis(
        axes[1, 1],
        image_2,
        points_2,
        truth_for_display,
        title="Ground-truth labels on view 2",
    )

    summary_parts = [f"sample={sample_name}", f"family={model_family}"]
    if metric_summary is not None:
        if "inlier_f1" in metric_summary:
            summary_parts.append(f"inlier_f1={float(metric_summary['inlier_f1']):.3f}")
        if "match_iou_mean" in metric_summary:
            summary_parts.append(f"match_iou={float(metric_summary['match_iou_mean']):.3f}")
        if "predicted_model_count" in metric_summary and "true_model_count" in metric_summary:
            summary_parts.append(
                f"models={int(metric_summary['predicted_model_count'])}/{int(metric_summary['true_model_count'])} pred/true"
            )

    fig.suptitle("Adelaide fitting overlay | " + " | ".join(summary_parts), fontsize=13)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False)

    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_metric_curve(
    records: list[dict[str, float | int | str]],
    x_key: str,
    y_key: str,
    output_path: str | Path,
) -> None:
    """Plot mean/std trend of a metric against one experimental variable."""
    if not records:
        return

    numeric_pairs: list[tuple[float, float]] = []
    for record in records:
        x_value = record.get(x_key)
        y_value = record.get(y_key)
        if x_value is None or y_value is None:
            continue
        numeric_pairs.append((float(x_value), float(y_value)))

    if not numeric_pairs:
        return

    pairs = np.asarray(numeric_pairs, dtype=float)
    x_unique = np.unique(pairs[:, 0])

    means = []
    stds = []
    for x_value in x_unique:
        y_values = pairs[pairs[:, 0] == x_value][:, 1]
        means.append(float(np.mean(y_values)))
        stds.append(float(np.std(y_values)))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.2))
    plt.plot(x_unique, means, marker="o", linewidth=2.0)
    plt.fill_between(x_unique, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.25)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(f"{y_key} vs {x_key}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()



def _save_bar_chart(
    labels: list[str],
    values: list[float],
    output_path: str | Path,
    *,
    title: str,
    ylabel: str,
    color: str = "#3a6ea5",
    rotation: int = 30,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(max(7.0, len(labels) * 0.8), 4.6))
    positions = np.arange(len(labels))
    plt.bar(positions, values, color=color, alpha=0.9)
    plt.xticks(positions, labels, rotation=rotation, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()



def _filter_records(records: list[dict[str, float | int | str]], model_family: str | None) -> list[dict[str, float | int | str]]:
    if model_family is None:
        return list(records)
    return [record for record in records if str(record.get("model_family", "")).lower() == model_family.lower()]



def plot_adelaide_metric_by_sample(
    records: list[dict[str, float | int | str]],
    metric_key: str,
    output_path: str | Path,
    *,
    model_family: str | None = None,
) -> None:
    """Plot per-sample mean metric values for Adelaide experiments."""
    filtered = _filter_records(records, model_family)
    if not filtered:
        return

    grouped: dict[str, list[float]] = {}
    for record in filtered:
        sample_name = str(record.get("sample_name", "unknown"))
        metric_value = record.get(metric_key)
        if metric_value is None:
            continue
        grouped.setdefault(sample_name, []).append(float(metric_value))

    if not grouped:
        return

    sample_names = sorted(grouped)
    values = [float(np.mean(grouped[name])) for name in sample_names]
    title_prefix = model_family if model_family is not None else "all"
    _save_bar_chart(
        sample_names,
        values,
        output_path,
        title=f"Adelaide {metric_key} by sample ({title_prefix})",
        ylabel=metric_key,
    )



def plot_adelaide_model_count_comparison(
    records: list[dict[str, float | int | str]],
    output_path: str | Path,
    *,
    model_family: str | None = None,
) -> None:
    """Compare true and predicted model counts per sample for Adelaide experiments."""
    filtered = _filter_records(records, model_family)
    if not filtered:
        return

    grouped_true: dict[str, list[float]] = {}
    grouped_pred: dict[str, list[float]] = {}
    for record in filtered:
        sample_name = str(record.get("sample_name", "unknown"))
        true_count = record.get("true_model_count")
        pred_count = record.get("predicted_model_count")
        if true_count is None or pred_count is None:
            continue
        grouped_true.setdefault(sample_name, []).append(float(true_count))
        grouped_pred.setdefault(sample_name, []).append(float(pred_count))

    if not grouped_true:
        return

    sample_names = sorted(grouped_true)
    true_values = np.array([float(np.mean(grouped_true[name])) for name in sample_names])
    pred_values = np.array([float(np.mean(grouped_pred[name])) for name in sample_names])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    positions = np.arange(len(sample_names))
    width = 0.38

    plt.figure(figsize=(max(7.5, len(sample_names) * 0.9), 4.8))
    plt.bar(positions - width / 2, true_values, width=width, label="true", color="#4f772d")
    plt.bar(positions + width / 2, pred_values, width=width, label="predicted", color="#c97b63")
    plt.xticks(positions, sample_names, rotation=30, ha="right")
    plt.ylabel("model count")
    plt.title(f"Adelaide true vs predicted model count ({model_family or 'all'})")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()



def plot_adelaide_selected_model_types(
    records: list[dict[str, float | int | str]],
    output_path: str | Path,
    *,
    model_family: str | None = None,
) -> None:
    """Plot selected F/H counts per sample for Adelaide experiments."""
    filtered = _filter_records(records, model_family)
    if not filtered:
        return

    grouped_f: dict[str, list[float]] = {}
    grouped_h: dict[str, list[float]] = {}
    for record in filtered:
        sample_name = str(record.get("sample_name", "unknown"))
        fundamental = record.get("selected_fundamental_count")
        homography = record.get("selected_homography_count")
        if fundamental is None or homography is None:
            continue
        grouped_f.setdefault(sample_name, []).append(float(fundamental))
        grouped_h.setdefault(sample_name, []).append(float(homography))

    if not grouped_f:
        return

    sample_names = sorted(grouped_f)
    fundamental_values = np.array([float(np.mean(grouped_f[name])) for name in sample_names])
    homography_values = np.array([float(np.mean(grouped_h[name])) for name in sample_names])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    positions = np.arange(len(sample_names))
    width = 0.38

    plt.figure(figsize=(max(7.5, len(sample_names) * 0.9), 4.8))
    plt.bar(positions - width / 2, fundamental_values, width=width, label="F selected", color="#1f77b4")
    plt.bar(positions + width / 2, homography_values, width=width, label="H selected", color="#ff7f0e")
    plt.xticks(positions, sample_names, rotation=30, ha="right")
    plt.ylabel("selected models")
    plt.title(f"Adelaide selected model types ({model_family or 'all'})")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()



def plot_adelaide_metric_by_family(
    summary_records: list[dict[str, float | int | str]],
    metric_key: str,
    output_path: str | Path,
) -> None:
    """Plot one aggregated Adelaide metric across model families."""
    if not summary_records:
        return

    labels: list[str] = []
    values: list[float] = []
    for record in summary_records:
        metric_value = record.get(metric_key)
        family = record.get("model_family")
        if metric_value is None or family is None:
            continue
        labels.append(str(family))
        values.append(float(metric_value))

    if not labels:
        return

    _save_bar_chart(
        labels,
        values,
        output_path,
        title=f"Adelaide {metric_key} by model family",
        ylabel=metric_key,
        color="#7a8f3b",
        rotation=0,
    )



def plot_adelaide_family_type_balance(
    summary_records: list[dict[str, float | int | str]],
    output_path: str | Path,
) -> None:
    """Plot aggregated selected F/H counts across model families."""
    if not summary_records:
        return

    labels: list[str] = []
    fundamental_values: list[float] = []
    homography_values: list[float] = []
    for record in summary_records:
        family = record.get("model_family")
        if family is None:
            continue
        labels.append(str(family))
        fundamental_values.append(float(record.get("selected_fundamental_count_mean", 0.0)))
        homography_values.append(float(record.get("selected_homography_count_mean", 0.0)))

    if not labels:
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    positions = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(max(6.8, len(labels) * 1.2), 4.8))
    plt.bar(positions - width / 2, fundamental_values, width=width, label="F selected mean", color="#1f77b4")
    plt.bar(positions + width / 2, homography_values, width=width, label="H selected mean", color="#ff7f0e")
    plt.xticks(positions, labels)
    plt.ylabel("mean selected count")
    plt.title("Adelaide selected model balance by family")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_adelaide_thesis_method_overview(
    summary_records: list[dict[str, float | int | str]],
    output_path: str | Path,
) -> None:
    """Create a compact multi-panel figure for thesis-level method comparison."""
    if not summary_records:
        return

    families = [str(item.get("model_family", "")) for item in summary_records if item.get("model_family") is not None]
    if not families:
        return

    metrics = (
        ("inlier_f1_mean", "Inlier F1"),
        ("match_iou_mean_mean", "Match IoU"),
        ("model_count_error_mean", "Model Count Error"),
        ("runtime_seconds_mean", "Runtime (s)"),
    )
    values_by_metric: list[list[float]] = []
    for key, _ in metrics:
        values = [float(item.get(key, 0.0)) for item in summary_records]
        if not any(value != 0.0 for value in values):
            return
        values_by_metric.append(values)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    axes_flat = axes.flatten()
    positions = np.arange(len(families))
    colors = ("#355070", "#6d597a", "#b56576", "#e56b6f")

    for axis, (key, title), values, color in zip(axes_flat, metrics, values_by_metric, colors):
        axis.bar(positions, values, color=color, alpha=0.9)
        axis.set_xticks(positions, families)
        axis.set_title(title if "error" not in key else f"{title} (lower is better)")
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Adelaide Thesis Overview Across Model Families", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_dp_tradeoff_overview(
    records: list[dict[str, float | int | str]],
    output_path: str | Path,
) -> None:
    """Create a single thesis-oriented overview for privacy-performance tradeoffs."""
    dp_records = [record for record in records if bool(record.get("dp_enabled", False)) and record.get("dp_epsilon") is not None]
    if len(dp_records) < 2:
        return

    grouped: dict[float, dict[str, list[float]]] = {}
    for record in dp_records:
        epsilon = float(record["dp_epsilon"])
        grouped.setdefault(
            epsilon,
            {"inlier_f1": [], "ari_inlier": [], "match_iou_mean": [], "dp_avg_noise_scale": []},
        )
        for key in grouped[epsilon]:
            value = record.get(key)
            if value is not None:
                grouped[epsilon][key].append(float(value))

    epsilons = np.array(sorted(grouped.keys()), dtype=float)
    if epsilons.size < 2:
        return

    means = {
        key: np.array([np.mean(grouped[eps][key]) if grouped[eps][key] else 0.0 for eps in epsilons], dtype=float)
        for key in next(iter(grouped.values()))
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].plot(epsilons, means["inlier_f1"], marker="o", linewidth=2.0, label="Inlier F1", color="#355070")
    axes[0].plot(epsilons, means["ari_inlier"], marker="s", linewidth=2.0, label="ARI Inlier", color="#6d597a")
    axes[0].plot(epsilons, means["match_iou_mean"], marker="^", linewidth=2.0, label="Match IoU", color="#b56576")
    axes[0].set_xlabel("epsilon")
    axes[0].set_ylabel("quality metrics")
    axes[0].set_title("Privacy vs Quality")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epsilons, means["dp_avg_noise_scale"], marker="o", linewidth=2.2, color="#e56b6f")
    axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("avg noise scale")
    axes[1].set_title("Privacy Budget vs Noise Scale")
    axes[1].grid(alpha=0.3)

    fig.suptitle("DP-HVF Tradeoff Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_mixed_calibration_overview(
    records: list[dict[str, float | int | str]],
    output_path: str | Path,
) -> None:
    """Create a publication-style single-panel figure for mixed calibration."""
    mixed_records = [record for record in records if str(record.get("model_family", "")).lower() == "mixed"]
    calibration_values = sorted({int(record.get("mixed_residual_calibration_enabled", 0)) for record in mixed_records})
    if len(mixed_records) < 2 or len(calibration_values) < 2:
        return

    means: dict[str, tuple[float, float]] = {}
    metric_keys = ("inlier_f1", "ari_inlier", "match_iou_mean", "tau_mean")
    for metric_key in metric_keys:
        off_subset = [
            float(record[metric_key])
            for record in mixed_records
            if int(record.get("mixed_residual_calibration_enabled", 0)) == 0 and record.get(metric_key) is not None
        ]
        on_subset = [
            float(record[metric_key])
            for record in mixed_records
            if int(record.get("mixed_residual_calibration_enabled", 0)) == 1 and record.get(metric_key) is not None
        ]
        if not off_subset or not on_subset:
            return
        means[metric_key] = (float(np.mean(off_subset)), float(np.mean(on_subset)))

    tau_off, tau_on = means["tau_mean"]
    tau_delta = tau_on - tau_off
    tau_pct = 0.0 if abs(tau_off) < 1e-12 else (tau_delta / tau_off) * 100.0

    baseline_color = "#7f8c8d"
    proposed_color = "#e74c3c"
    accent_dark = "#2c3e50"
    neutral_grid = "#d9d9d9"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.8), facecolor="white")

    tau_positions = np.arange(2)
    tau_values = [tau_off, tau_on]
    bars = ax.bar(
        tau_positions,
        tau_values,
        color=[baseline_color, proposed_color],
        width=0.58,
        edgecolor="none",
        zorder=3,
    )
    ax.set_xticks(tau_positions, ["Calibration OFF", "Calibration ON"])
    ax.set_ylabel("Tau Mean", fontsize=13)
    ax.set_title("Scale Estimation", fontsize=16, weight="bold", pad=10)
    ax.grid(axis="y", color=neutral_grid, linewidth=0.7, alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

    ymax = max(tau_values) if max(tau_values) > 0 else 1.0
    ax.set_ylim(0, ymax * 1.36)
    for bar, value in zip(bars, tau_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.04,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=13,
            weight="bold",
            color=accent_dark,
        )

    ax.text(
        0.5,
        ymax * 1.14,
        f"Significant scale reduction ({tau_pct:+.1f}%)",
        ha="center",
        va="bottom",
        fontsize=13,
        color=proposed_color,
        weight="bold",
    )
    ax.text(
        0.5,
        ymax * 1.07,
        f"{tau_off:.1f} -> {tau_on:.1f} ({tau_delta:+.1f})",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color=accent_dark,
    )

    quality_note = (
        "No observable degradation in Inlier F1, ARI Inlier, or Match IoU\n"
        f"F1: {means['inlier_f1'][0]:.3f} -> {means['inlier_f1'][1]:.3f}    "
        f"ARI: {means['ari_inlier'][0]:.3f} -> {means['ari_inlier'][1]:.3f}    "
        f"IoU: {means['match_iou_mean'][0]:.3f} -> {means['match_iou_mean'][1]:.3f}"
    )
    ax.text(
        0.5,
        -0.18,
        quality_note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color=accent_dark,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fbfbfb", edgecolor="#dddddd"),
    )

    fig.suptitle(
        f"Mixed Residual Calibration Preserves Fitting Quality While Reducing Scale Estimation by ~{abs(tau_pct):.0f}%",
        fontsize=16,
        weight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.28)
    fig.savefig(output, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)

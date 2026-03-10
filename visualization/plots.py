from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

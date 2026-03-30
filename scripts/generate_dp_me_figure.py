"""
Generate ME% comparison figure for DP-HVF table (Table 4-2).

Values are taken directly from the thesis table to ensure
the figure matches the written results exactly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Exact values from Table 4-2 ──────────────────────────────────────────────
SAMPLES = ["BC", "BCC", "CHC", "C", "CBTC", "CC", "CT", "TCC", "Avg."]

TABLE_DATA: dict[str, list[float]] = {
    "Auto-HVF":         [38.03, 45.83, 47.80, 35.53, 50.32, 34.37, 42.18, 41.82, 41.98],
    r"DP-HVF ($\varepsilon$=2.0)": [49.70, 44.52, 46.34, 25.97, 57.83, 34.22, 46.28, 48.79, 44.21],
    r"DP-HVF ($\varepsilon$=1.0)": [52.10, 51.39, 49.63, 31.59, 55.35, 39.64, 40.59, 48.38, 46.09],
    r"DP-HVF ($\varepsilon$=0.5)": [55.45, 48.61, 46.59, 29.56, 55.61, 40.00, 46.78, 47.37, 46.25],
    r"DP-HVF ($\varepsilon$=0.2)": [51.67, 54.78, 54.02, 32.00, 62.87, 40.72, 47.70, 54.14, 49.74],
}

# Bold = lowest ME% per column (lower is better)
def _bold_mask(data: dict[str, list[float]]) -> dict[str, list[bool]]:
    n_cols = len(SAMPLES)
    methods = list(data.keys())
    bold: dict[str, list[bool]] = {m: [False] * n_cols for m in methods}
    for col_idx in range(n_cols):
        col_vals = [data[m][col_idx] for m in methods]
        min_val = min(col_vals)
        for m in methods:
            if data[m][col_idx] == min_val:
                bold[m][col_idx] = True
    return bold


PALETTE = {
    "Auto-HVF":   "#1f3a5f",
    "eps_2.0":    "#2a9d8f",
    "eps_1.0":    "#4e8da3",
    "eps_0.5":    "#e67e22",
    "eps_0.2":    "#c0392b",
}

METHOD_COLORS = [
    PALETTE["Auto-HVF"],
    PALETTE["eps_2.0"],
    PALETTE["eps_1.0"],
    PALETTE["eps_0.5"],
    PALETTE["eps_0.2"],
]

METHOD_MARKERS = ["o", "s", "^", "D", "v"]
METHOD_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#bfc5cc",
            "text.color": "#333333",
            "svg.fonttype": "none",
        }
    )

    methods = list(TABLE_DATA.keys())
    bold = _bold_mask(TABLE_DATA)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), facecolor="white")
    fig.suptitle(
        "ME% of DP-HVF under Different Privacy Budgets on AdelaideCubeFH",
        fontsize=13,
        weight="bold",
        y=1.01,
        color="#222222",
    )

    # ── Left panel: line chart per sample ─────────────────────────────────
    ax_line = axes[0]
    x = np.arange(len(SAMPLES))
    for idx, (method, color, marker, ls) in enumerate(
        zip(methods, METHOD_COLORS, METHOD_MARKERS, METHOD_LINESTYLES)
    ):
        vals = TABLE_DATA[method]
        ax_line.plot(
            x,
            vals,
            marker=marker,
            color=color,
            linewidth=1.8,
            linestyle=ls,
            markersize=5.5,
            label=method,
            zorder=3,
        )
        # Mark bold (minimum) points with a larger hollow circle
        for xi, (v, is_bold) in enumerate(zip(vals, bold[method])):
            if is_bold:
                ax_line.plot(xi, v, "o", color=color, markersize=11,
                             markerfacecolor="none", markeredgewidth=2.2, zorder=4)

    ax_line.set_xticks(x, SAMPLES)
    ax_line.set_ylabel("ME%")
    ax_line.set_title("Per-Scene ME% (circled = column minimum)")
    ax_line.set_ylim(0, 75)
    ax_line.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.5)
    ax_line.spines["top"].set_visible(False)
    ax_line.spines["right"].set_visible(False)
    ax_line.legend(frameon=False, loc="upper right", fontsize=8.5)

    # Shade the "Avg." column region
    ax_line.axvspan(len(SAMPLES) - 1.5, len(SAMPLES) - 0.5, color="#f0f4f8", zorder=0, alpha=0.7)
    ax_line.text(
        len(SAMPLES) - 1,
        2,
        "Avg.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#888",
        style="italic",
    )

    # ── Right panel: grouped bar chart for scene averages only ────────────
    ax_bar = axes[1]
    avg_vals = [TABLE_DATA[m][-1] for m in methods]
    bar_positions = np.arange(len(methods))
    bars = ax_bar.bar(
        bar_positions,
        avg_vals,
        color=METHOD_COLORS,
        width=0.58,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    # Annotate bars
    for bar, val, method in zip(bars, avg_vals, methods):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )
    # Mark the minimum bar with a bold border
    min_avg = min(avg_vals)
    for bar, val in zip(bars, avg_vals):
        if val == min_avg:
            bar.set_edgecolor("#1f3a5f")
            bar.set_linewidth(2.2)

    short_labels = ["Auto-HVF", "ε=2.0", "ε=1.0", "ε=0.5", "ε=0.2"]
    ax_bar.set_xticks(bar_positions, short_labels)
    ax_bar.set_ylabel("Average ME%")
    ax_bar.set_title("Average ME% Across All Scenes (bold border = best)")
    ax_bar.set_ylim(0, 58)
    ax_bar.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "thesis_figures_for_paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "thesis_fig_dp_me_comparison"

    for suffix in (".png", ".svg", ".pdf"):
        fig = build_figure()
        fig.savefig(stem.with_suffix(suffix), dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {stem.with_suffix(suffix)}")


if __name__ == "__main__":
    main()

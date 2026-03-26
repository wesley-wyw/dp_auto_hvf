from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

PALETTE = {
    "blue": "#1f3a5f",
    "teal": "#4e8da3",
    "orange": "#e67e22",
    "gray": "#bfc5cc",
    "text": "#333333",
    "light": "#eef2f5",
    "off": "#aeb7c2",
}

FAMILY_COLORS = {
    "fundamental": PALETTE["blue"],
    "homography": PALETTE["teal"],
    "mixed": PALETTE["orange"],
}

NON_DP_SUMMARY_CANDIDATES = (
    "codex_ch4_non_dp_check_summary.csv",
    "adelaide_paper_pack_v2_summary.csv",
    "adelaide_paper_pack_summary.csv",
)

DP_SWEEP_CSV_CANDIDATES = (
    "codex_ch4_dp_check.csv",
    "adelaide_dp_sweep_pack.csv",
)

MIXED_CALIBRATION_CSV_CANDIDATES = (
    "codex_ch4_calib_check.csv",
    "adelaide_thesis_visual_pub_v7.csv",
)


def apply_style() -> None:
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
            "axes.edgecolor": PALETTE["gray"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "text.color": PALETTE["text"],
            "svg.fonttype": "none",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_output_path(candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = OUTPUT_DIR / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No available output file found for candidates: {candidates}")


def save_all(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".pdf", ".png"):
        fig.savefig(OUTPUT_DIR / f"{stem}{suffix}", dpi=300, facecolor="white", bbox_inches="tight")


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["gray"])
    ax.spines["bottom"].set_color(PALETTE["gray"])
    ax.grid(axis="y", color=PALETTE["gray"], linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)


def plot_method_comparison() -> None:
    fig = plt.figure(figsize=(11.4, 5.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.95, "Method Comparison", fontsize=16, weight="bold", ha="left", va="top")
    ax.text(
        0.05,
        0.905,
        "The thesis progresses from baseline HVF to adaptive Auto-HVF and privacy-aware DP-HVF while keeping one shared fitting backbone.",
        fontsize=9.6,
        ha="left",
        va="top",
    )

    left = 0.06
    top = 0.83
    col_w = 0.27
    gap = 0.045
    card_h = 0.61
    header_h = 0.10

    methods = [
        {
            "name": "HVF",
            "subtitle": "Fixed-threshold baseline",
            "color": PALETTE["blue"],
            "sections": [("Backbone", "Fixed-threshold HVF pipeline")],
            "contribution": "One shared inlier scale across\nfitting, voting, and extraction.",
            "note": "No adaptive scale estimation and no privacy perturbation.",
        },
        {
            "name": "Auto-HVF",
            "subtitle": "Adaptive scale refinement",
            "color": PALETTE["teal"],
            "sections": [
                ("Backbone", "Same HVF fitting backbone"),
                ("Added module", "AIKOSE scale estimation\nbefore preference construction"),
            ],
            "contribution": "Improves robustness when\nresidual scale varies across scenes.",
            "note": "Adaptive scale enters before\npreference construction and voting.",
        },
        {
            "name": "DP-HVF",
            "subtitle": "Adaptive fitting with differential privacy",
            "color": PALETTE["orange"],
            "sections": [
                ("Backbone", "Adaptive HVF core\ninherited from Auto-HVF"),
                ("Privacy injection", "DP noise on model scoring\nand model selection"),
            ],
            "contribution": "Protects sensitive decision statistics\nwhile keeping post-DP evaluation.",
            "note": "Privacy is localized to sensitive decision\nstages instead of the full pipeline.",
        },
    ]

    for idx, method in enumerate(methods):
        x = left + idx * (col_w + gap)
        y = top - card_h
        color = method["color"]

        ax.add_patch(Rectangle((x, y), col_w, card_h, facecolor="white", edgecolor=PALETTE["gray"], linewidth=1.4))
        ax.add_patch(Rectangle((x, top - header_h), col_w, header_h, facecolor=PALETTE["light"], edgecolor=color, linewidth=1.8))
        ax.text(x + 0.018, top - 0.035, method["name"], fontsize=14, weight="bold", color=color, ha="left", va="center")
        ax.text(x + 0.018, top - 0.073, method["subtitle"], fontsize=9.2, color=PALETTE["text"], ha="left", va="center")

        section_y = top - header_h - 0.022
        for section_title, section_body in method["sections"]:
            box_h = 0.088
            ax.add_patch(Rectangle((x + 0.018, section_y - box_h), col_w - 0.036, box_h, facecolor="white", edgecolor=color, linewidth=1.15))
            ax.add_patch(Rectangle((x + 0.018, section_y - 0.032), col_w - 0.036, 0.032, facecolor=PALETTE["light"], edgecolor=color, linewidth=1.15))
            ax.text(x + 0.032, section_y - 0.016, section_title, fontsize=9.1, weight="bold", color=color, ha="left", va="center")
            ax.text(x + 0.032, section_y - 0.054, section_body, fontsize=8.4, color=PALETTE["text"], ha="left", va="center", linespacing=1.3)
            section_y -= box_h + 0.02

        ax.text(x + 0.02, y + 0.17, "Main contribution", fontsize=9.3, weight="bold", color=color, ha="left", va="center")
        ax.text(x + 0.02, y + 0.125, method["contribution"], fontsize=8.4, color=PALETTE["text"], ha="left", va="center", linespacing=1.3)

        ax.plot([x + 0.018, x + col_w - 0.018], [y + 0.095, y + 0.095], color=PALETTE["gray"], linewidth=1.0)
        ax.text(x + 0.02, y + 0.055, method["note"], fontsize=8.2, color=PALETTE["text"], ha="left", va="center", linespacing=1.3)

        if idx < len(methods) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + col_w + 0.008, top - 0.05),
                    (x + col_w + gap - 0.008, top - 0.05),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.25,
                    color=PALETTE["gray"],
                )
            )

    ax.text(0.50, 0.10, "Shared narrative: baseline fitting -> adaptive scale estimation -> privacy-preserving decision stages", fontsize=9.4, ha="center", va="center")

    save_all(fig, "thesis_fig_method_comparison")
    plt.close(fig)


def draw_pipeline_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, *, edge: str, fill: str = "white", title_fill: str | None = None) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=1.25))
    if title_fill is not None:
        ax.add_patch(Rectangle((x, y + h - 0.07), w, 0.07, facecolor=title_fill, edgecolor=edge, linewidth=1.25))
        ax.text(x + 0.015, y + h - 0.035, title, ha="left", va="center", fontsize=10.5, weight="bold", color=edge)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=10.2, color=PALETTE["text"])


def draw_pipeline_arrow(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=PALETTE["gray"],
        )
    )


def plot_hvf_pipeline() -> None:
    fig = plt.figure(figsize=(11.0, 3.6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.92, "HVF Pipeline", fontsize=15, weight="bold", ha="left", va="top")
    ax.text(0.05, 0.87, "Baseline robust multi-model fitting with one fixed inlier threshold.", fontsize=9.3, ha="left", va="top")
    y = 0.38
    h = 0.24
    w = 0.15
    xs = [0.05, 0.235, 0.42, 0.605, 0.79]
    labels = [
        "Input\ncorrespondences",
        "Hypothesis\ngeneration",
        "Residuals and\npreference matrix",
        "Hierarchical voting\nand clustering",
        "Model extraction\nand labels",
    ]

    for x, label in zip(xs, labels):
        draw_pipeline_box(ax, x, y, w, h, label, edge=PALETTE["blue"], fill="white")
    for idx in range(len(xs) - 1):
        draw_pipeline_arrow(ax, xs[idx] + w, xs[idx + 1], y + h / 2)

    ax.text(0.5, 0.16, "Baseline HVF relies on a fixed inlier threshold throughout preference construction and downstream model separation.", ha="center", va="center", fontsize=9.4)
    save_all(fig, "thesis_fig_hvf_pipeline")
    plt.close(fig)


def plot_auto_hvf_pipeline() -> None:
    fig = plt.figure(figsize=(11.0, 4.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.94, "Auto-HVF Pipeline", fontsize=15, weight="bold", ha="left", va="top")
    ax.text(0.05, 0.89, "Adaptive scale estimation is inserted before preference construction and voting.", fontsize=9.3, ha="left", va="top")
    y = 0.42
    h = 0.22
    w = 0.14
    xs = [0.05, 0.22, 0.39, 0.58, 0.77]
    labels = [
        "Input\ncorrespondences",
        "Hypothesis\ngeneration",
        "Residual\nanalysis",
        "Adaptive preference\nand voting",
        "Model extraction\nand labels",
    ]

    for idx, (x, label) in enumerate(zip(xs, labels)):
        edge = PALETTE["teal"] if idx in (2, 3) else PALETTE["blue"]
        fill = PALETTE["light"] if idx in (2, 3) else "white"
        draw_pipeline_box(ax, x, y, w, h, label, edge=edge, fill=fill)
    for idx in range(len(xs) - 1):
        draw_pipeline_arrow(ax, xs[idx] + w, xs[idx + 1], y + h / 2)

    draw_pipeline_box(ax, 0.38, 0.14, 0.16, 0.11, "AIKOSE scale\nestimation", edge=PALETTE["teal"], fill=PALETTE["light"])
    ax.add_patch(
        FancyArrowPatch(
            (0.46, 0.25),
            (0.46, y),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=PALETTE["teal"],
        )
    )
    ax.text(0.5, 0.07, "Auto-HVF replaces the fixed threshold with data-driven scale estimation before preference construction and voting.", ha="center", va="center", fontsize=9.4)
    save_all(fig, "thesis_fig_auto_hvf_pipeline")
    plt.close(fig)


def plot_dp_hvf_pipeline() -> None:
    fig = plt.figure(figsize=(11.2, 4.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.95, "DP-HVF Pipeline", fontsize=15, weight="bold", ha="left", va="top")
    ax.text(0.05, 0.90, "Differential privacy perturbs only sensitive decision stages and keeps evaluation on final post-DP output.", fontsize=9.3, ha="left", va="top")
    y = 0.46
    h = 0.20
    w = 0.13
    xs = [0.05, 0.20, 0.35, 0.52, 0.69, 0.84]
    labels = [
        "Input\ncorrespondences",
        "Adaptive\nHVF core",
        "Model\nscoring",
        "Model\nselection",
        "Final labels\nand models",
        "Post-DP\nevaluation",
    ]

    for idx, (x, label) in enumerate(zip(xs, labels)):
        edge = PALETTE["orange"] if idx in (2, 3, 5) else PALETTE["blue"]
        fill = PALETTE["light"] if idx in (2, 3, 5) else "white"
        draw_pipeline_box(ax, x, y, w, h, label, edge=edge, fill=fill)
    for idx in range(len(xs) - 1):
        draw_pipeline_arrow(ax, xs[idx] + w, xs[idx + 1], y + h / 2)

    draw_pipeline_box(ax, 0.34, 0.17, 0.14, 0.12, "DP noise", edge=PALETTE["orange"], fill="white")
    draw_pipeline_box(ax, 0.51, 0.17, 0.14, 0.12, "DP noise", edge=PALETTE["orange"], fill="white")
    ax.add_patch(FancyArrowPatch((0.41, 0.29), (0.41, y), arrowstyle="-|>", mutation_scale=12, linewidth=1.25, color=PALETTE["orange"]))
    ax.add_patch(FancyArrowPatch((0.58, 0.29), (0.58, y), arrowstyle="-|>", mutation_scale=12, linewidth=1.25, color=PALETTE["orange"]))
    ax.text(0.5, 0.07, "Differential privacy is injected at model scoring and model selection; evaluation is performed on the final post-DP output.", ha="center", va="center", fontsize=9.4)
    save_all(fig, "thesis_fig_dp_hvf_pipeline")
    plt.close(fig)


def plot_mixed_geometry_calibration() -> None:
    fig = plt.figure(figsize=(11.2, 4.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.94, "Mixed Geometry and Residual Calibration", fontsize=15, weight="bold", ha="left", va="top")

    draw_pipeline_box(ax, 0.05, 0.36, 0.23, 0.24, "Mixed F / H\ninput scene", edge=PALETTE["blue"], fill="white")
    draw_pipeline_box(ax, 0.37, 0.36, 0.23, 0.24, "Same-type and cross-type\nconsistency handling", edge=PALETTE["blue"], fill=PALETTE["light"])
    draw_pipeline_box(ax, 0.69, 0.36, 0.23, 0.24, "Calibrated mixed\ncomparison", edge=PALETTE["teal"], fill=PALETTE["light"])
    draw_pipeline_arrow(ax, 0.28, 0.37, 0.48)
    draw_pipeline_arrow(ax, 0.60, 0.69, 0.48)

    draw_pipeline_box(ax, 0.69, 0.14, 0.23, 0.12, "Residual quantile calibration", edge=PALETTE["teal"], fill="white")
    ax.add_patch(FancyArrowPatch((0.805, 0.26), (0.805, 0.36), arrowstyle="-|>", mutation_scale=12, linewidth=1.25, color=PALETTE["teal"]))

    ax.text(0.5, 0.08, "Mixed scenes require stricter cross-type structure handling and residual calibration to make F/H hypotheses more comparable.", ha="center", va="center", fontsize=9.4)
    save_all(fig, "thesis_fig_mixed_geometry_calibration")
    plt.close(fig)


def plot_dp_budget_allocation() -> None:
    fig = plt.figure(figsize=(10.6, 4.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.94, "DP Budget Allocation", fontsize=15, weight="bold", ha="left", va="top")

    draw_pipeline_box(ax, 0.07, 0.50, 0.16, 0.16, "Total privacy\nbudget", edge=PALETTE["orange"], fill="white")
    draw_pipeline_box(ax, 0.34, 0.50, 0.20, 0.16, "Equal allocation", edge=PALETTE["orange"], fill=PALETTE["light"])
    draw_pipeline_box(ax, 0.62, 0.50, 0.20, 0.16, "Manual allocation", edge=PALETTE["orange"], fill=PALETTE["light"])
    draw_pipeline_arrow(ax, 0.23, 0.34, 0.58)
    draw_pipeline_arrow(ax, 0.54, 0.62, 0.58)

    draw_pipeline_box(ax, 0.31, 0.21, 0.12, 0.12, "Scoring", edge=PALETTE["orange"], fill="white")
    draw_pipeline_box(ax, 0.45, 0.21, 0.12, 0.12, "Selection", edge=PALETTE["orange"], fill="white")
    ax.text(0.44, 0.40, "split evenly", fontsize=9.2, ha="center", va="center")
    ax.add_patch(FancyArrowPatch((0.44, 0.50), (0.37, 0.33), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=PALETTE["gray"]))
    ax.add_patch(FancyArrowPatch((0.44, 0.50), (0.51, 0.33), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=PALETTE["gray"]))

    draw_pipeline_box(ax, 0.66, 0.21, 0.12, 0.12, "Scoring", edge=PALETTE["orange"], fill="white")
    draw_pipeline_box(ax, 0.80, 0.21, 0.12, 0.12, "Selection", edge=PALETTE["orange"], fill="white")
    ax.text(0.79, 0.40, "weighted by importance", fontsize=9.2, ha="center", va="center")
    ax.add_patch(FancyArrowPatch((0.72, 0.50), (0.72, 0.33), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=PALETTE["gray"]))
    ax.add_patch(FancyArrowPatch((0.78, 0.50), (0.86, 0.33), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=PALETTE["gray"]))

    ax.text(0.5, 0.08, "The codebase supports both equal and manual budget splits so privacy can be analyzed at the injection-point level.", ha="center", va="center", fontsize=9.4)
    save_all(fig, "thesis_fig_dp_budget_allocation")
    plt.close(fig)


def plot_metric_system() -> None:
    fig = plt.figure(figsize=(11.0, 4.6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.94, "Evaluation Metric System", fontsize=15, weight="bold", ha="left", va="top")

    groups = [
        (0.05, "Fitting quality", PALETTE["blue"], ["Precision", "Recall", "Inlier F1"]),
        (0.29, "Clustering quality", PALETTE["teal"], ["ARI", "NMI", "Label purity"]),
        (0.53, "Matching quality", PALETTE["blue"], ["GT best IoU", "Pred best IoU", "Match IoU"]),
        (0.77, "Runtime and privacy", PALETTE["orange"], ["Runtime", "Epsilon", "Noise scale"]),
    ]

    for x, title, color, lines in groups:
        ax.add_patch(Rectangle((x, 0.26), 0.18, 0.42, facecolor="white", edgecolor=color, linewidth=1.35))
        ax.add_patch(Rectangle((x, 0.60), 0.18, 0.08, facecolor=PALETTE["light"], edgecolor=color, linewidth=1.35))
        ax.text(x + 0.015, 0.64, title, fontsize=10.2, weight="bold", color=color, ha="left", va="center")
        yy = 0.53
        for line in lines:
            ax.text(x + 0.018, yy, line, fontsize=9.4, ha="left", va="center")
            yy -= 0.10

    ax.text(0.5, 0.12, "The thesis evaluates fitting, clustering, matching, runtime, and privacy cost as complementary aspects of robust multi-model fitting performance.", ha="center", va="center", fontsize=9.4)
    save_all(fig, "thesis_fig_metric_system")
    plt.close(fig)


def plot_adelaide_task_setup() -> None:
    fig = plt.figure(figsize=(11.2, 5.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.94, "AdelaideRMF Task Setup", fontsize=15, weight="bold", ha="left", va="top")
    ax.text(
        0.05,
        0.90,
        "The real-data evaluation covers three scene families, with mixed scenes introducing the strongest cross-geometry ambiguity.",
        fontsize=9.6,
        ha="left",
        va="top",
    )

    panel_y = 0.53
    panel_h = 0.27
    panel_w = 0.25
    panel_gap = 0.055
    xs = [0.05, 0.05 + panel_w + panel_gap, 0.05 + 2 * (panel_w + panel_gap)]
    panels = [
        ("Fundamental", PALETTE["blue"], ["Two-view epipolar structure", "Single-family geometry", "Separate multiple F models"]),
        ("Homography", PALETTE["teal"], ["Planar / near-planar structure", "Single-family geometry", "Separate multiple H models"]),
        ("Mixed", PALETTE["orange"], ["Fundamental + homography coexist", "Cross-type comparison", "F/H comparability"]),
    ]

    for x, (title, color, lines) in zip(xs, panels):
        ax.add_patch(Rectangle((x, panel_y), panel_w, panel_h, facecolor="white", edgecolor=color, linewidth=1.35))
        ax.add_patch(Rectangle((x, panel_y + panel_h - 0.07), panel_w, 0.07, facecolor=PALETTE["light"], edgecolor=color, linewidth=1.35))
        ax.text(x + 0.015, panel_y + panel_h - 0.035, title, fontsize=11, weight="bold", color=color, ha="left", va="center")
        yy = panel_y + panel_h - 0.095
        for line in lines:
            ax.text(x + 0.018, yy, line, fontsize=9.0, ha="left", va="top")
            yy -= 0.058

    challenge_x = 0.12
    challenge_y = 0.34
    challenge_w = 0.76
    challenge_h = 0.10
    ax.add_patch(Rectangle((challenge_x, challenge_y), challenge_w, challenge_h, facecolor="white", edgecolor=PALETTE["gray"], linewidth=1.3))
    ax.add_patch(Rectangle((challenge_x, challenge_y + challenge_h - 0.045), challenge_w, 0.045, facecolor=PALETTE["light"], edgecolor=PALETTE["gray"], linewidth=1.3))
    ax.text(challenge_x + 0.015, challenge_y + challenge_h - 0.022, "Shared evaluation target", fontsize=10.1, weight="bold", ha="left", va="center")
    ax.text(
        challenge_x + 0.018,
        challenge_y + 0.030,
        "Estimate multiple structures, reject outliers, and preserve agreement with scene-level ground truth under one unified protocol.",
        fontsize=9.0,
        ha="left",
        va="center",
    )

    bar_x = 0.12
    bar_y = 0.15
    bar_w = 0.76
    bar_h = 0.13
    ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, facecolor="white", edgecolor=PALETTE["gray"], linewidth=1.35))
    ax.add_patch(Rectangle((bar_x, bar_y + bar_h - 0.05), bar_w, 0.05, facecolor=PALETTE["light"], edgecolor=PALETTE["gray"], linewidth=1.35))
    ax.text(bar_x + 0.015, bar_y + bar_h - 0.025, "Unified post-DP real-data evaluation", fontsize=10.3, weight="bold", ha="left", va="center")
    ax.text(bar_x + 0.018, bar_y + 0.060, "ARI / NMI / label purity / IoU matching / runtime / privacy trade-off", fontsize=9.1, ha="left", va="center")
    ax.text(bar_x + 0.018, bar_y + 0.028, "Common output view: predicted labels, model count, fitting quality, and privacy cost", fontsize=8.9, ha="left", va="center")

    for x in xs:
        center = x + panel_w / 2
        ax.add_patch(FancyArrowPatch((center, panel_y), (center, challenge_y + challenge_h), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=PALETTE["gray"]))

    ax.add_patch(FancyArrowPatch((0.50, challenge_y), (0.50, bar_y + bar_h), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=PALETTE["gray"]))

    ax.text(0.5, 0.07, "Mixed scenes are the most demanding because the evaluation must compare structures across different geometric families without changing the protocol.", fontsize=9.2, ha="center", va="center")
    save_all(fig, "thesis_fig_adelaide_task_setup")
    plt.close(fig)



def plot_method_overview() -> None:
    rows = read_csv(resolve_output_path(NON_DP_SUMMARY_CANDIDATES))
    families = [row["model_family"] for row in rows]
    labels = [family.capitalize() for family in families]
    metrics = [
        ("inlier_f1_mean", "Inlier F1", False),
        ("runtime_seconds_mean", "Runtime (s)", False),
        ("model_count_error_mean", "Model Count Error", True),
        ("tau_mean_mean", "Tau Mean", False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.5), facecolor="white")
    fig.suptitle("Adelaide Overview Across Model Families", fontsize=14, color=PALETTE["text"], weight="bold", y=1.02)

    for ax, (key, title, lower_is_better) in zip(axes, metrics):
        values = [float(row[key]) for row in rows]
        positions = np.arange(len(labels))
        colors = [FAMILY_COLORS[family] for family in families]
        ax.bar(positions, values, color=colors, width=0.62)
        ax.set_xticks(positions, labels)
        ax.set_title(title + ("  lower is better" if lower_is_better else ""))
        style_axes(ax)

    handles = [Line2D([0], [0], color=FAMILY_COLORS[family], linewidth=6) for family in families]
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    save_all(fig, "thesis_fig_method_overview")
    plt.close(fig)


def plot_dp_tradeoff() -> None:
    rows = read_csv(resolve_output_path(DP_SWEEP_CSV_CANDIDATES))
    grouped: dict[float, dict[str, list[float]]] = {}
    for row in rows:
        if row["dp_enabled"] != "True":
            continue
        eps = float(row["dp_epsilon"])
        grouped.setdefault(eps, {"inlier_f1": [], "ari_inlier": [], "match_iou_mean": [], "dp_avg_noise_scale": []})
        for key in grouped[eps]:
            grouped[eps][key].append(float(row[key]))

    epsilons = np.array(sorted(grouped.keys()), dtype=float)
    means = {key: np.array([np.mean(grouped[eps][key]) for eps in epsilons], dtype=float) for key in next(iter(grouped.values()))}

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), facecolor="white")
    fig.suptitle("DP Trade-off Overview", fontsize=14, color=PALETTE["text"], weight="bold", y=1.03)

    axes[0].plot(epsilons, means["inlier_f1"], marker="o", linewidth=2.0, color=PALETTE["blue"], label="Inlier F1")
    axes[0].plot(epsilons, means["ari_inlier"], marker="s", linewidth=2.0, color=PALETTE["teal"], label="ARI Inlier")
    axes[0].plot(epsilons, means["match_iou_mean"], marker="^", linewidth=2.0, color=PALETTE["orange"], label="Match IoU")
    axes[0].set_xlabel("Epsilon")
    axes[0].set_ylabel("Quality")
    axes[0].set_title("Privacy vs Quality")
    axes[0].legend(frameon=False, loc="best")
    style_axes(axes[0])

    axes[1].plot(epsilons, means["dp_avg_noise_scale"], marker="o", linewidth=2.2, color=PALETTE["orange"])
    axes[1].fill_between(epsilons, 0, means["dp_avg_noise_scale"], color=PALETTE["orange"], alpha=0.08)
    axes[1].set_xlabel("Epsilon")
    axes[1].set_ylabel("Average noise scale")
    axes[1].set_title("Privacy vs Noise")
    style_axes(axes[1])

    fig.tight_layout()
    save_all(fig, "thesis_fig_dp_tradeoff_overview")
    plt.close(fig)


def plot_mixed_calibration() -> None:
    rows = [
        row
        for row in read_csv(resolve_output_path(MIXED_CALIBRATION_CSV_CANDIDATES))
        if row["model_family"] == "mixed"
    ]
    off = [row for row in rows if int(row["mixed_residual_calibration_enabled"]) == 0]
    on = [row for row in rows if int(row["mixed_residual_calibration_enabled"]) == 1]

    def mean(subset: list[dict[str, str]], key: str) -> float:
        return float(np.mean([float(item[key]) for item in subset]))

    tau_off = mean(off, "tau_mean")
    tau_on = mean(on, "tau_mean")
    f1_off, f1_on = mean(off, "inlier_f1"), mean(on, "inlier_f1")
    iou_off, iou_on = mean(off, "match_iou_mean"), mean(on, "match_iou_mean")

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.3), facecolor="white")
    fig.suptitle("Mixed Calibration Overview", fontsize=14, color=PALETTE["text"], weight="bold", y=0.98)

    positions = np.arange(2)
    bars = ax.bar(positions, [tau_off, tau_on], width=0.56, color=[PALETTE["off"], PALETTE["teal"]])
    ax.set_xticks(positions, ["Calibration OFF", "Calibration ON"])
    ax.set_ylabel("Tau Mean")
    ax.set_title("Scale Estimation")
    style_axes(ax)

    ymax = max(tau_off, tau_on)
    ax.set_ylim(0, ymax * 1.28)
    for bar, value in zip(bars, [tau_off, tau_on]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.04, f"{value:.1f}", ha="center", va="bottom", fontsize=10)

    reduction = 0.0 if abs(tau_off) < 1e-12 else (tau_on - tau_off) / tau_off * 100.0
    ax.text(
        0.5,
        ymax * 1.12,
        f"Tau mean reduced by {abs(reduction):.1f}% after calibration",
        ha="center",
        va="bottom",
        fontsize=10,
        color=PALETTE["teal"],
        weight="bold",
    )

    note = (
        "Quality remains unchanged in the current A/B result: "
        f"Inlier F1 {f1_off:.3f} -> {f1_on:.3f}; "
        f"Match IoU {iou_off:.3f} -> {iou_on:.3f}."
    )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=9.4, color=PALETTE["text"])

    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    save_all(fig, "thesis_fig_mixed_calibration_overview")
    plt.close(fig)



def main() -> None:
    apply_style()
    plot_method_comparison()
    plot_hvf_pipeline()
    plot_auto_hvf_pipeline()
    plot_dp_hvf_pipeline()
    plot_mixed_geometry_calibration()
    plot_dp_budget_allocation()
    plot_metric_system()
    plot_adelaide_task_setup()
    plot_method_overview()
    plot_dp_tradeoff()
    plot_mixed_calibration()


if __name__ == "__main__":
    main()

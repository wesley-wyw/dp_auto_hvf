from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


PALETTE = {
    "blue": "#1f3a5f",
    "teal": "#4e8da3",
    "orange": "#e67e22",
    "gray": "#cccccc",
    "text": "#222222",
}


METHODS = [
    {
        "name": "HVF",
        "subtitle": "fixed threshold",
        "color": PALETTE["blue"],
        "scores": {"Structure": 2, "Noise": 1, "Privacy": 0},
    },
    {
        "name": "Auto-HVF",
        "subtitle": "adaptive scale",
        "color": PALETTE["teal"],
        "scores": {"Structure": 3, "Noise": 3, "Privacy": 0},
    },
    {
        "name": "DP-HVF",
        "subtitle": "adaptive + DP",
        "color": PALETTE["orange"],
        "scores": {"Structure": 2, "Noise": 2, "Privacy": 3},
    },
]


ROW_LABELS = ("Structure", "Noise", "Privacy")


def build_figure() -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(11.0, 5.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    line_w = 1.35
    border_w = 1.45
    left = 0.06
    right = 0.96
    bottom = 0.09
    col_w = 0.24
    gap = 0.06
    card_y = 0.31
    card_h = 0.54

    ax.text(
        left,
        0.955,
        "Robust Multi-Model Fitting: Method Comparison",
        ha="left",
        va="top",
        fontsize=18,
        color=PALETTE["text"],
        weight="bold",
    )

    for idx in range(2):
        start = left + idx * (col_w + gap) + col_w
        end = left + (idx + 1) * (col_w + gap)
        ax.add_patch(
            FancyArrowPatch(
                (start + 0.01, card_y + card_h + 0.028),
                (end - 0.01, card_y + card_h + 0.028),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=line_w,
                color=PALETTE["gray"],
            )
        )

    for idx, method in enumerate(METHODS):
        x = left + idx * (col_w + gap)
        ax.add_patch(
            Rectangle(
                (x, card_y),
                col_w,
                card_h,
                facecolor="white",
                edgecolor=PALETTE["gray"],
                linewidth=border_w,
            )
        )
        ax.plot([x, x + col_w], [card_y + card_h - 0.105, card_y + card_h - 0.105], color=method["color"], linewidth=2.2)
        ax.text(
            x + 0.02,
            card_y + card_h - 0.05,
            method["name"],
            ha="left",
            va="center",
            fontsize=15,
            color=method["color"],
            weight="bold",
        )
        ax.text(
            x + 0.02,
            card_y + card_h - 0.09,
            method["subtitle"],
            ha="left",
            va="center",
            fontsize=10.5,
            color=PALETTE["text"],
        )

        icon_y = card_y + card_h - 0.18
        square = 0.028
        features = (
            PALETTE["blue"],
            PALETTE["teal"] if idx > 0 else PALETTE["gray"],
            PALETTE["orange"] if idx == 2 else PALETTE["gray"],
        )
        for feature_idx, fill in enumerate(features):
            icon_x = x + 0.02 + feature_idx * 0.048
            ax.add_patch(
                Rectangle(
                    (icon_x, icon_y),
                    square,
                    square,
                    facecolor=fill,
                    edgecolor=fill if fill != PALETTE["gray"] else "#b8b8b8",
                    linewidth=line_w,
                )
            )

        label_x = x + 0.02
        bar_x = x + 0.10
        segment_w = 0.034
        bar_gap = 0.008
        row_start = card_y + card_h - 0.31
        row_gap = 0.13

        for row_idx, row_label in enumerate(ROW_LABELS):
            y = row_start - row_idx * row_gap
            ax.text(label_x, y + 0.015, row_label, ha="left", va="center", fontsize=10.5, color=PALETTE["text"])
            level = int(method["scores"][row_label])
            for seg in range(3):
                seg_x = bar_x + seg * (segment_w + bar_gap)
                active = seg < level
                fill = method["color"] if active else "white"
                edge = method["color"] if active else PALETTE["gray"]
                ax.add_patch(
                    Rectangle(
                        (seg_x, y),
                        segment_w,
                        0.028,
                        facecolor=fill,
                        edgecolor=edge,
                        linewidth=line_w,
                    )
                )

    ax.plot([left, right], [0.22, 0.22], color=PALETTE["gray"], linewidth=line_w)

    summary_boxes = [
        ("Improvement", "Auto-HVF improves noise adaptation\nwithout weakening structure extraction.", PALETTE["teal"], left),
        ("Trade-off", "DP-HVF introduces privacy-preserving noise\nwhile retaining most structural information.", PALETTE["orange"], left + 0.32),
        ("Effect", "Structure, noise, and privacy are exposed\nas explicit design dimensions.", PALETTE["blue"], left + 0.64),
    ]
    box_w = 0.26
    box_h = 0.10
    for title, body, color, x in summary_boxes:
        ax.add_patch(
            Rectangle(
                (x, bottom),
                box_w,
                box_h,
                facecolor="white",
                edgecolor=color,
                linewidth=border_w,
            )
        )
        ax.text(x + 0.015, bottom + 0.068, title, ha="left", va="center", fontsize=11.5, color=color, weight="bold")
        ax.text(x + 0.015, bottom + 0.031, body, ha="left", va="center", fontsize=9.5, color=PALETTE["text"])

    return fig


def main() -> None:
    output_root = Path("outputs")
    output_root.mkdir(parents=True, exist_ok=True)
    stem = output_root / "hvf_method_comparison_publication"

    for suffix in (".svg", ".pdf", ".png"):
        fig = build_figure()
        fig.savefig(stem.with_suffix(suffix), dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()

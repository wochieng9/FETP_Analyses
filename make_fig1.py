"""Draw Figure 1: model stocks, flows, and conditional feedbacks."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#1d2026"
BLUE = "#2b3a55"
RED = "#b23a2e"
GREEN = "#2f6b4f"
GOLD = "#c08a2e"
MUTED = "#637083"
PALE = "#f2f5f8"


def box(ax, x, y, w, h, title, sub="", fc=PALE, ec=BLUE, fs=11):
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.04,rounding_size=0.08",
                           facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=3)
    ax.add_patch(patch)
    ax.text(x, y + (0.11 if sub else 0), title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=4)
    if sub:
        ax.text(x, y - 0.20, sub, ha="center", va="center", fontsize=8.3,
                color=MUTED, zorder=4)


def arrow(ax, start, end, label, color=BLUE, fs=8.2):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                            linewidth=1.9, color=color, zorder=2)
    ax.add_patch(patch)
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.15,
                label, ha="center", va="center", fontsize=fs, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.95),
                zorder=5)


def feedback_card(ax, x, title, line1, line2, ec):
    w, h, y = 3.55, 2.15, 1.77
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.06,rounding_size=0.08",
                           facecolor="white", edgecolor=ec, linewidth=1.5, zorder=2)
    ax.add_patch(patch)
    ax.text(x - w / 2 + 0.18, y + 0.72, title, ha="left", va="center",
            fontsize=10.3, fontweight="bold", color=ec, zorder=3)
    ax.text(x, y + 0.14, line1, ha="center", va="center", fontsize=9.0,
            color=INK, zorder=3, wrap=True, linespacing=1.25)
    ax.text(x, y - 0.42, line2, ha="center", va="center", fontsize=8.0,
            color=MUTED, zorder=3, wrap=True)


def main():
    fig, ax = plt.subplots(figsize=(15.5, 8.6))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    ax.text(0.35, 8.24,
            "Figure 1. Stocks, flows, and feedbacks in the workforce model",
            fontsize=15, fontweight="bold", color=INK, ha="left")
    ax.text(0.38, 7.77, "WORKFORCE STOCKS AND FLOWS",
            fontsize=9, fontweight="bold", color=MUTED, ha="left")

    # Main stock-and-flow chain.
    y = 6.85
    box(ax, 1.70, y, 2.55, 1.02, "Training pipelines",
        "intake · duration · completion", fc="#f7f2e8", ec=GOLD)
    box(ax, 5.05, y, 2.20, 1.02, "Graduation",
        "tier completion", fc="#f7f2e8", ec=GOLD)
    box(ax, 8.75, y, 2.95, 1.02, "Unabsorbed pool",
        "public seekers without a post", fc="#fdf0ee", ec=RED)
    box(ax, 12.70, y, 2.85, 1.02, "Serving workforce",
        "post holders · age structured", fc="#eef2f7", ec=BLUE)
    arrow(ax, (2.98, y), (3.92, y), "completers", GOLD)
    arrow(ax, (6.16, y), (7.25, y), "public entrants", RED)
    arrow(ax, (10.24, y), (11.24, y), "post absorption", GREEN)

    # Exit flows and the non-public route.
    box(ax, 5.05, 5.05, 2.45, 0.68, "Outside tracked workforce",
        "non-public completers", fc="white", ec=MUTED, fs=9.0)
    box(ax, 8.75, 5.05, 2.45, 0.68, "Pool exits",
        "attrition · emigration · mortality", fc="#fdf0ee", ec=RED, fs=8.4)
    box(ax, 12.70, 5.05, 2.60, 0.68, "Serving exits",
        "attrition · emigration · mortality · retirement",
        fc="#eef2f7", ec=BLUE, fs=8.2)
    arrow(ax, (5.05, 6.33), (5.05, 5.41), "non-public", MUTED, fs=7.7)
    arrow(ax, (8.75, 6.33), (8.75, 5.41), "", RED)
    arrow(ax, (12.70, 6.33), (12.70, 5.41), "", BLUE)

    callout = FancyBboxPatch((1.10, 4.10), 13.30, 0.45,
                             boxstyle="round,pad=0.04,rounding_size=0.05",
                             facecolor="#f7f9fb", edgecolor="#c3cad6",
                             linewidth=0.9, zorder=2)
    ax.add_patch(callout)
    ax.text(7.75, 4.325,
            "Internal progression transfers the same person between tiers, preserving serving or pool status.",
            ha="center", va="center", fontsize=8.7, color=BLUE, zorder=3)

    ax.text(0.38, 3.68, "CONDITIONAL FEEDBACK STRUCTURE",
            fontsize=9, fontweight="bold", color=MUTED, ha="left")
    feedback_card(ax, 2.20, "R1 · Mentor capacity",
                  "Serving advanced staff → mentor\ncapacity → training intake",
                  "Mentor share and training load are\nanalyst-selected.", RED)
    feedback_card(ax, 5.95, "R2 · Workload attrition",
                  "Workload / serving capability →\nload ratio → serving attrition",
                  "Load-to-attrition link is a scenario\nassumption.", RED)
    feedback_card(ax, 9.70, "R3 · Pool emigration",
                  "Unabsorbed share → pool emigration\nhazard → pool exits",
                  "The added hazard applies to the\nunabsorbed pool only.", RED)
    feedback_card(ax, 13.45, "B1 · Funded-post response",
                  "Benchmark − serving − vacancies\n→ perceived gap → funded posts",
                  "Post creation is bounded by a fiscal\nceiling and delay.", GREEN)

    ax.text(0.38, 0.27,
            "Benchmark account: counted tiers, serving + unabsorbed.  Capability account: serving staff in all tiers only. "
            "Structural checks do not establish empirical calibration.",
            fontsize=8.2, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(pad=0.4)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"fig1_stock_and_flow.{ext}", dpi=220,
                    bbox_inches="tight", facecolor="white")
    print("wrote fig1_stock_and_flow.{svg,pdf,png}")


if __name__ == "__main__":
    main()

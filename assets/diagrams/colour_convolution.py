"""Draw assets/colour_convolution.png for notebook 01.

A. One filter on a colour image: each of the three channels is convolved with its own 3 x 3 slice of the filter,
   the three results are added up with the bias, and the result is one feature map.
B. A layer with several filters: every filter gives one feature map, and the maps are stacked into a box of
   height x width x filters.

Run from the repository root: uv run python assets/diagrams/colour_convolution.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

OUT = Path(__file__).parent.parent / "colour_convolution.png"
INK, GREY, AMBER = "#222222", "#6b6b6b", "#c47612"
CHANNELS = [("red", "#d9534f", "#f6d5d4"), ("green", "#3a8a4f", "#d6eadb"), ("blue", "#2f6fb0", "#d5e3f2")]

fig, ax = plt.subplots(figsize=(15, 6.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 31.5)
ax.set_ylim(-10.6, 1.6)
ax.set_aspect("equal")
ax.axis("off")


def grid(x, y, n, edge, face, cell=0.55):
    """Draw an n x n grid with its top-left corner at (x, y)."""
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((x + c * cell, y - (r + 1) * cell), cell, cell, facecolor=face, edgecolor=edge,
                                   linewidth=0.9))


def arrow(x0, y0, x1, y1, colour=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=5,head_width=3", color=colour,
                                 linewidth=1.3))


def box(x, y, width, height, depth, face, edge, label):
    """Draw a 3D box (front face width x height, receding depth) with its front bottom-left corner at (x, y)."""
    dx, dy = depth * 0.6, depth * 0.45
    ax.add_patch(Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)], facecolor=face,
                         edgecolor=edge, linewidth=1.4))
    ax.add_patch(Polygon([(x, y + height), (x + dx, y + height + dy), (x + width + dx, y + height + dy),
                          (x + width, y + height)], facecolor="white", edgecolor=edge, linewidth=1.4))
    ax.add_patch(Polygon([(x + width, y), (x + width + dx, y + dy), (x + width + dx, y + height + dy),
                          (x + width, y + height)], facecolor="#eeeeee", edgecolor=edge, linewidth=1.4))
    ax.text(x + width / 2 + dx / 2, y - 0.5, label, ha="center", va="top", fontsize=11, color=INK)


# Panel A
ax.text(0, 1.1, "A. One filter on a colour image", fontsize=14, color=INK, va="center")
for i, (name, edge, face) in enumerate(CHANNELS):
    y = -0.2 - i * 2.75
    grid(0.4, y, 6, edge, face, cell=0.37)
    ax.text(0.4 + 1.1, y - 2.3, f"{name} channel, 6 × 6", ha="center", va="top", fontsize=9.5, color=edge)
    ax.text(3.2, y - 1.1, "∗", fontsize=20, ha="center", va="center", color=INK)
    grid(3.8, y - 0.28, 3, edge, "white", cell=0.55)
    ax.text(3.8 + 0.82, y - 2.0, f"{name} slice", ha="center", va="top", fontsize=9.5, color=edge)
    arrow(5.6, y - 1.1, 7.4, -3.4 + (1 - i) * 0.9)
ax.text(9.0, -3.4, "add up\n+ bias\nthen ReLU", ha="center", va="center", fontsize=11, color=INK,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fdf6ec", "edgecolor": AMBER})
arrow(10.3, -3.4, 11.5, -3.4, AMBER)
grid(11.7, -2.3, 4, AMBER, "#fdf6ec", cell=0.55)
ax.text(12.8, -4.6, "one feature map, 4 × 4", ha="center", va="top", fontsize=10, color=AMBER)
ax.text(7.0, -8.6, "the filter has the depth of the input:\n3 × 3 × 3 = 27 weights + 1 bias", ha="center",
        va="top", fontsize=10.5, color=GREY, linespacing=1.4)

# Panel B
ax.text(16.5, 1.1, "B. A layer with 4 filters: the output is a box", fontsize=14, color=INK, va="center")
box(17.0, -6.6, 3.0, 3.0, 1.6, "#f6e7e6", "#8a3b38", "input\n6 × 6 × 3")
ax.text(23.0, -4.4, "4 filters\n3 × 3 × 3", ha="center", va="center", fontsize=11, color=AMBER)
arrow(21.6, -5.3, 24.4, -5.3, AMBER)
box(25.0, -6.4, 2.0, 2.0, 1.6, "#fdf6ec", AMBER, "output\n4 × 4 × 4")
for i in range(1, 4):  # the boundaries between the 4 stacked feature maps
    dx, dy = i * 1.6 / 4 * 0.6, i * 1.6 / 4 * 0.45
    ax.plot([25.0 + dx, 27.0 + dx], [-4.4 + dy, -4.4 + dy], color=AMBER, linewidth=0.9)
    ax.plot([27.0 + dx, 27.0 + dx], [-6.4 + dy, -4.4 + dy], color=AMBER, linewidth=0.9)
ax.text(28.9, -4.2, "one slice\nper filter", ha="left", va="center", fontsize=10, color=AMBER)
ax.text(23.2, -8.6, "height and width follow the output-size formula;\nthe depth is the number of filters",
        ha="center", va="top", fontsize=10.5, color=GREY, linespacing=1.4)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")

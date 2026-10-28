"""Draw assets/shift_invariance.png for notebook 01: what convolution, pooling and padding do when a feature moves.

A. An "eye" (one bright pixel) at three positions of an 8 x 8 image, convolved with a dot-detector filter, ReLU,
   then 2 x 2 max pooling. The feature map moves with the eye (equivariance); the pooled output stays the same
   for a shift inside a pooling window and moves when the shift crosses a window boundary.
B. The eye at the border of the image: without padding the filter never centres on it; with padding it does.

All values are computed, not typed in. Run from the repository root: uv run python assets/diagrams/shift_invariance.py
"""

from pathlib import Path

import numpy as np
from grids import AMBER, BLUE, GREY, INK, draw_grid, new_figure, outline
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).parent.parent / "shift_invariance.png"
GREEN, RED = "#3a8a4f", "#c0392b"
DOT = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])


def convolve(image, kernel):
    n, f = image.shape[0], kernel.shape[0]
    return np.array([[np.sum(image[r:r + f, c:c + f] * kernel) for c in range(n - f + 1)] for r in range(n - f + 1)])


def max_pool(feature_map):
    n = feature_map.shape[0] // 2 * 2
    return feature_map[:n, :n].reshape(n // 2, 2, n // 2, 2).max(axis=(1, 3))


def as_text(values):
    """Show non-zero values only, so the eye stands out."""
    return [[f"{v:g}" if v else "" for v in row] for row in values]


def arrow(ax, x0, x1, y, text, colour=GREY):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>,head_length=5,head_width=3", color=colour,
                                 linewidth=1.3))
    ax.text((x0 + x1) / 2, y + 0.35, text, ha="center", va="bottom", fontsize=9, color=colour)


C = 0.42  # cell size
fig, ax = new_figure(15, 12.5, (-0.3, 21.5), (-19.6, 1.6))
ax.text(-0.2, 1.2, "A. The eye moves to the right", fontsize=14, color=INK, va="center")
for label, x in [("image 8 × 8", 1.7), ("feature map after ReLU, 6 × 6", 7.6), ("after 2 × 2 max pooling", 12.7)]:
    ax.text(x, 0.45, label, ha="center", fontsize=10.5, color=GREY)

NOTES = [
    ("eye at column 1", "", BLUE),
    ("eye at column 2: shifted by one pixel", "the map shifts too; the pooled output\nis the same: the shift stays inside\none pooling window", GREEN),
    ("eye at column 3: shifted again", "the shift crosses a pooling window:\nthe pooled output moves by one cell", AMBER),
]
for i, (col, (title, note, colour)) in enumerate(zip([1, 2, 3], NOTES)):
    y0 = -0.1 - i * 4.15
    image = np.zeros((8, 8))
    image[3, col] = 1
    feature_map = np.maximum(0, convolve(image, DOT))
    pooled = max_pool(feature_map)
    draw_grid(ax, image.astype(int).tolist(), 0, y0, cell=C, fmt="", shade=lambda v: 0.85 * v)
    ax.text(-0.2, y0 - 8 * C - 0.25, title, fontsize=10, color=colour, va="top")
    arrow(ax, 8 * C + 0.4, 6.1, y0 - 4 * C, "convolve + ReLU")
    draw_grid(ax, as_text(feature_map), 6.4, y0 - C, cell=C, fmt="{}", fontsize=9,
              pale={tuple(p) for p in np.argwhere(feature_map > 0)})
    for r in range(0, 6, 2):  # the 2 x 2 pooling windows
        for c in range(0, 6, 2):
            outline(ax, 6.4, y0 - C, r, c, 2, colour=INK, cell=C, width=2.2)
    arrow(ax, 6.4 + 6 * C + 0.3, 11.9, y0 - 4 * C, "max pool")
    draw_grid(ax, as_text(pooled), 12.1, y0 - 2 * C, cell=2 * C * 0.7, fmt="{}", fontsize=11,
              pale={tuple(p) for p in np.argwhere(pooled > 0)})
    ax.text(14.6, y0 - 2.2 * C, note, fontsize=10, color=colour, va="top", linespacing=1.4)

# Panel B: the eye at the border, without and with padding
y0 = -14.0
ax.text(-0.2, y0 + 0.5, "B. The eye at the border of the image", fontsize=14, color=INK, va="center")
image = np.zeros((8, 8))
image[3, 0] = 1
padded = np.pad(image, 1)
valid = np.maximum(0, convolve(image, DOT))
same = np.maximum(0, convolve(padded, DOT))
B = C * 0.75

# without padding
ax.text(0, y0 - 0.2, "no padding", fontsize=11, color=RED, va="center")
draw_grid(ax, image.astype(int).tolist(), 0, y0 - 0.6, cell=B, fmt="", shade=lambda v: 0.85 * v)
arrow(ax, 8 * B + 0.2, 7.0, y0 - 0.6 - 4 * B, "convolve + ReLU", RED)
draw_grid(ax, as_text(valid), 7.2, y0 - 0.6 - B, cell=B, fmt="{}", fontsize=8)
ax.text(0, y0 - 0.6 - 8 * B - 0.35, f"6 × 6 map, maximum {valid.max():g}: the eye is never under\nthe centre of the filter, so it is not detected",
        fontsize=10, color=RED, va="top", linespacing=1.4)

# with padding
x0 = 11.4
ax.text(x0, y0 - 0.2, 'padding = 1 ("same")', fontsize=11, color=GREEN, va="center")
draw_grid(ax, padded.astype(int).tolist(), x0, y0 - 0.6, cell=B * 0.8, fmt="", shade=lambda v: 0.85 * v,
          pale={(r, c) for r in range(10) for c in range(10) if r in (0, 9) or c in (0, 9)})
arrow(ax, x0 + 10 * B * 0.8 + 0.2, x0 + 5.1, y0 - 0.6 - 4 * B, "convolve + ReLU", GREEN)
draw_grid(ax, as_text(same), x0 + 5.3, y0 - 0.6 - 0.5 * B, cell=B * 0.8, fmt="{}", fontsize=8,
          pale={tuple(p) for p in np.argwhere(same > 0)})
ax.text(x0, y0 - 0.6 - 8 * B - 0.35, f"8 × 8 map, maximum {same.max():g}: the ring of zeros lets the\nfilter centre on the edge pixel, so the eye is detected",
        fontsize=10, color=GREEN, va="top", linespacing=1.4)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")

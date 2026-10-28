"""Draw assets/face_convolution.png: a 6x6 face convolved with a 3x3 horizontal-line filter, for notebook 01.

The amber window and the amber cell show one step of the convolution: the window at row 3, column 1 of the image
gives the value 6 in the feature map.

Run from the repository root: uv run python assets/diagrams/face_convolution.py
"""

from pathlib import Path

import numpy as np
from grids import AMBER, GREY, INK, draw_grid, new_figure, outline

OUT = Path(__file__).parent.parent / "face_convolution.png"

FACE = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
])
KERNEL = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
FEATURE_MAP = np.array([[np.sum(FACE[r:r + 3, c:c + 3] * KERNEL) for c in range(4)] for r in range(4)])
ROW, COL = 3, 1  # the highlighted step

fig, ax = new_figure(13, 4.6, (-0.5, 19.5), (-7.3, 1.4))
draw_grid(ax, FACE.tolist(), 0, 0, title="image (6 × 6)", shade=lambda v: 0.85 * v)
outline(ax, 0, 0, ROW, COL, 3)
ax.text(7.0, -3, "∗", fontsize=30, ha="center", va="center", color=INK)
draw_grid(ax, KERNEL.tolist(), 8, -1.5, title="filter (3 × 3)", fmt="{:+d}",
          shade=lambda v: 0.0)
ax.text(12.0, -3, "=", fontsize=26, ha="center", va="center", color=INK)
draw_grid(ax, FEATURE_MAP.tolist(), 13, -1, title="feature map (4 × 4)", pale={(ROW, COL)})
outline(ax, 13, -1, ROW, COL, 1)
ax.text(9.5, -6.4, "window at row 3, column 1 (counted from 0):  0·(-1) + 0·(-1) + 0·(-1)  +  1·2 + 1·2 + 1·2  +  0·(-1) + 0·(-1) + 0·(-1)  =  6",
        ha="center", va="center", fontsize=11, color=AMBER)
ax.text(9.5, -7.0, "slide the filter one cell at a time; each position gives one number of the feature map",
        ha="center", va="center", fontsize=10, color=GREY)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")

"""Draw assets/max_pooling.png: 2x2 max pooling of the face feature map after ReLU, for notebook 01.

Run from the repository root: uv run python assets/diagrams/max_pooling.py
"""

from pathlib import Path

from grids import GREY, INK, draw_grid, new_figure
from matplotlib.patches import Rectangle

OUT = Path(__file__).parent.parent / "max_pooling.png"

FEATURE_MAP = [[2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [3, 6, 6, 3]]  # the face feature map after ReLU
COLOURS = ["#dbe8f5", "#fbe3c4", "#e3f0dc", "#eadff2"]  # one pale colour per 2x2 window

fig, ax = new_figure(9, 4, (-0.5, 11), (-5.2, 1.3))
draw_grid(ax, FEATURE_MAP, 0, 0, title="feature map after ReLU (4 × 4)")
for i, (r, c) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
    ax.add_patch(Rectangle((c, -(r + 2)), 2, 2, facecolor=COLOURS[i], edgecolor="none", zorder=0.5, alpha=1))
    for rr in range(2):
        for cc in range(2):
            ax.add_patch(Rectangle((c + cc, -(r + rr + 1)), 1, 1, facecolor=COLOURS[i], edgecolor=GREY, linewidth=1))
            ax.text(c + cc + 0.5, -(r + rr + 0.5), str(FEATURE_MAP[r + rr][c + cc]), ha="center", va="center",
                    fontsize=13, color=INK)
ax.text(5.5, -2, "→", fontsize=24, ha="center", va="center", color=INK)
ax.text(5.5, -2.7, "max of each\n2 × 2 window", fontsize=10, ha="center", va="top", color=GREY)
pooled = [[2, 2], [6, 6]]
for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
    ax.add_patch(Rectangle((7 + c, -(r + 2)), 1, 1, facecolor=COLOURS[i], edgecolor=GREY, linewidth=1))
    ax.text(7 + c + 0.5, -(r + 1.5), str(pooled[r][c]), ha="center", va="center", fontsize=13, color=INK)
ax.text(8, -0.55, "pooled (2 × 2)", ha="center", va="bottom", fontsize=13, color=INK)
ax.text(5, -4.8, "the mouth (6) and the eyes (2) survive; the size falls from 4 × 4 to 2 × 2",
        ha="center", va="center", fontsize=10, color=GREY)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")

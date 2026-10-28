"""Draw assets/padding_stride.png: zero padding and a stride of 2 on a 5x5 image with a 3x3 filter, for notebook 01.

Run from the repository root: uv run python assets/diagrams/padding_stride.py
"""

from pathlib import Path

from grids import AMBER, BLUE, GREY, INK, draw_grid, new_figure, outline

OUT = Path(__file__).parent.parent / "padding_stride.png"

fig, ax = new_figure(14, 5.6, (-0.5, 28.5), (-8.6, 2.4))

# Panel A: padding of 1 around a 5x5 image; output keeps the size 5x5
padded = [[0] * 7 for _ in range(7)]
ax.text(0, 1.6, "A. Padding = 1: the 5 × 5 image inside a ring of zeros", fontsize=13, color=INK, va="center")
draw_grid(ax, padded, 0, 0,
          fmt="", pale={(r, c) for r in range(7) for c in range(7) if r in (0, 6) or c in (0, 6)})
for r in range(1, 6):
    for c in range(1, 6):
        ax.text(c + 0.5, -r - 0.5, "x", ha="center", va="center", fontsize=12, color=INK)
for r in range(7):
    for c in range(7):
        if r in (0, 6) or c in (0, 6):
            ax.text(c + 0.5, -r - 0.5, "0", ha="center", va="center", fontsize=12, color=GREY)
outline(ax, 0, 0, 0, 0, 3, colour=AMBER)
ax.text(8.0, -3.5, "→", fontsize=24, ha="center", va="center", color=INK)
draw_grid(ax, [[""] * 5 for _ in range(5)], 9, -1, fmt="{}", title="5 × 5 output")
outline(ax, 9, -1, 0, 0, 1, colour=AMBER)
ax.text(7.0, -8.0, "(5 + 2·1 − 3) / 1 + 1 = 5:  the output keeps the size of the image",
        ha="center", va="center", fontsize=11, color=GREY)

# Panel B: stride 2 on a 5x5 image without padding; output 2x2
ax.text(18, 1.6, "B. Stride = 2: the window jumps 2 cells", fontsize=13, color=INK, va="center")
draw_grid(ax, [["x"] * 5 for _ in range(5)], 18, -1, fmt="{}")
outline(ax, 18, -1, 0, 0, 3, colour=AMBER)
outline(ax, 18, -1, 0, 2, 3, colour=BLUE, width=2.5)
ax.text(24.2, -3.5, "→", fontsize=24, ha="center", va="center", color=INK)
draw_grid(ax, [[""] * 2 for _ in range(2)], 25.3, -2.5, fmt="{}", title="2 × 2 output")
outline(ax, 25.3, -2.5, 0, 0, 1, colour=AMBER)
outline(ax, 25.3, -2.5, 0, 1, 1, colour=BLUE, width=2.5)
ax.text(22.5, -8.0, "(5 + 2·0 − 3) / 2 + 1 = 2", ha="center", va="center", fontsize=11, color=GREY)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")

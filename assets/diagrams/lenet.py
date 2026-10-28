"""Draw assets/lenet.png: the LeNet-5 style network of notebook 02, with the output shape of every layer.

Feature maps are drawn as stacks of squares (side proportional to the image size, one square per channel up to a
few); the dense part as bars. Shapes are those printed by model.summary() in notebook 02.

Run from the repository root: uv run python assets/diagrams/lenet.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).parent.parent / "lenet.png"
INK, GREY, BLUE, AMBER, GREEN = "#222222", "#6b6b6b", "#2f6fb0", "#c47612", "#3a8a4f"

fig, ax = plt.subplots(figsize=(16, 4.6), dpi=200)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 32.5)
ax.set_ylim(-3.2, 4.2)
ax.set_aspect("equal")
ax.axis("off")


def stack(x, side, channels, colour, label, shown=None):
    """Draw a feature map stack whose front square has the given side, centred vertically at 0.5."""
    shown = shown or min(channels, 6)
    step = 0.18
    for i in reversed(range(shown)):
        ax.add_patch(Rectangle((x + i * step, 0.5 - side / 2 + i * step), side, side, facecolor="white",
                               edgecolor=colour, linewidth=1.3))
    ax.text(x + side / 2 + shown * step / 2, -2.0, label, ha="center", va="top", fontsize=10.5, color=INK)
    return x + side + shown * step


def bar(x, height, colour, label):
    ax.add_patch(Rectangle((x, 0.5 - height / 2), 0.45, height, facecolor="white", edgecolor=colour, linewidth=1.5))
    ax.text(x + 0.22, -2.0, label, ha="center", va="top", fontsize=10.5, color=INK)
    return x + 0.45


def arrow(x0, x1, text, colour):
    ax.add_patch(FancyArrowPatch((x0 + 0.15, 0.5), (x1 - 0.15, 0.5), arrowstyle="-|>,head_length=5,head_width=3",
                                 color=colour, linewidth=1.4))
    ax.text((x0 + x1) / 2, 0.9, text, ha="center", va="bottom", fontsize=9.5, color=colour, linespacing=1.3)


SCALE = 3.4 / 28  # side of a 28 x 28 map
x = 0.3
end = stack(x, 28 * SCALE, 1, INK, "input\n28 × 28 × 1")
x = end + 1.9
arrow(end, x, "conv 5 × 5\n6 filters\nsame", BLUE)
end = stack(x, 28 * SCALE, 6, BLUE, "28 × 28 × 6")
x = end + 1.6
arrow(end, x, "max pool\n2 × 2", AMBER)
end = stack(x, 14 * SCALE, 6, AMBER, "14 × 14 × 6")
x = end + 1.9
arrow(end, x, "conv 5 × 5\n16 filters\nvalid", BLUE)
end = stack(x, 10 * SCALE, 16, BLUE, "10 × 10 × 16", shown=8)
x = end + 1.6
arrow(end, x, "max pool\n2 × 2", AMBER)
end = stack(x, 5 * SCALE, 16, AMBER, "5 × 5 × 16", shown=8)
x = end + 1.6
arrow(end, x, "flatten", GREY)
end = bar(x, 4.0, GREY, "400")
for units, height in [(120, 2.6), (84, 2.0)]:
    x = end + 1.8
    arrow(end, x, "dense\nReLU", GREEN)
    end = bar(x, height, GREEN, str(units))
x = end + 1.8
arrow(end, x, "dense\nsoftmax", GREEN)
bar(x, 1.2, GREEN, "10\nclasses")

ax.text(6.5, 3.75, "feature extraction: convolution and pooling", ha="center", fontsize=12, color=BLUE)
ax.text(26.5, 3.75, "classification: dense layers", ha="center", fontsize=12, color=GREEN)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")

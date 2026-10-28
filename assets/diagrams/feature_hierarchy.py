"""Draw assets/feature_hierarchy.png for notebook 01: how features of successive convolutional layers combine.

Layer 1 detects small edges, layer 2 combines edges into face parts, layer 3 combines parts into whole faces.
A schematic drawing, not the learnt filters of a real network (Lee et al. 2009 and Zeiler and Fergus 2014 show
those for real networks).

Run from the repository root: uv run python assets/diagrams/feature_hierarchy.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, ConnectionPatch, Ellipse

OUT = Path(__file__).parent.parent / "feature_hierarchy.png"
INK, GREY, BLUE, AMBER, GREEN = "#222222", "#6b6b6b", "#2f6fb0", "#c47612", "#3a8a4f"

fig = plt.figure(figsize=(15, 8.2), dpi=200)
fig.patch.set_facecolor("white")


def panel(x, y, size, colour):
    """Return an axis without ticks for one feature, framed in the colour of its layer."""
    ax = fig.add_axes([x, y, size, size * 15 / 8.2])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)
        spine.set_linewidth(1.6)
    return ax


def edge_patch(ax, angle=None, curve=False):
    """Draw a light/dark edge as a small greyscale image: a straight edge at an angle, or a curved one."""
    grid = np.linspace(-1, 1, 60)
    xx, yy = np.meshgrid(grid, -grid)
    if curve:
        value = np.hypot(xx, yy + 1.1) - 1.1  # distance from a circle: a curved edge
    else:
        value = np.cos(np.radians(angle)) * xx + np.sin(np.radians(angle)) * yy
    ax.imshow(np.tanh(6 * value), cmap="gray", extent=(-1, 1, -1, 1), vmin=-1.2, vmax=1.2)


# --- the parts of layer 2 and the faces of layer 3 are drawn with the same functions ---
def eye(ax, cx=0.0, cy=0.0, s=1.0):
    ax.add_patch(Ellipse((cx, cy), 1.3 * s, 0.7 * s, fill=False, color=INK, linewidth=2.2 * max(s, 0.6)))
    ax.add_patch(Ellipse((cx, cy), 0.35 * s, 0.35 * s, color=INK))


def eyebrow(ax, cx=0.0, cy=0.0, s=1.0):
    ax.add_patch(Arc((cx, cy - 0.35 * s), 1.4 * s, 0.9 * s, theta1=30, theta2=150, color=INK, linewidth=2.6 * max(s, 0.6)))


def nose(ax, cx=0.0, cy=0.0, s=1.0):
    ax.plot([cx, cx - 0.25 * s, cx + 0.15 * s], [cy + 0.6 * s, cy - 0.4 * s, cy - 0.45 * s], color=INK,
            linewidth=2.4 * max(s, 0.6), solid_capstyle="round")


def mouth(ax, cx=0.0, cy=0.0, s=1.0):
    ax.add_patch(Arc((cx, cy + 0.25 * s), 1.4 * s, 0.9 * s, theta1=200, theta2=340, color=INK, linewidth=2.6 * max(s, 0.6)))


def face(ax, wide=1.0, smile=True):
    ax.add_patch(Ellipse((0, 0), 1.45 * wide, 1.85, fill=False, color=INK, linewidth=2))
    for side in (-1, 1):
        eye(ax, side * 0.3 * wide, 0.25, 0.3)
        eyebrow(ax, side * 0.3 * wide, 0.47, 0.3)
    nose(ax, 0.0, 0.0, 0.3)
    if smile:
        mouth(ax, 0.0, -0.45, 0.4)
    else:
        ax.plot([-0.22, 0.22], [-0.48, -0.48], color=INK, linewidth=2)


# Column titles
for x, title, colour, note in [
    (0.03, "Layer 1: edges", BLUE, "each filter covers a few pixels"),
    (0.36, "Layer 2: parts", AMBER, "each part combines edges from layer 1"),
    (0.74, "Layer 3: faces", GREEN, "each face combines parts from layer 2"),
]:
    fig.text(x + 0.1, 0.95, title, ha="center", fontsize=15, color=colour)
    fig.text(x + 0.1, 0.915, note, ha="center", fontsize=10.5, color=GREY)

# Layer 1: six edge detectors in two columns
SPECS = {"vertical": 0, "horizontal": 90, "diagonal": 45, "other diagonal": 135, "curve": None, "corner": 60}
for i, (name, angle) in enumerate(SPECS.items()):
    x, y = 0.04 + (i % 2) * 0.12, 0.66 - (i // 2) * 0.27
    ax = panel(x, y, 0.085, BLUE)
    edge_patch(ax, angle, curve=(angle is None))
    ax.set_xlabel(name, fontsize=9.5, color=BLUE, labelpad=3)

# Layer 2: each part with the edges it combines, as a small sum: edge + edge = part
RECIPES = [("eye", eye, ["horizontal", "curve"]), ("eyebrow", eyebrow, ["curve", "other diagonal"]),
           ("nose", nose, ["vertical", "diagonal"]), ("mouth", mouth, ["curve", "corner"])]
parts = []
for i, (name, draw, used) in enumerate(RECIPES):
    y = 0.735 - i * 0.205
    for j, edge in enumerate(used):
        small = panel(0.335 + j * 0.06, y + 0.02, 0.04, BLUE)
        edge_patch(small, SPECS[edge], curve=(SPECS[edge] is None))
    fig.text(0.385, y + 0.057, "+", ha="center", va="center", fontsize=14, color=INK)
    fig.text(0.4575, y + 0.057, "=", ha="center", va="center", fontsize=16, color=INK)
    ax = panel(0.475, y, 0.065, AMBER)
    draw(ax)
    ax.set_xlabel(name, fontsize=10, color=AMBER, labelpad=2)
    parts.append(ax)

# Layer 3: two faces
faces = []
for i, (wide, smile) in enumerate([(1.0, True), (0.85, False)]):
    ax = panel(0.765, 0.53 - i * 0.39, 0.15, GREEN)
    face(ax, wide, smile)
    faces.append(ax)

for part in parts:
    for face_ax in faces:
        fig.add_artist(ConnectionPatch((1.0, 0.5), (0.0, 0.5), coordsA="axes fraction", coordsB="axes fraction",
                                       axesA=part, axesB=face_ax, color=AMBER, linewidth=1.1, alpha=0.7,
                                       arrowstyle="-|>", mutation_scale=12))

fig.text(0.5, 0.025, "Each layer covers a larger area of the image than the one before, because it works on the "
         "feature maps of the previous layer.\nSchematic drawing: real networks learn similar, less regular features.",
         ha="center", fontsize=11, color=GREY, linespacing=1.5)
fig.savefig(OUT, facecolor="white")

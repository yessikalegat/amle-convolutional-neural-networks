"""Shared drawing helpers for the grid figures in assets/: images, kernels and feature maps as numbered cells."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK, GREY, BLUE, AMBER, PALE = "#222222", "#6b6b6b", "#2f6fb0", "#c47612", "#fdf6ec"


def draw_grid(ax, values, x0, y0, cell=1.0, title=None, shade=None, fmt="{:g}", fontsize=13, pale=None):
    """Draw a 2D list of numbers as a grid with its top-left corner at (x0, y0).

    shade maps a value to a grey level (0 white, 1 black); pale lists (row, col) cells to fill in amber.
    """
    rows, cols = len(values), len(values[0])
    for r in range(rows):
        for c in range(cols):
            v = values[r][c]
            level = shade(v) if shade else 0.0
            face = (1 - level,) * 3
            if pale and (r, c) in pale:
                face = PALE
            ax.add_patch(Rectangle((x0 + c * cell, y0 - (r + 1) * cell), cell, cell, facecolor=face,
                                   edgecolor=GREY, linewidth=1))
            text_colour = "white" if level > 0.55 else INK
            ax.text(x0 + (c + 0.5) * cell, y0 - (r + 0.5) * cell, fmt.format(v), ha="center", va="center",
                    fontsize=fontsize, color=text_colour)
    if title:
        ax.text(x0 + cols * cell / 2, y0 + 0.45 * cell, title, ha="center", va="bottom", fontsize=13, color=INK)


def outline(ax, x0, y0, row, col, size, colour=AMBER, cell=1.0, width=3):
    """Draw a thick outline around a size x size window starting at (row, col) of a grid at (x0, y0)."""
    ax.add_patch(Rectangle((x0 + col * cell, y0 - (row + size) * cell), size * cell, size * cell, fill=False,
                           edgecolor=colour, linewidth=width, zorder=5))


def new_figure(width, height, xlim, ylim):
    """Return a figure and an axis without frame, with equal scaling."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax

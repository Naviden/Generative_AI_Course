"""Build the illustrative Lesson 1 scatter plot (requires NumPy and Matplotlib).

Run from any directory: python scripts/build_fidelity_coverage_figure.py
The points are constructed examples, not measurements from a trained model.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reading-group/images"
RADIUS = 0.44
rng = np.random.default_rng(17)
angles = np.pi / 2 + np.arange(8) * np.pi / 4
centres = 2.2 * np.column_stack((np.cos(angles), np.sin(angles)))


def cluster_points(centre, count):
    """Keep every example inside its explicitly defined real-data region."""
    points = []
    while len(points) < count:
        offset = rng.normal(0, 0.19, 2)
        if np.linalg.norm(offset) < RADIUS * 0.92:
            points.append(centre + offset)
    return np.array(points)


reference = np.concatenate([
    cluster_points(c, 13 if i < 4 else 12) for i, c in enumerate(centres)
])
generator_a = cluster_points(centres[0], 100)
generator_b_in = np.concatenate([cluster_points(c, 10) for c in centres])
outside = []
while len(outside) < 20:
    point = rng.uniform(-2.85, 2.85, 2)
    if np.min(np.linalg.norm(centres - point, axis=1)) > RADIUS + 0.20:
        outside.append(point)
generator_b_out = np.array(outside)

# Validate that the annotated counts describe the plotted points exactly.
def memberships(points):
    return np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2) <= RADIUS


assert len(reference) == len(generator_a) == len(generator_b_in) + len(generator_b_out) == 100
assert memberships(reference).any(axis=1).sum() == 100
assert memberships(generator_a).any(axis=1).sum() == 100
assert memberships(generator_a).any(axis=0).sum() == 1
assert memberships(generator_b_in).sum(axis=0).tolist() == [10] * 8
assert not memberships(generator_b_out).any()

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 3, figsize=(15, 6.8), facecolor="white")
fig.subplots_adjust(left=0.025, right=0.975, top=0.76, bottom=0.24, wspace=0.12)
ink, blue, orange = "#172b4d", "#087eaa", "#bd4819"
titles = ["Reference observations", "Generator A", "Generator B"]
subtitles = ["Eight groups of real points", "Realistic, but seven groups missing", "All groups reached, plus unrealistic points"]
scores = [
    "100 observations across 8 clusters",
    "100/100 in real regions | 1/8 clusters",
    "80/100 in real regions | 8/8 clusters",
]
for ax, title, subtitle, score in zip(axes, titles, subtitles, scores):
    for centre in centres:
        ax.add_patch(Circle(centre, RADIUS, facecolor="#e9f1f7", edgecolor="#92a9bd", lw=1.15))
    ax.set(xlim=(-3.1, 3.1), ylim=(-3.1, 3.1), aspect="equal")
    ax.axis("off")
    ax.text(0.5, 1.16, title, transform=ax.transAxes, ha="center", fontsize=19, weight="bold", color=ink)
    ax.text(0.5, 1.065, subtitle, transform=ax.transAxes, ha="center", fontsize=12.5, color=ink)
    ax.text(0.5, -0.095, score, transform=ax.transAxes, ha="center", fontsize=13.5, weight="bold", color=ink)

for ax, points in zip(axes, [reference, generator_a, generator_b_in]):
    ax.scatter(points[:, 0], points[:, 1], s=20, c=blue, alpha=0.85, edgecolors="white", linewidths=0.25, zorder=3)
axes[2].scatter(generator_b_out[:, 0], generator_b_out[:, 1], s=45, c=orange, marker="x", linewidths=1.8, zorder=4)

fig.suptitle("Realistic individual samples do not guarantee broad coverage", x=0.5, y=0.97, fontsize=22, weight="bold", color=ink)
handles = [
    Patch(facecolor="#e9f1f7", edgecolor="#92a9bd", label="Real-data region (an 'island')"),
    Line2D([], [], marker="o", ls="", color=blue, label="Point inside a real region"),
    Line2D([], [], marker="x", ls="", color=orange, label="Point outside all real regions"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.077), ncol=3, frameon=False, fontsize=13)
fig.text(0.5, 0.035, "Constructed example: 100 points per panel, identical coordinates and scale. Shaded regions are fixed across panels.", ha="center", fontsize=12, color="#42566c")
OUTPUT.mkdir(exist_ok=True)
fig.savefig(OUTPUT / "fidelity-coverage.png", dpi=220, facecolor="white")
fig.savefig(OUTPUT / "fidelity-coverage.svg", facecolor="white", metadata={"Date": None})
plt.close(fig)
print("Built fidelity-coverage.png and fidelity-coverage.svg; point counts verified.")

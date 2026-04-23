"""Minimal 2D example.

Writes two figures:

* ``example_2d.png``         — obstacles + inflated regions.
* ``example_2d_cliques.png`` — obstacles + clique convex hulls + clique points.

Run with ``uv run python scripts/example_2d.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pydrake.geometry.optimization import HPolyhedron, VPolytope

from convexobstaclesvcc.clique_cover import truncated_clique_cover
from convexobstaclesvcc.inflation import inflate_regions_from_cliques
from convexobstaclesvcc.plotting import (
    plot_hpoly_matplotlib,
    plot_hpoly_skeleton_matplotlib,
)
from convexobstaclesvcc.visibility_graph import (
    build_visibility_graph,
    sample_free_space,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "tmp"
OUT_DIR.mkdir(exist_ok=True)


def make_box(lo, hi) -> HPolyhedron:
    return HPolyhedron.MakeBox(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))


def make_polygon(vertices: np.ndarray) -> HPolyhedron:
    return HPolyhedron(VPolytope(vertices.T))


def setup_axes(ax, obstacles, domain) -> None:
    plot_hpoly_skeleton_matplotlib(ax, domain, color="k")
    for obs in obstacles:
        plot_hpoly_matplotlib(ax, obs, color="tab:red", zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)


def main() -> None:
    domain = make_box([0.0, 0.0], [10.0, 10.0])
    obstacles = [
        make_box([2.0, 2.0], [4.0, 4.0]),
        make_box([6.0, 1.0], [8.0, 6.0]),
        make_polygon(np.array([[1.0, 7.0], [3.0, 7.0], [2.0, 9.0]])),
    ]

    rng = np.random.default_rng(0)
    samples = sample_free_space(domain, obstacles, num_samples=1000, rng=rng)
    graph = build_visibility_graph(samples, obstacles)
    cliques = truncated_clique_cover(graph, min_clique_size=3)
    cliques_points = [samples[sorted(c)] for c in cliques]
    regions = inflate_regions_from_cliques(
        cliques_points, obstacles, domain, step_back_margin=0.05
    )
    print(
        f"samples={len(samples)}, edges={graph.number_of_edges()}, "
        f"cliques={len(cliques)}, regions={len(regions)}"
    )

    # ---- figure 1: inflated regions ----
    fig, ax = plt.subplots(figsize=(7, 7))
    setup_axes(ax, obstacles, domain)
    for r in regions:
        plot_hpoly_matplotlib(ax, r, color="tab:blue", zorder=2)
    ax.set_title(f"convexobstaclesvcc — {len(regions)} regions")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "example_2d.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: clique convex hulls + points ----
    fig, ax = plt.subplots(figsize=(7, 7))
    setup_axes(ax, obstacles, domain)
    cmap = plt.get_cmap("tab20")
    for k, pts in enumerate(cliques_points):
        color = cmap(k % cmap.N)
        hull = HPolyhedron(VPolytope(pts.T))
        plot_hpoly_matplotlib(ax, hull, color=color, zorder=2)
        ax.scatter(pts[:, 0], pts[:, 1], s=18, c=[color], edgecolor="k",
                   linewidths=0.5, zorder=3)
    ax.set_title(f"convexobstaclesvcc — {len(cliques_points)} cliques (conv hulls)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "example_2d_cliques.png", dpi=150)
    plt.close(fig)

    print(f"Saved figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()

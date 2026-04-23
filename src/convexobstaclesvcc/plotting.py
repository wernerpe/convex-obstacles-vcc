"""Minimal matplotlib plotting helpers for 2D HPolyhedra.

Adapted from ``cspace_utils.plotting`` — kept here to avoid pulling in the
full dependency just for two functions.
"""

from __future__ import annotations

import numpy as np
from pydrake.geometry.optimization import HPolyhedron, VPolytope


def _sorted_vertices(vpoly: VPolytope) -> np.ndarray:
    assert vpoly.ambient_dimension() == 2
    V = vpoly.vertices()
    center = V.mean(axis=1, keepdims=True)
    angles = np.arctan2(V[1] - center[1], V[0] - center[0])
    return V[:, np.argsort(angles)]


def plot_hpoly_matplotlib(ax, hpoly: HPolyhedron, color=None, zorder: int = 0):
    v = _sorted_vertices(VPolytope(hpoly)).T
    v = np.vstack([v, v[:1]])
    kw = dict(linewidth=2, alpha=0.7, zorder=zorder)
    p = ax.plot(v[:, 0], v[:, 1], **kw) if color is None else ax.plot(v[:, 0], v[:, 1], c=color, **kw)
    ax.fill(v[:, 0], v[:, 1], alpha=0.5, c=p[0].get_color(), zorder=zorder)


def plot_hpoly_skeleton_matplotlib(ax, hpoly: HPolyhedron, color=None, zorder: int = 0):
    v = _sorted_vertices(VPolytope(hpoly)).T
    v = np.vstack([v, v[:1]])
    kw = dict(linewidth=2, alpha=0.7, zorder=zorder)
    if color is None:
        ax.plot(v[:, 0], v[:, 1], **kw)
    else:
        ax.plot(v[:, 0], v[:, 1], c=color, **kw)

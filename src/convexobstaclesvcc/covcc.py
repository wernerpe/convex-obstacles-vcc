"""Top-level COVCC routine."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydrake.geometry.optimization import ConvexSet, HPolyhedron

from .clique_cover import truncated_clique_cover
from .inflation import inflate_regions_from_cliques
from .visibility_graph import build_visibility_graph, sample_free_space


def COVCC(
    obstacles: List[ConvexSet],
    domain: HPolyhedron,
    num_samples: int,
    step_back_margin: float,
    min_clique_size: int = 3,
    seed: Optional[int] = None,
) -> List[HPolyhedron]:
    """Convex-Obstacles Visibility Clique Cover.

    Samples ``num_samples`` points in the free space, builds a visibility graph
    under the convex obstacles, computes a truncated clique cover, and inflates
    one collision-free HPolyhedron per clique using the separating-halfplane
    QP. The union of the returned regions approximates the free space.
    """
    rng = np.random.default_rng(seed)

    samples = sample_free_space(domain, obstacles, num_samples, rng)
    graph = build_visibility_graph(samples, obstacles)
    cliques = truncated_clique_cover(graph, min_clique_size=min_clique_size)

    cliques_points = [samples[sorted(c)] for c in cliques]
    regions = inflate_regions_from_cliques(
        cliques_points, obstacles, domain, step_back_margin=step_back_margin
    )
    return regions

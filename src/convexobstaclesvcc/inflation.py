"""Closed-form separating-halfplane inflation from a clique of seed points.

For each convex obstacle we solve a QP that finds the closest pair of points
between the convex hull of the clique and the obstacle, then append the
induced separating halfplane to ``domain``. The clique's convex hull lies on
the near side of every such halfplane by construction, so the resulting region
contains every seed point in the clique.

Generalization of the segment-based version in
``franka_manipulation_station/.../scs_trajopt.py::construct_regions_from_obstacles``:
instead of parameterizing a line ``p = a + t d, t ∈ [0,1]``, we parameterize
``p = Σ_i λ_i q_i`` with ``λ`` on the simplex over the k clique points.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydrake.common import Parallelism
from pydrake.geometry.optimization import ConvexSet, HPolyhedron
from pydrake.solvers import MathematicalProgram, SolveInParallel

_DIST_EPS = 1e-6


def _build_min_distance_qp(clique_points: np.ndarray, obstacle: ConvexSet):
    k, dim = clique_points.shape

    prog = MathematicalProgram()
    p_obs = prog.NewContinuousVariables(dim, "p_obs")
    p_cliq = prog.NewContinuousVariables(dim, "p_cliq")
    lam = prog.NewContinuousVariables(k, "lam")

    # p_cliq = clique_points.T @ lam  →  [I | -Qᵀ] [p_cliq; lam] = 0
    A_eq = np.hstack([np.eye(dim), -clique_points.T])
    prog.AddLinearEqualityConstraint(
        A_eq, np.zeros(dim), np.concatenate([p_cliq, lam])
    )
    prog.AddLinearEqualityConstraint(np.ones((1, k)), np.ones(1), lam)
    prog.AddBoundingBoxConstraint(0.0, 1.0, lam)
    obstacle.AddPointInSetConstraints(prog, p_obs)

    # min ‖p_obs − p_cliq‖²   (Drake uses ½ xᵀQx so scale by 2).
    Q = np.zeros((2 * dim, 2 * dim))
    for i in range(dim):
        Q[i, i] = 2.0
        Q[dim + i, dim + i] = 2.0
        Q[i, dim + i] = -2.0
        Q[dim + i, i] = -2.0
    prog.AddQuadraticCost(
        Q, np.zeros(2 * dim), np.concatenate([p_obs, p_cliq]), is_convex=True
    )
    return prog, p_obs, p_cliq


def inflate_regions_from_cliques(
    cliques_points: List[np.ndarray],
    obstacles: List[ConvexSet],
    domain: HPolyhedron,
    step_back_margin: float = 0.0,
    verbose: bool = False,
) -> List[HPolyhedron]:
    """Inflate one HPolyhedron per clique; non-separable cliques are dropped.

    Parameters
    ----------
    cliques_points : list of (k_i, dim) arrays, each a collision-free clique.
    obstacles : convex obstacles to separate from.
    domain : ambient domain; halfplanes are appended to its H-representation.
    step_back_margin : desired offset pushed into the obstacle side. If it
        exceeds the closest-pair distance, the halfplane is relaxed to just
        contain the clique (a clique for which *any* obstacle sits inside
        ``conv(clique)`` is dropped entirely).

    Returns
    -------
    List of HPolyhedra (length ≤ ``len(cliques_points)``).
    """
    if not cliques_points:
        return []
    if not obstacles:
        return [HPolyhedron(domain.A().copy(), domain.b().copy()) for _ in cliques_points]

    progs: List[MathematicalProgram] = []
    pair_clique_idx: List[int] = []
    pair_obs_idx: List[int] = []
    p_obs_vars = []
    p_cliq_vars = []
    for ci, cp in enumerate(cliques_points):
        cp = np.asarray(cp, dtype=float)
        for oi, obs in enumerate(obstacles):
            prog, p_obs, p_cliq = _build_min_distance_qp(cp, obs)
            progs.append(prog)
            pair_clique_idx.append(ci)
            pair_obs_idx.append(oi)
            p_obs_vars.append(p_obs)
            p_cliq_vars.append(p_cliq)

    results = SolveInParallel(progs=progs, parallelism=Parallelism.Max())

    A_by_clique = [domain.A().copy() for _ in cliques_points]
    b_by_clique = [domain.b().copy() for _ in cliques_points]
    dropped = [False] * len(cliques_points)

    for res, ci, oi, p_obs, p_cliq in zip(
        results, pair_clique_idx, pair_obs_idx, p_obs_vars, p_cliq_vars
    ):
        if dropped[ci]:
            continue
        if not res.is_success():
            print(f"[COVCC] QP failed: clique {ci}, obstacle {oi}. Dropping clique.")
            dropped[ci] = True
            continue

        p_obs_v = res.GetSolution(p_obs)
        p_cliq_v = res.GetSolution(p_cliq)
        diff = p_obs_v - p_cliq_v
        dist = float(np.linalg.norm(diff))

        if dist <= _DIST_EPS:
            # conv(clique) touches or contains the obstacle — not separable.
            print(
                f"[COVCC] clique {ci} not separable from obstacle {oi} "
                f"(dist={dist:.2e}). Dropping clique."
            )
            dropped[ci] = True
            continue

        n = diff / dist
        relaxation = max(0.0, step_back_margin - 0.999 * dist)
        c = float(n @ p_obs_v) - step_back_margin + relaxation
        A_by_clique[ci] = np.vstack([A_by_clique[ci], n.reshape(1, -1)])
        b_by_clique[ci] = np.concatenate([b_by_clique[ci], [c]])

    return [
        HPolyhedron(A, b)
        for A, b, drop in zip(A_by_clique, b_by_clique, dropped)
        if not drop
    ]


def inflate_region_from_clique(
    clique_points: np.ndarray,
    obstacles: List[ConvexSet],
    domain: HPolyhedron,
    step_back_margin: float = 0.0,
    verbose: bool = False,
) -> Optional[HPolyhedron]:
    """Single-clique convenience wrapper around :func:`inflate_regions_from_cliques`."""
    out = inflate_regions_from_cliques(
        [np.asarray(clique_points, dtype=float)],
        obstacles,
        domain,
        step_back_margin=step_back_margin,
        verbose=verbose,
    )
    return out[0] if out else None

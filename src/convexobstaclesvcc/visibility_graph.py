"""Sampling and pairwise-visibility checks against convex obstacles.

Obstacles must be ``HPolyhedron`` so we can do an LP-free edge check by
intersecting per-row halfplane ranges along the segment.

The O(N² · rows) pairwise loop is JIT-compiled with numba and parallelized
over the outer index.
"""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np
from numba import njit, prange
from pydrake.common import RandomGenerator
from pydrake.geometry.optimization import HPolyhedron


def _stack_obstacles(obstacles: List[HPolyhedron]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack H-reps into CSR-style contiguous arrays for numba."""
    if not obstacles:
        dim = 0
        return (
            np.zeros((0, dim), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(1, dtype=np.int64),
        )
    dim = obstacles[0].A().shape[1]
    rows = [o.A().shape[0] for o in obstacles]
    A_stack = np.zeros((sum(rows), dim), dtype=np.float64)
    b_stack = np.zeros(sum(rows), dtype=np.float64)
    starts = np.zeros(len(obstacles) + 1, dtype=np.int64)
    off = 0
    for k, o in enumerate(obstacles):
        n = rows[k]
        A_stack[off : off + n] = o.A()
        b_stack[off : off + n] = o.b()
        starts[k + 1] = starts[k] + n
        off += n
    return A_stack, b_stack, starts


@njit(cache=True, fastmath=True)
def _point_in_any_obstacle(q, A_stack, b_stack, starts, tol):
    n_obs = starts.shape[0] - 1
    dim = q.shape[0]
    for k in range(n_obs):
        inside = True
        for r in range(starts[k], starts[k + 1]):
            s = 0.0
            for d in range(dim):
                s += A_stack[r, d] * q[d]
            if s > b_stack[r] + tol:
                inside = False
                break
        if inside:
            return True
    return False


@njit(cache=True, fastmath=True)
def _segment_hits_any_obstacle(p, q, A_stack, b_stack, starts, tol):
    n_obs = starts.shape[0] - 1
    dim = p.shape[0]
    for k in range(n_obs):
        lo = 0.0
        hi = 1.0
        feasible = True
        for r in range(starts[k], starts[k + 1]):
            alpha = 0.0
            ap = 0.0
            for d in range(dim):
                alpha += A_stack[r, d] * (q[d] - p[d])
                ap += A_stack[r, d] * p[d]
            beta = b_stack[r] - ap
            if alpha > tol:
                t = beta / alpha
                if t < hi:
                    hi = t
            elif alpha < -tol:
                t = beta / alpha
                if t > lo:
                    lo = t
            else:
                if beta < -tol:
                    feasible = False
                    break
            if lo > hi + tol:
                feasible = False
                break
        if feasible and lo <= hi + tol:
            return True
    return False


@njit(cache=True, fastmath=True, parallel=True)
def _pairwise_visibility(samples, A_stack, b_stack, starts, tol):
    n = samples.shape[0]
    adj = np.zeros((n, n), dtype=np.bool_)
    for i in prange(n):
        for j in range(i + 1, n):
            if not _segment_hits_any_obstacle(
                samples[i], samples[j], A_stack, b_stack, starts, tol
            ):
                adj[i, j] = True
                adj[j, i] = True
    return adj


def sample_free_space(
    domain: HPolyhedron,
    obstacles: List[HPolyhedron],
    num_samples: int,
    rng: np.random.Generator,
    max_attempts_per_sample: int = 100,
    mixing_steps: int = 10,
) -> np.ndarray:
    """Uniform free-space samples via Drake hit-and-run + obstacle rejection."""
    A_stack, b_stack, starts = _stack_obstacles(obstacles)
    drake_rng = RandomGenerator(int(rng.integers(0, 2**31 - 1)))
    current = domain.ChebyshevCenter()

    samples = []
    attempts = 0
    budget = num_samples * max_attempts_per_sample
    while len(samples) < num_samples and attempts < budget:
        attempts += 1
        current = domain.UniformSample(drake_rng, current, mixing_steps=mixing_steps)
        if _point_in_any_obstacle(current, A_stack, b_stack, starts, 0.0):
            continue
        samples.append(current.copy())

    if len(samples) < num_samples:
        print(
            f"[COVCC] sample_free_space: only found {len(samples)}/{num_samples} "
            f"free samples after {attempts} attempts."
        )
    return np.asarray(samples, dtype=np.float64)


def segment_is_collision_free(
    p: np.ndarray,
    q: np.ndarray,
    obstacles: List[HPolyhedron],
    tol: float = 1e-9,
) -> bool:
    A_stack, b_stack, starts = _stack_obstacles(obstacles)
    return not _segment_hits_any_obstacle(
        np.ascontiguousarray(p, dtype=np.float64),
        np.ascontiguousarray(q, dtype=np.float64),
        A_stack, b_stack, starts, tol,
    )


def build_visibility_graph(
    samples: np.ndarray,
    obstacles: List[HPolyhedron],
    tol: float = 1e-9,
) -> nx.Graph:
    """Pairwise visibility graph over ``samples``; O(N² · rows) numba kernel."""
    A_stack, b_stack, starts = _stack_obstacles(obstacles)
    samples = np.ascontiguousarray(samples, dtype=np.float64)
    adj = _pairwise_visibility(samples, A_stack, b_stack, starts, tol)
    G = nx.Graph()
    G.add_nodes_from(range(samples.shape[0]))
    ii, jj = np.where(np.triu(adj, k=1))
    G.add_edges_from(zip(ii.tolist(), jj.tolist()))
    return G

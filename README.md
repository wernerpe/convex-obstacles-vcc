# convexobstaclesvcc

Visibility Clique Cover (VCC) for approximating free space in Euclidean domains
populated with convex obstacles. Inspired by Werner et al., *Approximating Robot
Configuration Spaces with few Convex Sets using Clique Covers of Visibility Graphs*,
but specialized to the case where the obstacles are given explicitly as convex sets
(Drake `HPolyhedron` / `ConvexSet`). This lets us replace IRIS-NP inflation with a
closed-form separating-halfplane inflation against each obstacle.

## Install

```bash
uv sync
```

## Run the example

```bash
uv run python scripts/example_2d.py
```

Writes `tmp/example_2d.png` (obstacles + inflated regions) and
`tmp/example_2d_cliques.png` (obstacles + clique convex hulls).

## Usage

```python
from convexobstaclesvcc import COVCC

regions = COVCC(
    obstacles=obstacles,          # list[pydrake.geometry.optimization.HPolyhedron]
    domain=domain,                # pydrake.geometry.optimization.HPolyhedron
    num_samples=500,              # samples per visibility-graph round
    step_back_margin=0.05,        # separating-halfplane margin
)
```

## Set inflation

Given a clique `Q = {q_1, …, q_k}` of collision-free seed points and a set of
convex obstacles `O_1, …, O_m`, we want a convex region `R ⊇ Q` that avoids
every obstacle.

For each obstacle `O_j` we solve one quadratic program for the closest pair
between the clique's convex hull and the obstacle,

```
    min  ‖p_O − p_Q‖²
    s.t. p_O ∈ O_j
         p_Q = Σ_i λ_i q_i,  λ ≥ 0,  Σ λ_i = 1          (p_Q ∈ conv(Q))
```

All `m` QPs (one per obstacle) are convex and solved in parallel via
`pydrake.solvers.SolveInParallel`. From the optimizer `(p_O*, p_Q*)` we
extract

```
    d_j = ‖p_O* − p_Q*‖
    n_j = (p_O* − p_Q*) / d_j                           (unit normal)
    H_j = { x : n_jᵀ x  ≤  n_jᵀ p_O* − δ }              (separating halfplane,
                                                         pushed back by margin δ)
```

`H_j` separates `O_j` from `conv(Q)`. The returned region is

```
    R = domain ∩ H_1 ∩ … ∩ H_m
```

Because `conv(Q)` sits on the near side of every `H_j` (at distance `d_j` from
the supporting hyperplane), `R ⊇ conv(Q) ⊇ Q` by construction — every seed
lies in the region. If `δ > d_j` the halfplane would cut the clique off, so
we relax `δ` down to just inside the clique edge for that obstacle.

This is a direct generalization of the segment-based inflation in
`franka_manipulation_station/.../scs_trajopt.py::construct_regions_from_obstacles`:
a segment is the special case `k = 2`, `conv({a, b}) = {a + t(b−a) : t ∈ [0,1]}`.

## Non-separable cliques

Pairwise visibility does *not* imply that the convex hull of a clique is
collision-free — three mutually-visible points can wrap around an obstacle.
When that happens the closest-pair QP between `conv(clique)` and the
offending obstacle returns distance ≈ 0, so no separating halfplane exists.
The current implementation **detects this case, prints a warning naming the
clique and obstacle, and drops the clique entirely** (no region returned for
it). A denser visibility graph makes the situation rare; if it keeps
occurring the intended fix is to shrink/split the clique and retry.

## Layout

```
src/convexobstaclesvcc/
    covcc.py              # top-level COVCC(...)
    visibility_graph.py   # sample free space, check pairwise visibility
    clique_cover.py       # truncated greedy clique cover
    inflation.py          # closed-form halfplane inflation from a clique

scripts/                  # runnable examples
```

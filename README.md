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

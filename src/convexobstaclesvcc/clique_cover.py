"""Truncated greedy clique cover of a visibility graph."""

from __future__ import annotations

from typing import List, Set

import networkx as nx


def _greedy_max_clique(G: nx.Graph) -> Set[int]:
    """Greedy maximal clique: start from highest-degree vertex, extend by
    max-degree common neighbor."""
    if G.number_of_nodes() == 0:
        return set()
    start = max(G.degree, key=lambda x: x[1])[0]
    clique = {start}
    candidates = set(G.neighbors(start))
    while candidates:
        best, best_deg = -1, -1
        for v in candidates:
            deg = G.degree(v)
            if deg > best_deg:
                best, best_deg = v, deg
        clique.add(best)
        candidates &= set(G.neighbors(best))
    return clique


def truncated_clique_cover(
    graph: nx.Graph,
    min_clique_size: int = 3,
) -> List[Set[int]]:
    """Repeatedly pull a (greedy) maximum clique; stop when it falls below
    ``min_clique_size``. Returned cliques are disjoint (each vertex covered
    at most once)."""
    G = graph.copy()
    cliques: List[Set[int]] = []
    while True:
        c = _greedy_max_clique(G)
        if len(c) < min_clique_size:
            break
        cliques.append(c)
        G.remove_nodes_from(c)
        if G.number_of_nodes() == 0:
            break
    return cliques

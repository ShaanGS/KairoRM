"""PageRank-based importance scoring over the call graph.

Edges flow caller → callee, so heavily-called utilities and public entry points
accumulate weight. Cycles (mutual recursion, circular imports) are detected and
the lowest-weight back-edge of each cycle is removed before scoring, so PageRank
sees a tractable graph and the cycle is logged for the architecture agent later.
"""

from __future__ import annotations

import logging

import networkx as nx

from ingestion.types import Chunk, RankedChunk, RankResult

logger = logging.getLogger(__name__)


def _build_graph(chunks: list[Chunk]) -> nx.DiGraph:
    g = nx.DiGraph()
    # Index chunks by callable name (functions/methods only — classes still get
    # added as nodes but receive no incoming call edges by name).
    name_to_ids: dict[str, list[str]] = {}
    for ch in chunks:
        g.add_node(ch.chunk_id)
        if ch.unit_type in ("function", "method"):
            name_to_ids.setdefault(ch.name, []).append(ch.chunk_id)

    for caller in chunks:
        for callee_name in caller.calls:
            targets = name_to_ids.get(callee_name, [])
            for target_id in targets:
                if target_id == caller.chunk_id:
                    continue  # ignore direct self-loops
                if g.has_edge(caller.chunk_id, target_id):
                    g[caller.chunk_id][target_id]["weight"] += 1.0
                else:
                    g.add_edge(caller.chunk_id, target_id, weight=1.0)
    return g


def _break_cycles(g: nx.DiGraph) -> list[tuple[str, str]]:
    """Remove the lowest-weight edge from each simple cycle.

    Returns the broken back-edges as `(caller_chunk_id, callee_chunk_id)` pairs so
    downstream agents can surface them as circular-dependency risks.
    """
    removed: list[tuple[str, str]] = []
    # Iterate until acyclic; each pass kills one edge per cycle found.
    while True:
        try:
            cycle = next(nx.simple_cycles(g))
        except StopIteration:
            break
        # `cycle` is a list of nodes; build edge list and drop the weakest one.
        edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
        weakest = min(edges, key=lambda e: g[e[0]][e[1]].get("weight", 1.0))
        logger.info("ranker: breaking cycle of len=%d at edge %s", len(cycle), weakest)
        g.remove_edge(*weakest)
        removed.append((weakest[0], weakest[1]))
    return removed


def rank(chunks: list[Chunk]) -> RankResult:
    """Score chunks by PageRank over the call graph.

    Chunks with no incoming or outgoing call edges still receive the uniform
    baseline score from PageRank's damping term — never zero, never raise. The
    returned `RankResult` also carries any back-edges broken to remove cycles,
    expressed as `(file_path, file_path)` so downstream agents can report readable
    circular-dependency risks instead of opaque chunk IDs.
    """
    if not chunks:
        return RankResult(chunks=[], cycles=[])

    # Map chunk ids → file paths so broken edges can be reported by file, not id.
    id_to_path = {ch.chunk_id: str(ch.file_path) for ch in chunks}

    g = _build_graph(chunks)
    call_edges = list(g.edges())  # capture before _break_cycles mutates the graph
    broken = _break_cycles(g)
    cycles = [(id_to_path.get(a, a), id_to_path.get(b, b)) for a, b in broken]

    if g.number_of_edges() == 0:
        # No call edges — every chunk gets the uniform baseline so downstream
        # consumers can still sort/rank without special-casing this.
        uniform = 1.0 / len(chunks)
        ranked = [RankedChunk(chunk=ch, importance_score=uniform) for ch in chunks]
        return RankResult(chunks=ranked, cycles=cycles, call_edges=call_edges)

    try:
        scores = nx.pagerank(g, alpha=0.85, weight="weight")
    except nx.PowerIterationFailedConvergence:
        scores = nx.pagerank(g, alpha=0.85, weight="weight", max_iter=1000, tol=1e-4)

    fallback = 1.0 / len(chunks)
    ranked = [
        RankedChunk(chunk=ch, importance_score=scores.get(ch.chunk_id, fallback)) for ch in chunks
    ]
    return RankResult(chunks=ranked, cycles=cycles, call_edges=call_edges)

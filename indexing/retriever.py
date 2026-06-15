"""Hybrid retrieval: semantic + keyword, fused and PageRank-boosted.

Two independent signals are computed over the stored chunks — cosine similarity
against the query embedding, and BM25 keyword overlap — then merged with Reciprocal
Rank Fusion. The fused score is finally multiplied by each chunk's PageRank
`importance_score`, so structurally central code (entry points, hot utilities) is
promoted even when a naive RAG ranker would bury it. This boost is the core reason
KairoRM's retrieval beats a plain vector search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from indexing import embeddings as embeddings_mod
from indexing import vectorstore
from ingestion.types import EmbeddedChunk, Err, Ok, RankedChunk, Result

RRF_K = 60

_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class RetrieveError:
    reason: str


def _tokenize(text: str) -> list[str]:
    # Split camelCase, then snake_case/punctuation falls out of the alnum tokenizer.
    spaced = _CAMEL.sub(r"\1 \2", text)
    return _TOKEN.findall(spaced.lower())


def _semantic_order(query_vec: list[float], chunks: list[EmbeddedChunk]) -> list[str]:
    matrix = np.array([ec.embedding for ec in chunks], dtype=float)
    q = np.array(query_vec, dtype=float)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    sims = (matrix / row_norms) @ q_norm
    order = np.argsort(-sims)
    return [chunks[i].ranked.chunk.chunk_id for i in order]


def _bm25_order(query: str, chunks: list[EmbeddedChunk]) -> list[str]:
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(ec.ranked.chunk.content) for ec in chunks]
    # Guard against an all-empty corpus, which BM25Okapi cannot build from.
    if not any(corpus):
        return [ec.ranked.chunk.chunk_id for ec in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    order = np.argsort(-np.asarray(scores))
    return [chunks[i].ranked.chunk.chunk_id for i in order]


def _fuse(
    semantic_order: list[str],
    bm25_order: list[str],
    chunks: list[EmbeddedChunk],
) -> dict[str, float]:
    sem_rank = {cid: i + 1 for i, cid in enumerate(semantic_order)}
    bm_rank = {cid: i + 1 for i, cid in enumerate(bm25_order)}
    importance = {ec.ranked.chunk.chunk_id: ec.ranked.importance_score for ec in chunks}

    scores: dict[str, float] = {}
    for cid in importance:
        rrf = 1.0 / (RRF_K + sem_rank[cid]) + 1.0 / (RRF_K + bm_rank[cid])
        # PageRank boost: central chunks win ties and overcome small RRF deficits.
        scores[cid] = rrf * importance[cid]
    return scores


async def retrieve(
    query: str, *, repo_id: str, db_path: Path, k: int = 15
) -> Result[list[RankedChunk], RetrieveError]:
    """Return the top-`k` chunks for `query` via hybrid, PageRank-boosted retrieval."""
    load_result = await vectorstore.load(repo_id, db_path)
    if not load_result.is_ok():
        return Err(RetrieveError(reason=f"load failed: {load_result.error.reason}"))

    chunks = load_result.unwrap()
    if not chunks:
        return Ok([])

    query_result = await embeddings_mod.embed_texts([query])
    if not query_result.is_ok():
        return Err(RetrieveError(reason=f"query embedding failed: {query_result.error.reason}"))
    query_vec = query_result.unwrap()[0]

    semantic_order = _semantic_order(query_vec, chunks)
    bm25_order = _bm25_order(query, chunks)
    fused = _fuse(semantic_order, bm25_order, chunks)

    by_id = {ec.ranked.chunk.chunk_id: ec.ranked for ec in chunks}
    ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return Ok([by_id[cid] for cid in ranked_ids[:k]])

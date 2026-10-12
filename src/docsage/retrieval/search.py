"""Searching a collection by BM25, dense vectors, or both fused."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from docsage.retrieval.bm25 import bm25_search
from docsage.retrieval.dense import DenseIndex, Embedder, embed_missing
from docsage.retrieval.fusion import DEFAULT_K, reciprocal_rank_fusion
from docsage.store import Collection, Store, StoredChunk

#: Each retriever contributes at least this many candidates to a fusion.
MIN_CANDIDATES = 50


class Mode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class Hit:
    """A chunk found by a search.

    ``score`` is the BM25 score, the cosine similarity or the fused score,
    depending on the mode, so only compare scores within one search. The ranks
    are the chunk's 1-based positions in each retriever's own list, and are
    ``None`` where that retriever did not return it. A reranker replaces the
    score with its own and records the position the search had given the hit
    as ``first_rank``.
    """

    chunk: StoredChunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    first_rank: int | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: list[Hit]
    embedded: int
    """Chunks that were embedded for this search because they had no vector yet."""


def search(
    store: Store,
    collection: Collection,
    query: str,
    *,
    mode: Mode = Mode.HYBRID,
    k: int = 10,
    embedder: Embedder | None = None,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    rrf_k: int = DEFAULT_K,
) -> SearchResult:
    """The ``k`` best chunks of ``collection`` for ``query``.

    Dense and hybrid searches need an ``embedder`` and first embed any chunks
    that lack a vector from it. A hybrid search takes the top candidates of
    both retrievers and fuses their ranks, weighting each list; a weight of 0
    leaves that retriever out.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, not {k}")
    use_bm25 = mode is Mode.BM25 or (mode is Mode.HYBRID and bm25_weight > 0)
    use_dense = mode is Mode.DENSE or (mode is Mode.HYBRID and dense_weight > 0)
    if use_dense and embedder is None:
        raise ValueError(f"{mode.value} search needs an embedder")
    depth = k if mode is not Mode.HYBRID else max(MIN_CANDIDATES, 4 * k)

    embedded = 0
    bm25_hits: list[tuple[int, float]] = []
    dense_hits: list[tuple[int, float]] = []
    if use_bm25:
        bm25_hits = bm25_search(store, collection, query, depth)
    if use_dense:
        assert embedder is not None
        embedded = embed_missing(store, collection, embedder)
        index = DenseIndex.load(store, collection, embedder.name)
        if query.strip():
            dense_hits = index.search(embedder.embed_query(query), depth)

    bm25_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_hits, start=1)}
    dense_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense_hits, start=1)}
    if mode is Mode.BM25:
        ranked = bm25_hits[:k]
    elif mode is Mode.DENSE:
        ranked = dense_hits[:k]
    else:
        fused = reciprocal_rank_fusion(
            [[i for i, _ in bm25_hits], [i for i, _ in dense_hits]],
            [bm25_weight, dense_weight],
            k=rrf_k,
        )
        ranked = fused[:k]
    chunks = store.chunks_by_id([chunk_id for chunk_id, _ in ranked])
    hits = [
        Hit(chunks[chunk_id], score, bm25_ranks.get(chunk_id), dense_ranks.get(chunk_id))
        for chunk_id, score in ranked
    ]
    return SearchResult(hits, embedded)

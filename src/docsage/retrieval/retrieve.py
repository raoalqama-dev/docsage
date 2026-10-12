"""The retrieval pipeline: search, rerank, filter by score, diversify."""

from __future__ import annotations

from dataclasses import dataclass

from docsage.retrieval.dense import DenseIndex, Embedder, embed_missing
from docsage.retrieval.fusion import DEFAULT_K
from docsage.retrieval.rerank import Reranker, mmr, rerank
from docsage.retrieval.search import Hit, Mode, search
from docsage.store import Collection, Store

#: Candidates reranked per hit wanted, and the fewest candidates reranked.
CANDIDATE_FACTOR = 3
MIN_RERANKED = 30


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    hits: list[Hit]
    embedded: int
    """Chunks that were embedded for this retrieval because they had no vector yet."""
    candidates: int
    """How many hits the first-stage search handed to the later stages."""
    below_threshold: int
    """How many candidates the minimum score removed."""


def retrieve(
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
    reranker: Reranker | None = None,
    candidates: int | None = None,
    min_score: float | None = None,
    mmr_lambda: float | None = None,
) -> RetrievalResult:
    """The ``k`` best chunks of ``collection`` for ``query``, after the optional later stages.

    The search fetches ``candidates`` hits (by default three times ``k``, and
    at least 30, when a reranker, a minimum score or MMR will narrow them
    down; otherwise ``k``). Then:

    1. a ``reranker`` re-scores them and re-orders them by its own score;
    2. hits scoring below ``min_score`` are dropped. The score is the
       reranker's when there is one, otherwise the search's, whose scale
       depends on the mode;
    3. ``mmr_lambda`` (0 to 1) picks ``k`` of the rest by maximal marginal
       relevance, comparing chunks by their vectors from ``embedder``, which
       is therefore required. Without it the best ``k`` are kept.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, not {k}")
    if candidates is not None and candidates < k:
        raise ValueError(f"candidates ({candidates}) must be at least k ({k})")
    if mmr_lambda is not None and embedder is None:
        raise ValueError("MMR needs an embedder")
    staged = reranker is not None or min_score is not None or mmr_lambda is not None
    if candidates is None:
        candidates = max(CANDIDATE_FACTOR * k, MIN_RERANKED) if staged else k

    found = search(
        store,
        collection,
        query,
        mode=mode,
        k=candidates,
        embedder=embedder,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        rrf_k=rrf_k,
    )
    hits = found.hits
    embedded = found.embedded
    if reranker is not None:
        hits = rerank(reranker, query, hits)
    if min_score is not None:
        kept = [hit for hit in hits if hit.score >= min_score]
        below, hits = len(hits) - len(kept), kept
    else:
        below = 0

    if mmr_lambda is not None and hits:
        assert embedder is not None
        embedded += embed_missing(store, collection, embedder)
        index = DenseIndex.load(store, collection, embedder.name)
        vectors = index.vectors_of([hit.chunk.id for hit in hits])
        picked = mmr([hit.score for hit in hits], vectors, k, mmr_lambda)
        hits = [hits[i] for i in picked]
    return RetrievalResult(hits[:k], embedded, len(found.hits), below)

"""Reranking the candidates of a search, and picking a diverse set from them.

First-stage retrieval is cheap and approximate. A reranker reads the query and
each candidate together, which is slower and far more accurate, so it is run on
a few dozen candidates only. MMR then trades some relevance for variety, so the
results are not several copies of one passage.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Protocol

import numpy as np

from docsage.retrieval.dense import Vectors
from docsage.retrieval.search import Hit

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_MMR_LAMBDA = 0.7


class RerankerError(Exception):
    """A reranker that cannot be created or used."""


class Reranker(Protocol):
    """Scores passages against a query, reading both together.

    ``name`` identifies the model. Scores are only comparable within one call,
    and a higher score is a better match; their scale depends on the model.
    """

    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """One score per passage, in order."""
        ...


class CrossEncoderReranker:
    """A local ONNX cross-encoder through fastembed (the ``rerank`` extra).

    The model is downloaded on first use. ``model`` can be supplied instead,
    anything with fastembed's ``rerank(query, documents)`` method.
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, *, model: Any = None) -> None:
        self._model_name = model_name
        self._model = model

    @property
    def name(self) -> str:
        return f"cross-encoder-{self._model_name}"

    def _load(self) -> Any:
        if self._model is None:
            try:
                module = importlib.import_module("fastembed.rerank.cross_encoder")
            except ImportError:
                raise RerankerError(
                    "the cross-encoder reranker needs the rerank extra: "
                    "pip install 'docsage[rerank]'"
                ) from None
            try:
                self._model = module.TextCrossEncoder(model_name=self._model_name)
            except Exception as exc:  # fastembed raises ValueError, OSError and others
                raise RerankerError(f"cannot load {self._model_name}: {exc}") from exc
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return [float(score) for score in self._load().rerank(query, list(passages))]


def create_reranker(model: str = DEFAULT_RERANK_MODEL) -> Reranker:
    """The cross-encoder reranker with the given fastembed ``model``."""
    return CrossEncoderReranker(model)


def passage_text(hit: Hit) -> str:
    """What a reranker reads of a hit: the heading path, then the text.

    The headings say what the chunk is about when the text alone does not.
    """
    chunk = hit.chunk.chunk
    return "\n".join((*chunk.heading_path, chunk.text))


def rerank(reranker: Reranker, query: str, hits: Sequence[Hit]) -> list[Hit]:
    """``hits`` ordered by the reranker's score, which replaces their own.

    Each result remembers its position before reranking in ``first_rank``.
    Hits the reranker scores equally keep their first-stage order.
    """
    if not hits:
        return []
    scores = reranker.score(query, [passage_text(hit) for hit in hits])
    if len(scores) != len(hits):
        raise RerankerError(f"{reranker.name} returned {len(scores)} scores for {len(hits)} hits")
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    return [replace(hits[i], score=scores[i], first_rank=i + 1) for i in order]


def mmr(
    relevance: Sequence[float],
    vectors: Vectors,
    k: int,
    lambda_: float = DEFAULT_MMR_LAMBDA,
) -> list[int]:
    """Pick ``k`` items by maximal marginal relevance; returns their indices, in pick order.

    Each pick maximises ``lambda_ * relevance - (1 - lambda_) * redundancy``,
    where redundancy is the item's highest cosine similarity to an item
    already picked (not below 0); the first pick is simply the most relevant.
    Relevance is rescaled to 0..1 across the candidates first, so it is
    comparable with a similarity whatever the scale of the scores. ``vectors``
    has one unit-length row per item. A ``lambda_`` of 1 is plain ranking by
    relevance; lower values favour variety. Ties go to the item that comes
    first.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda must be between 0 and 1, not {lambda_}")
    if k < 1:
        raise ValueError(f"k must be at least 1, not {k}")
    if vectors.shape[0] != len(relevance):
        raise ValueError(f"{vectors.shape[0]} vectors for {len(relevance)} items")
    count = len(relevance)
    scores = np.asarray(relevance, dtype=np.float64)
    if count == 0:
        return []
    low, high = min(relevance), max(relevance)
    scaled = (scores - low) / (high - low) if high > low else np.ones(count)
    similarity = np.clip(vectors @ vectors.T, 0.0, None)

    picked: list[int] = []
    redundancy = np.zeros(count)
    remaining = np.ones(count, dtype=bool)
    for _ in range(min(k, count)):
        # The first pick has nothing to be redundant with, even when lambda is 0.
        value = lambda_ * scaled - (1.0 - lambda_) * redundancy if picked else scaled.copy()
        value[~remaining] = -np.inf
        best = int(np.argmax(value))
        picked.append(best)
        remaining[best] = False
        redundancy = np.maximum(redundancy, similarity[best])
    return picked

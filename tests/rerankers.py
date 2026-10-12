"""Fake rerankers for tests: small, deterministic and offline."""

from __future__ import annotations

import re
from collections.abc import Sequence

_WORD_RE = re.compile(r"[^\W_]+")


class OverlapReranker:
    """Scores a passage by how many different query words it contains."""

    name = "overlap-test"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        wanted = set(_WORD_RE.findall(query.lower()))
        return [float(len(wanted & set(_WORD_RE.findall(p.lower())))) for p in passages]


class FixedReranker:
    """Returns the scores it was given, whatever the passages, to test the plumbing."""

    name = "fixed-test"

    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = list(scores)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return list(self._scores)


class FakeCrossEncoderModel:
    """Stands in for ``fastembed``'s ``TextCrossEncoder``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, documents))
        return [float(len(document)) for document in documents]

"""Fake embedders for tests: small, deterministic and offline."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from docsage.retrieval.dense import Vectors, normalize

#: Words that mean the same thing share a concept. A real embedder learns
#: this; here it is a table, which is how the tests get paraphrases.
CONCEPTS: dict[str, tuple[str, ...]] = {
    "vehicle": ("car", "cars", "automobile", "automobiles", "vehicle", "vehicles"),
    "purchase": ("buy", "buying", "purchase", "purchasing", "sells", "sell", "dealership"),
    "feline": ("cat", "cats", "kitten", "kittens", "feline", "felines"),
    "sleep": ("sleep", "sleeps", "nap", "naps", "slumber", "slumbers", "rest"),
    "money": ("money", "cash", "funds", "payment", "payments"),
}

_WORD_RE = re.compile(r"[^\W_]+")


class ConceptEmbedder:
    """One dimension per concept; a text's vector counts the concept words in it.

    Words outside the table are ignored, so a text without any concept word
    embeds to the zero vector and is similar to nothing.
    """

    name = "concepts-test"

    def __init__(self) -> None:
        self._index = {
            word: number for number, words in enumerate(CONCEPTS.values()) for word in words
        }
        self.document_calls = 0
        self.query_calls = 0

    @property
    def dimension(self) -> int:
        return len(CONCEPTS)

    def _vector(self, text: str) -> Vectors:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in _WORD_RE.findall(text.lower()):
            if word in self._index:
                vector[self._index[word]] += 1.0
        return vector

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.document_calls += 1
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return normalize(np.stack([self._vector(text) for text in texts]))

    def embed_query(self, text: str) -> Vectors:
        self.query_calls += 1
        vector: Vectors = normalize(self._vector(text))[0]
        return vector


class FakeFastEmbedModel:
    """Stands in for ``fastembed.TextEmbedding``, with different query and passage vectors."""

    def __init__(self) -> None:
        self.passages: list[list[str]] = []

    def passage_embed(self, texts: list[str]) -> list[Vectors]:
        self.passages.append(texts)
        return [np.array([3.0, 4.0, 0.0], dtype=np.float32) for _ in texts]

    def query_embed(self, texts: list[str]) -> list[Vectors]:
        return [np.array([0.0, 0.0, 2.0], dtype=np.float32) for _ in texts]

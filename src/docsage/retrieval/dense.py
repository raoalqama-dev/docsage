"""Dense retrieval: embedders and a cosine-similarity index over stored vectors.

Vectors are L2-normalised float32, so cosine similarity is a dot product and
a collection's vectors are searched as one numpy matrix. That is exact, and
fast enough for collections up to about 100k chunks.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from docsage.store import Collection, EmbeddingRow, Store

Vectors = NDArray[np.float32]

DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_HASHING_DIMENSION = 512
EMBED_BATCH_SIZE = 64


class EmbedderError(Exception):
    """An embedder that cannot be created or used."""


class Embedder(Protocol):
    """Turns text into L2-normalised float32 vectors of one fixed dimension.

    Queries and documents are separate calls because some models embed them
    differently. ``name`` identifies the model: vectors are stored under it
    and only compared with vectors of the same name.
    """

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """One row per text."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """A single vector."""
        ...


def normalize(vectors: NDArray[Any]) -> Vectors:
    """Scale each row to unit length; all-zero rows stay zero."""
    matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    scaled: Vectors = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return scaled


_WORD_RE = re.compile(r"[^\W_]+")


class HashingEmbedder:
    """A deterministic embedder with no model: hashed words and character trigrams.

    Each word and each trigram of the word (with its boundaries marked) adds
    +1 or -1 to one of ``dimension`` buckets chosen by a hash, which is the
    hashing trick. Texts sharing words or word fragments, such as "replicate"
    and "replication", get similar vectors. It captures no meaning beyond
    that, so it will not match paraphrases; it exists to make dense search
    work offline, and for tests.
    """

    def __init__(self, dimension: int = DEFAULT_HASHING_DIMENSION) -> None:
        if dimension < 8:
            raise ValueError(f"dimension must be at least 8, not {dimension}")
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"hashing-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _features(self, text: str) -> Iterable[tuple[str, float]]:
        for word in _WORD_RE.findall(text.lower()):
            yield word, 1.0
            padded = f"<{word}>"
            for start in range(len(padded) - 2):
                yield padded[start : start + 3], 0.5

    def _vector(self, text: str) -> NDArray[np.float32]:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:6], "little") % self._dimension
            vector[bucket] += weight if digest[7] & 1 else -weight
        return vector

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return normalize(np.stack([self._vector(text) for text in texts]))

    def embed_query(self, text: str) -> Vectors:
        vector: Vectors = self.embed_documents([text])[0]
        return vector


class FastEmbedEmbedder:
    """Local ONNX embedding models through fastembed (the ``embed`` extra).

    The model is downloaded on first use. ``model`` can be supplied instead,
    anything with fastembed's ``passage_embed`` and ``query_embed`` methods.
    """

    def __init__(self, model_name: str = DEFAULT_FASTEMBED_MODEL, *, model: Any = None) -> None:
        self._model_name = model_name
        self._model = model
        self._dimension: int | None = None

    @property
    def name(self) -> str:
        return f"fastembed-{self._model_name}"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.embed_query("dimension").shape[0])
        return self._dimension

    def _load(self) -> Any:
        if self._model is None:
            try:
                module = importlib.import_module("fastembed")
            except ImportError:
                raise EmbedderError(
                    "the fastembed embedder needs the embed extra: pip install 'docsage[embed]'"
                ) from None
            try:
                self._model = module.TextEmbedding(model_name=self._model_name)
            except Exception as exc:  # fastembed raises ValueError, OSError and others
                raise EmbedderError(f"cannot load {self._model_name}: {exc}") from exc
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return normalize(np.stack(list(self._load().passage_embed(list(texts)))))

    def embed_query(self, text: str) -> Vectors:
        vector: Vectors = normalize(np.stack(list(self._load().query_embed([text]))))[0]
        return vector


def create_embedder(kind: str, model: str = DEFAULT_FASTEMBED_MODEL) -> Embedder:
    """The embedder called ``kind``: ``hashing``, or ``fastembed`` with the given ``model``."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder(model)
    raise EmbedderError(f"unknown embedder {kind!r}: use 'hashing' or 'fastembed'")


def encode_vector(vector: NDArray[np.float32]) -> bytes:
    """A vector as little-endian float32 bytes, the stored form."""
    return np.asarray(vector, dtype="<f4").tobytes()


def decode_vectors(rows: Sequence[EmbeddingRow], dimension: int) -> Vectors:
    """The stored vectors as a matrix with one row per entry in ``rows``."""
    if not rows:
        return np.zeros((0, dimension), dtype=np.float32)
    matrix = np.frombuffer(b"".join(row.vector for row in rows), dtype="<f4")
    return matrix.reshape(len(rows), dimension).astype(np.float32, copy=False)


def embed_missing(
    store: Store,
    collection: Collection,
    embedder: Embedder,
    *,
    batch_size: int = EMBED_BATCH_SIZE,
) -> int:
    """Embed the chunks of ``collection`` that have no vector from ``embedder`` yet.

    Returns how many were embedded. Batches are stored as they complete, so an
    interrupted run keeps its progress.
    """
    pending = store.unembedded_chunks(collection, embedder.name)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed_documents([text for _, text in batch])
        if vectors.shape != (len(batch), embedder.dimension):
            raise EmbedderError(
                f"{embedder.name} returned vectors of shape {vectors.shape}, "
                f"expected {(len(batch), embedder.dimension)}"
            )
        store.put_embeddings(
            embedder.name,
            embedder.dimension,
            [
                EmbeddingRow(chunk_id, encode_vector(vector))
                for (chunk_id, _), vector in zip(batch, vectors, strict=True)
            ],
        )
    return len(pending)


@dataclass(frozen=True, slots=True)
class DenseIndex:
    """The normalised vectors of a collection's chunks, searched by dot product."""

    ids: NDArray[np.int64]
    matrix: Vectors

    @classmethod
    def load(cls, store: Store, collection: Collection, model: str) -> DenseIndex:
        dimension, rows = store.embeddings(collection, model)
        ids = np.fromiter((row.chunk_id for row in rows), dtype=np.int64, count=len(rows))
        return cls(ids, decode_vectors(rows, dimension))

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: Vectors, limit: int) -> list[tuple[int, float]]:
        """Chunk ids with cosine similarity to ``query``, best first, ties by id.

        Chunks with no similarity (zero or negative) are not results.
        """
        if len(self) == 0 or limit < 1:
            return []
        if query.shape != (self.matrix.shape[1],):
            raise EmbedderError(
                f"the query has {query.shape[0]} dimensions but the stored vectors have "
                f"{self.matrix.shape[1]}"
            )
        scores = self.matrix @ query
        order = np.lexsort((self.ids, -scores))[:limit]
        return [(int(self.ids[i]), float(scores[i])) for i in order if scores[i] > 0]

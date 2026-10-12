from __future__ import annotations

import inspect
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from docsage.models import Chunk
from docsage.retrieval.dense import (
    DEFAULT_FASTEMBED_MODEL,
    DenseIndex,
    EmbedderError,
    FastEmbedEmbedder,
    HashingEmbedder,
    Vectors,
    create_embedder,
    decode_vectors,
    embed_missing,
    encode_vector,
    normalize,
)
from docsage.store import Collection, EmbeddingRow, Store
from embedders import ConceptEmbedder, FakeFastEmbedModel


def chunk(ordinal: int, text: str) -> Chunk:
    return Chunk(ordinal, text, (), None, None, len(text.split()))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "test.db") as opened:
        yield opened


def put(store: Store, collection: Collection, source: str, *texts: str) -> list[int]:
    record = store.put_document(
        collection,
        source=source,
        title=source,
        format="text",
        content_hash=source,
        pipeline="1",
        chunks=[chunk(n, text) for n, text in enumerate(texts)],
    )
    return [stored.id for stored in store.chunks(record.id)]


def cosine(a: Vectors, b: Vectors) -> float:
    return float(a @ b)


class TestNormalize:
    def test_rows_get_unit_length(self) -> None:
        result = normalize(np.array([[3.0, 4.0], [0.0, 2.0]]))
        assert result.dtype == np.float32
        assert result.tolist() == [[pytest.approx(0.6), pytest.approx(0.8)], [0.0, 1.0]]

    def test_zero_rows_stay_zero(self) -> None:
        assert normalize(np.zeros((2, 3))).tolist() == [[0.0] * 3] * 2

    def test_a_single_vector_becomes_one_row(self) -> None:
        assert normalize(np.array([0.0, 5.0])).shape == (1, 2)


class TestHashingEmbedder:
    def test_vectors_are_deterministic_unit_float32(self) -> None:
        embedder = HashingEmbedder()
        first = embedder.embed_documents(["Raft elects a leader", "Paxos"])
        second = HashingEmbedder().embed_documents(["Raft elects a leader", "Paxos"])
        assert first.shape == (2, 512)
        assert first.dtype == np.float32
        assert np.array_equal(first, second)
        assert np.linalg.norm(first, axis=1) == pytest.approx([1.0, 1.0])

    def test_the_vector_for_a_text_does_not_depend_on_the_batch(self) -> None:
        embedder = HashingEmbedder()
        alone = embedder.embed_documents(["Paxos"])[0]
        batched = embedder.embed_documents(["Raft", "Paxos", "Viewstamped"])[1]
        assert np.array_equal(alone, batched)

    def test_query_and_document_embeddings_agree(self) -> None:
        embedder = HashingEmbedder()
        assert np.array_equal(
            embedder.embed_query("leader"), embedder.embed_documents(["leader"])[0]
        )

    def test_shared_words_make_texts_similar(self) -> None:
        embedder = HashingEmbedder()
        base = embedder.embed_query("leader election")
        assert cosine(base, embedder.embed_query("election of the leader")) > 0.7
        assert cosine(base, embedder.embed_query("whisk eggs flour")) == 0.0

    def test_shared_word_fragments_make_texts_similar(self) -> None:
        embedder = HashingEmbedder()
        word = embedder.embed_query("replicate")
        assert cosine(word, embedder.embed_query("replication")) > 0.4
        assert cosine(word, embedder.embed_query("whisk eggs")) == 0.0

    def test_case_and_punctuation_do_not_matter(self) -> None:
        embedder = HashingEmbedder()
        assert np.array_equal(
            embedder.embed_query("Leader, ELECTION!"), embedder.embed_query("leader election")
        )

    def test_text_without_words_is_the_zero_vector(self) -> None:
        assert not HashingEmbedder().embed_query("!!! ---").any()

    def test_no_texts(self) -> None:
        assert HashingEmbedder(64).embed_documents([]).shape == (0, 64)

    def test_name_and_dimension(self) -> None:
        embedder = HashingEmbedder(64)
        assert (embedder.name, embedder.dimension) == ("hashing-64", 64)

    def test_dimension_must_be_at_least_8(self) -> None:
        with pytest.raises(ValueError, match="at least 8"):
            HashingEmbedder(4)


class TestFastEmbedEmbedder:
    def test_vectors_are_normalised_and_queries_use_the_query_model(self) -> None:
        model = FakeFastEmbedModel()
        embedder = FastEmbedEmbedder("some/model", model=model)
        documents = embedder.embed_documents(["a", "b"])
        assert documents.shape == (2, 3)
        assert documents[0].tolist() == [pytest.approx(0.6), pytest.approx(0.8), 0.0]
        assert embedder.embed_query("q").tolist() == [0.0, 0.0, 1.0]
        assert model.passages == [["a", "b"]]

    def test_name_and_dimension(self) -> None:
        embedder = FastEmbedEmbedder("some/model", model=FakeFastEmbedModel())
        assert embedder.name == "fastembed-some/model"
        assert embedder.dimension == 3
        assert embedder.dimension == 3

    def test_no_texts(self) -> None:
        embedder = FastEmbedEmbedder(model=FakeFastEmbedModel())
        assert embedder.embed_documents([]).shape == (0, 3)

    def test_the_model_is_created_on_first_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created: list[str] = []

        class TextEmbedding(FakeFastEmbedModel):
            def __init__(self, model_name: str) -> None:
                super().__init__()
                created.append(model_name)

        module = types.ModuleType("fastembed")
        module.TextEmbedding = TextEmbedding  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fastembed", module)
        embedder = FastEmbedEmbedder("org/small")
        assert created == []
        embedder.embed_query("x")
        embedder.embed_documents(["y"])
        assert created == ["org/small"]

    def test_a_missing_extra_is_explained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fastembed", None)
        with pytest.raises(EmbedderError, match=r"pip install 'docsage\[embed\]'"):
            FastEmbedEmbedder().embed_query("x")

    def test_a_model_that_cannot_load_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(model_name: str) -> None:
            raise ValueError(f"model {model_name} is not supported")

        module = types.ModuleType("fastembed")
        module.TextEmbedding = broken  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fastembed", module)
        with pytest.raises(EmbedderError, match="cannot load nope: model nope is not supported"):
            FastEmbedEmbedder("nope").embed_query("x")


class TestRealFastEmbed:
    """Checks the installed fastembed against what the embedder expects; no model is loaded."""

    def test_the_library_offers_the_api_the_embedder_uses(self) -> None:
        fastembed = pytest.importorskip("fastembed")
        text_embedding = fastembed.TextEmbedding
        assert callable(text_embedding.passage_embed)
        assert callable(text_embedding.query_embed)
        assert "model_name" in inspect.signature(text_embedding).parameters
        supported = {model["model"] for model in text_embedding.list_supported_models()}
        assert DEFAULT_FASTEMBED_MODEL in supported


class TestCreateEmbedder:
    def test_hashing(self) -> None:
        assert create_embedder("hashing").name == "hashing-512"

    def test_fastembed_with_a_model(self) -> None:
        assert create_embedder("fastembed", "org/small").name == "fastembed-org/small"

    def test_unknown(self) -> None:
        with pytest.raises(EmbedderError, match="unknown embedder 'openai'"):
            create_embedder("openai")


class TestVectorEncoding:
    def test_round_trip(self) -> None:
        vectors = normalize(np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]]))
        rows = [EmbeddingRow(n, encode_vector(vector)) for n, vector in enumerate(vectors)]
        assert all(len(row.vector) == 12 for row in rows)
        assert np.array_equal(decode_vectors(rows, 3), vectors)

    def test_no_rows(self) -> None:
        assert decode_vectors([], 7).shape == (0, 7)

    def test_the_encoding_is_little_endian_float32(self) -> None:
        assert encode_vector(np.array([1.0], dtype=np.float32)) == bytes.fromhex("0000803f")


class TestEmbedMissing:
    def test_embeds_each_chunk_once(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        put(store, collection, "a.md", "cars", "cats")
        embedder = ConceptEmbedder()
        assert embed_missing(store, collection, embedder) == 2
        assert embedder.document_calls == 1
        assert embed_missing(store, collection, embedder) == 0
        assert embedder.document_calls == 1

    def test_only_new_chunks_are_embedded_later(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        put(store, collection, "a.md", "cars")
        embedder = ConceptEmbedder()
        embed_missing(store, collection, embedder)
        put(store, collection, "b.md", "cats", "money")
        assert embed_missing(store, collection, embedder) == 2

    def test_a_changed_document_is_embedded_again(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        put(store, collection, "a.md", "cars")
        embedder = ConceptEmbedder()
        embed_missing(store, collection, embedder)
        put(store, collection, "a.md", "cats")
        assert embed_missing(store, collection, embedder) == 1
        index = DenseIndex.load(store, collection, embedder.name)
        assert len(index) == 1

    def test_batches(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        put(store, collection, "a.md", *[f"cars {n}" for n in range(5)])
        embedder = ConceptEmbedder()
        assert embed_missing(store, collection, embedder, batch_size=2) == 5
        assert embedder.document_calls == 3

    def test_each_embedder_has_its_own_vectors(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        put(store, collection, "a.md", "cars")
        assert embed_missing(store, collection, ConceptEmbedder()) == 1
        assert embed_missing(store, collection, HashingEmbedder(64)) == 1
        assert len(DenseIndex.load(store, collection, "hashing-64")) == 1

    def test_a_wrong_shape_is_rejected_and_nothing_stored(self, store: Store) -> None:
        class Broken(ConceptEmbedder):
            def embed_documents(self, texts: object) -> Vectors:
                return np.zeros((1, 2), dtype=np.float32)

        collection = store.ensure_collection("docs")
        put(store, collection, "a.md", "cars", "cats")
        with pytest.raises(EmbedderError, match=r"returned vectors of shape \(1, 2\)"):
            embed_missing(store, collection, Broken())
        assert len(DenseIndex.load(store, collection, "concepts-test")) == 0


class TestDenseIndex:
    @pytest.fixture
    def index(self, store: Store) -> tuple[DenseIndex, ConceptEmbedder, list[int]]:
        collection = store.ensure_collection("docs")
        ids = put(
            store,
            collection,
            "a.md",
            "cars cars",
            "cars cats",
            "cats sleep",
            "money",
            "nothing here",
        )
        embedder = ConceptEmbedder()
        embed_missing(store, collection, embedder)
        return DenseIndex.load(store, collection, embedder.name), embedder, ids

    def test_results_are_ordered_by_cosine_similarity(
        self, index: tuple[DenseIndex, ConceptEmbedder, list[int]]
    ) -> None:
        dense, embedder, ids = index
        results = dense.search(embedder.embed_query("automobile"), 10)
        assert [i for i, _ in results] == [ids[0], ids[1]]
        assert results[0][1] == pytest.approx(1.0)
        assert results[1][1] == pytest.approx(1 / np.sqrt(2))

    def test_limit(self, index: tuple[DenseIndex, ConceptEmbedder, list[int]]) -> None:
        dense, embedder, ids = index
        assert [i for i, _ in dense.search(embedder.embed_query("sleep"), 1)] == [ids[2]]
        assert dense.search(embedder.embed_query("sleep"), 0) == []

    def test_chunks_with_no_similarity_are_not_results(
        self, index: tuple[DenseIndex, ConceptEmbedder, list[int]]
    ) -> None:
        dense, embedder, ids = index
        found = [i for i, _ in dense.search(embedder.embed_query("money"), 10)]
        assert found == [ids[3]]
        assert dense.search(embedder.embed_query("kubernetes"), 10) == []

    def test_ties_are_broken_by_chunk_id(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        ids = put(store, collection, "a.md", "cats", "cats", "cats")
        embedder = ConceptEmbedder()
        embed_missing(store, collection, embedder)
        dense = DenseIndex.load(store, collection, embedder.name)
        assert [i for i, _ in dense.search(embedder.embed_query("cat"), 10)] == ids

    def test_vectors_of_chunks_in_the_order_asked(
        self, index: tuple[DenseIndex, ConceptEmbedder, list[int]]
    ) -> None:
        dense, embedder, ids = index
        vectors = dense.vectors_of([ids[2], ids[0]])
        assert vectors.shape == (2, embedder.dimension)
        assert vectors[0].tolist() == embedder.embed_documents(["cats sleep"])[0].tolist()
        assert vectors[1].tolist() == embedder.embed_documents(["cars cars"])[0].tolist()
        assert dense.vectors_of([]).shape == (0, embedder.dimension)

    def test_vectors_of_a_chunk_that_is_not_indexed(
        self, index: tuple[DenseIndex, ConceptEmbedder, list[int]]
    ) -> None:
        dense, _, _ = index
        with pytest.raises(EmbedderError, match="chunk 999 has no vector"):
            dense.vectors_of([999])

    def test_an_empty_index(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        dense = DenseIndex.load(store, collection, "concepts-test")
        assert len(dense) == 0
        assert dense.search(ConceptEmbedder().embed_query("cats"), 5) == []

    def test_a_query_of_another_dimension_is_rejected(
        self, index: tuple[DenseIndex, ConceptEmbedder, list[int]]
    ) -> None:
        dense, _, _ = index
        with pytest.raises(EmbedderError, match="the query has 64 dimensions but the stored"):
            dense.search(HashingEmbedder(64).embed_query("cats"), 5)

    def test_collections_do_not_see_each_others_vectors(self, store: Store) -> None:
        mine = store.ensure_collection("mine")
        theirs = store.ensure_collection("theirs")
        put(store, mine, "a.md", "cats")
        put(store, theirs, "b.md", "cats", "cars")
        embedder = ConceptEmbedder()
        embed_missing(store, mine, embedder)
        embed_missing(store, theirs, embedder)
        assert len(DenseIndex.load(store, mine, embedder.name)) == 1
        assert len(DenseIndex.load(store, theirs, embedder.name)) == 2

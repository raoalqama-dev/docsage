from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from docsage.models import Chunk
from docsage.store import (
    SCHEMA_VERSION,
    Collection,
    EmbeddingRow,
    Store,
    StoreError,
    content_hash,
    validate_collection_name,
)


def fixed_clock() -> datetime:
    return datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def chunk(ordinal: int, text: str, heading: tuple[str, ...] = (), page: int | None = None) -> Chunk:
    return Chunk(ordinal, text, heading, page, page, len(text.split()))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "test.db", clock=fixed_clock) as opened:
        yield opened


def put(
    store: Store, collection: Collection, source: str, *chunks: Chunk, digest: str = "h1"
) -> int:
    record = store.put_document(
        collection,
        source=source,
        title=source.upper(),
        format="text",
        content_hash=digest,
        pipeline="1:400/60",
        chunks=chunks,
    )
    return record.id


def put_chunks(
    store: Store, collection: Collection, source: str, *chunks: Chunk, digest: str = "h1"
) -> list[int]:
    """Like :func:`put`, returning the ids of the stored chunks."""
    return [
        stored.id for stored in store.chunks(put(store, collection, source, *chunks, digest=digest))
    ]


def test_content_hash_is_sha256() -> None:
    assert content_hash(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


class TestCollectionNames:
    @pytest.mark.parametrize("name", ["default", "a", "Docs_2026.v1-x", "9" * 64])
    def test_valid(self, name: str) -> None:
        assert validate_collection_name(name) == name

    @pytest.mark.parametrize("name", ["", "-x", ".hidden", "a b", "a/b", "9" * 65, "café"])
    def test_invalid(self, name: str) -> None:
        with pytest.raises(StoreError, match="invalid collection name"):
            validate_collection_name(name)


class TestSchema:
    def test_new_store_is_at_the_current_version(self, tmp_path: Path) -> None:
        with Store(tmp_path / "a.db"):
            pass
        db = sqlite3.connect(tmp_path / "a.db")
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"collections", "documents", "chunks", "chunks_fts", "embeddings"} <= tables
        db.close()

    def test_a_version_1_database_gains_the_search_index_with_its_data(
        self, tmp_path: Path
    ) -> None:
        import docsage.store

        path = tmp_path / "old.db"
        old = sqlite3.connect(path)
        old.executescript(docsage.store._MIGRATIONS[0])
        old.execute("INSERT INTO collections (id, name, created_at) VALUES (1, 'c', 'now')")
        old.execute(
            "INSERT INTO documents (id, collection_id, source, title, format, content_hash, "
            "pipeline, ingested_at) VALUES (1, 1, 'a.txt', 'A', 'text', 'h', 'p', 'now')"
        )
        old.execute(
            "INSERT INTO chunks (id, document_id, ordinal, text, heading_path, token_count) "
            "VALUES (1, 1, 0, 'a pangolin was here', '[]', 4)"
        )
        old.execute("PRAGMA user_version = 1")
        old.commit()
        old.close()
        with Store(path) as upgraded:
            collection = upgraded.collection("c")
            assert collection is not None
            assert [i for i, _ in upgraded.fts_search(collection, '"pangolin"', 5)] == [1]

    def test_reopening_keeps_data(self, tmp_path: Path) -> None:
        with Store(tmp_path / "a.db") as first:
            put(first, first.ensure_collection("c"), "a.txt", chunk(0, "hello"))
        with Store(tmp_path / "a.db") as second:
            collection = second.collection("c")
            assert collection is not None
            assert [doc.source for doc in second.documents(collection)] == ["a.txt"]

    def test_newer_schema_is_refused(self, tmp_path: Path) -> None:
        db = sqlite3.connect(tmp_path / "new.db")
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        db.close()
        with pytest.raises(StoreError, match="newer than this docsage supports"):
            Store(tmp_path / "new.db")

    def test_not_a_database(self, tmp_path: Path) -> None:
        (tmp_path / "junk.db").write_bytes(b"this is not sqlite" * 100)
        with pytest.raises(StoreError, match="cannot open"):
            Store(tmp_path / "junk.db")

    def test_unopenable_path(self, tmp_path: Path) -> None:
        with pytest.raises(StoreError, match="cannot open"):
            Store(tmp_path / "missing-dir" / "x.db")

    def test_failed_migration_rolls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import docsage.store

        broken = (
            *docsage.store._MIGRATIONS,
            "CREATE TABLE extra (x); SELECT nonsense FROM;",
        )
        monkeypatch.setattr(docsage.store, "_MIGRATIONS", broken)
        monkeypatch.setattr(docsage.store, "SCHEMA_VERSION", len(broken))
        with pytest.raises(StoreError, match="cannot open"):
            Store(tmp_path / "m.db")
        db = sqlite3.connect(tmp_path / "m.db")
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "extra" not in tables
        db.close()


class TestCollections:
    def test_ensure_creates_once(self, store: Store) -> None:
        first = store.ensure_collection("docs")
        assert store.ensure_collection("docs") == first
        assert store.collection("docs") == first
        assert store.collection("other") is None

    def test_ensure_validates_the_name(self, store: Store) -> None:
        with pytest.raises(StoreError):
            store.ensure_collection("bad name")

    def test_stats(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        store.ensure_collection("empty")
        put(store, docs, "a.txt", chunk(0, "one two"), chunk(1, "three"))
        put(store, docs, "b.txt", chunk(0, "four five six"))
        put(store, docs, "c.txt")
        stats = {row.name: row for row in store.collection_stats()}
        assert (stats["docs"].documents, stats["docs"].chunks, stats["docs"].tokens) == (3, 3, 6)
        assert (stats["empty"].documents, stats["empty"].chunks, stats["empty"].tokens) == (
            0,
            0,
            0,
        )
        assert [row.name for row in store.collection_stats()] == ["docs", "empty"]


class TestDocuments:
    def test_put_and_read_back(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        doc_id = put(store, docs, "guide.md", chunk(0, "x", ("A", "B é"), 3), chunk(1, "y"))
        record = store.document(docs, "guide.md")
        assert record is not None
        assert record.id == doc_id
        assert (record.title, record.format, record.content_hash, record.pipeline) == (
            "GUIDE.MD",
            "text",
            "h1",
            "1:400/60",
        )
        assert record.ingested_at == "2026-01-02T03:04:05+00:00"
        stored = store.chunks(doc_id)
        assert [item.chunk for item in stored] == [chunk(0, "x", ("A", "B é"), 3), chunk(1, "y")]
        assert {(item.source, item.title, item.document_id) for item in stored} == {
            ("guide.md", "GUIDE.MD", doc_id)
        }

    def test_replacing_keeps_the_id_and_swaps_the_chunks(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        first = put(store, docs, "a.txt", chunk(0, "old"), chunk(1, "old two"))
        second = put(store, docs, "a.txt", chunk(0, "new"), digest="h2")
        assert first == second
        assert [item.chunk.text for item in store.chunks(first)] == ["new"]
        record = store.document(docs, "a.txt")
        assert record is not None
        assert record.content_hash == "h2"

    def test_same_source_in_two_collections_is_two_documents(self, store: Store) -> None:
        one, two = store.ensure_collection("one"), store.ensure_collection("two")
        assert put(store, one, "a.txt", chunk(0, "1")) != put(store, two, "a.txt", chunk(0, "2"))
        assert [doc.source for doc in store.documents(one)] == ["a.txt"]

    def test_documents_are_sorted_by_source(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        for source in ["b.md", "a/z.md", "a.md"]:
            put(store, docs, source)
        assert [doc.source for doc in store.documents(docs)] == ["a.md", "a/z.md", "b.md"]

    def test_delete_removes_chunks(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        doc_id = put(store, docs, "a.txt", chunk(0, "x"))
        assert store.delete_document(docs, "a.txt")
        assert not store.delete_document(docs, "a.txt")
        assert store.document(docs, "a.txt") is None
        assert store.chunks(doc_id) == []

    def test_failed_put_leaves_the_old_version(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        doc_id = put(store, docs, "a.txt", chunk(0, "kept"))
        duplicate_ordinals = (chunk(0, "new"), chunk(0, "clash"))
        with pytest.raises(sqlite3.IntegrityError):
            put(store, docs, "a.txt", *duplicate_ordinals, digest="h2")
        assert [item.chunk.text for item in store.chunks(doc_id)] == ["kept"]
        record = store.document(docs, "a.txt")
        assert record is not None
        assert record.content_hash == "h1"


class TestChunksById:
    def test_returns_the_requested_chunks(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        ids = [
            i
            for source in ("a.txt", "b.txt")
            for i in put_chunks(store, docs, source, chunk(0, source))
        ]
        found = store.chunks_by_id([ids[1], ids[0]])
        assert {i: stored.source for i, stored in found.items()} == {
            ids[0]: "a.txt",
            ids[1]: "b.txt",
        }
        assert found[ids[0]].title == "A.TXT"

    def test_unknown_ids_are_absent(self, store: Store) -> None:
        assert store.chunks_by_id([12345]) == {}
        assert store.chunks_by_id([]) == {}

    def test_more_ids_than_fit_in_one_statement(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        ids = put_chunks(store, docs, "big.txt", *[chunk(n, f"chunk {n}") for n in range(1200)])
        found = store.chunks_by_id(ids)
        assert len(found) == 1200
        assert found[ids[1100]].chunk.text == "chunk 1100"


class TestEmbeddings:
    @staticmethod
    def vector(*values: float) -> bytes:
        return struct.pack(f"<{len(values)}f", *values)

    def test_unembedded_chunks_shrink_as_vectors_are_stored(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        a, b = put_chunks(store, docs, "a.txt", chunk(0, "one"), chunk(1, "two"))
        assert store.unembedded_chunks(docs, "m") == [(a, "one"), (b, "two")]
        store.put_embeddings("m", 2, [EmbeddingRow(a, self.vector(1, 0))])
        assert store.unembedded_chunks(docs, "m") == [(b, "two")]
        assert store.unembedded_chunks(docs, "other") == [(a, "one"), (b, "two")]

    def test_read_back(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        a, b = put_chunks(store, docs, "a.txt", chunk(0, "one"), chunk(1, "two"))
        store.put_embeddings(
            "m", 2, [EmbeddingRow(b, self.vector(0, 1)), EmbeddingRow(a, self.vector(1, 0))]
        )
        dimension, rows = store.embeddings(docs, "m")
        assert dimension == 2
        assert rows == [EmbeddingRow(a, self.vector(1, 0)), EmbeddingRow(b, self.vector(0, 1))]
        assert store.embeddings(docs, "other") == (0, [])

    def test_storing_again_replaces(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        [a] = put_chunks(store, docs, "a.txt", chunk(0, "one"))
        store.put_embeddings("m", 2, [EmbeddingRow(a, self.vector(1, 0))])
        store.put_embeddings("m", 2, [EmbeddingRow(a, self.vector(0, 1))])
        assert store.embeddings(docs, "m")[1] == [EmbeddingRow(a, self.vector(0, 1))]

    def test_wrong_sized_vectors_are_rejected_and_nothing_is_stored(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        a, b = put_chunks(store, docs, "a.txt", chunk(0, "one"), chunk(1, "two"))
        with pytest.raises(StoreError, match=f"vector for chunk {b} has 12 bytes, expected 8"):
            store.put_embeddings(
                "m", 2, [EmbeddingRow(a, self.vector(1, 0)), EmbeddingRow(b, self.vector(1, 0, 0))]
            )
        assert store.embeddings(docs, "m") == (0, [])

    def test_mixed_dimensions_for_one_model_are_an_error(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        a, b = put_chunks(store, docs, "a.txt", chunk(0, "one"), chunk(1, "two"))
        store.put_embeddings("m", 2, [EmbeddingRow(a, self.vector(1, 0))])
        store.put_embeddings("m", 3, [EmbeddingRow(b, self.vector(1, 0, 0))])
        with pytest.raises(StoreError, match="different dimensions"):
            store.embeddings(docs, "m")

    def test_replacing_a_document_drops_its_vectors(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        [a] = put_chunks(store, docs, "a.txt", chunk(0, "one"))
        store.put_embeddings("m", 2, [EmbeddingRow(a, self.vector(1, 0))])
        [new] = put_chunks(store, docs, "a.txt", chunk(0, "uno"), digest="h2")
        assert store.embeddings(docs, "m") == (0, [])
        assert store.unembedded_chunks(docs, "m") == [(new, "uno")]

    def test_deleting_a_document_drops_its_vectors(self, store: Store) -> None:
        docs = store.ensure_collection("docs")
        [a] = put_chunks(store, docs, "a.txt", chunk(0, "one"))
        store.put_embeddings("m", 2, [EmbeddingRow(a, self.vector(1, 0))])
        store.delete_document(docs, "a.txt")
        count = store._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        assert count == 0

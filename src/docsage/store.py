"""The SQLite store: collections of documents and their chunks.

A store is a single SQLite file. Its schema is versioned with
``PRAGMA user_version`` and migrated forward when opened. Each document
records the SHA-256 of the file it came from and the pipeline that chunked it,
so ingesting it again only does work when either has changed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from docsage.models import Chunk

_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE collections (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL
    );
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        collection_id INTEGER NOT NULL REFERENCES collections (id) ON DELETE CASCADE,
        source TEXT NOT NULL,
        title TEXT NOT NULL,
        format TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        pipeline TEXT NOT NULL,
        ingested_at TEXT NOT NULL,
        UNIQUE (collection_id, source)
    );
    CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        document_id INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL,
        text TEXT NOT NULL,
        heading_path TEXT NOT NULL,
        page INTEGER,
        page_end INTEGER,
        token_count INTEGER NOT NULL,
        UNIQUE (document_id, ordinal)
    );
    """,
)

SCHEMA_VERSION = len(_MIGRATIONS)

_COLLECTION_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class StoreError(Exception):
    """A store that cannot be opened, or a request it cannot satisfy."""


def content_hash(data: bytes) -> str:
    """The SHA-256 of a file's bytes, as hex."""
    return hashlib.sha256(data).hexdigest()


def validate_collection_name(name: str) -> str:
    """Collection names are 1-64 letters, digits, ``_``, ``.`` or ``-``, starting alphanumeric."""
    if not _COLLECTION_NAME_RE.fullmatch(name):
        raise StoreError(
            f"invalid collection name {name!r}: use 1-64 letters, digits, '_', '.' or '-', "
            "starting with a letter or digit"
        )
    return name


@dataclass(frozen=True, slots=True)
class Collection:
    id: int
    name: str


@dataclass(frozen=True, slots=True)
class CollectionStats:
    name: str
    documents: int
    chunks: int
    tokens: int


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: int
    collection_id: int
    source: str
    title: str
    format: str
    content_hash: str
    pipeline: str
    ingested_at: str


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A chunk as stored, with the document it belongs to."""

    id: int
    document_id: int
    source: str
    title: str
    chunk: Chunk


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Store:
    """A docsage database. Use it as a context manager, or call :meth:`close`."""

    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self.path = Path(path)
        self._clock = clock
        try:
            self._db = sqlite3.connect(self.path, isolation_level=None)
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open {self.path}: {exc}") from exc
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._migrate()
        except (sqlite3.Error, StoreError) as exc:
            self._db.close()
            if isinstance(exc, StoreError):
                raise
            raise StoreError(f"cannot open {self.path}: {exc}") from exc

    def __enter__(self) -> Store:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def _migrate(self) -> None:
        version: int = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise StoreError(
                f"{self.path} has schema version {version}, newer than this docsage "
                f"supports ({SCHEMA_VERSION}); upgrade docsage"
            )
        for number in range(version, SCHEMA_VERSION):
            # executescript, unlike execute, accepts the semicolons inside trigger bodies.
            try:
                self._db.executescript(
                    f"BEGIN IMMEDIATE;\n{_MIGRATIONS[number]}\n"
                    f"PRAGMA user_version = {number + 1};\nCOMMIT;"
                )
            except sqlite3.Error:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run the block in one transaction, rolled back if it raises."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def _now(self) -> str:
        return self._clock().isoformat(timespec="seconds")

    # Collections

    def collection(self, name: str) -> Collection | None:
        row = self._db.execute(
            "SELECT id, name FROM collections WHERE name = ?", (name,)
        ).fetchone()
        return Collection(row["id"], row["name"]) if row else None

    def ensure_collection(self, name: str) -> Collection:
        """The collection called ``name``, created if it does not exist."""
        existing = self.collection(validate_collection_name(name))
        if existing:
            return existing
        cursor = self._db.execute(
            "INSERT INTO collections (name, created_at) VALUES (?, ?)", (name, self._now())
        )
        return Collection(_lastrowid(cursor), name)

    def collection_stats(self) -> list[CollectionStats]:
        rows = self._db.execute(
            """
            SELECT c.name,
                   (SELECT COUNT(*) FROM documents d WHERE d.collection_id = c.id) AS documents,
                   COUNT(k.id) AS chunks,
                   COALESCE(SUM(k.token_count), 0) AS tokens
            FROM collections c
            LEFT JOIN documents d ON d.collection_id = c.id
            LEFT JOIN chunks k ON k.document_id = d.id
            GROUP BY c.id
            ORDER BY c.name
            """
        ).fetchall()
        return [
            CollectionStats(row["name"], row["documents"], row["chunks"], row["tokens"])
            for row in rows
        ]

    # Documents

    def document(self, collection: Collection, source: str) -> DocumentRecord | None:
        row = self._db.execute(
            "SELECT * FROM documents WHERE collection_id = ? AND source = ?",
            (collection.id, source),
        ).fetchone()
        return _document(row) if row else None

    def documents(self, collection: Collection) -> list[DocumentRecord]:
        rows = self._db.execute(
            "SELECT * FROM documents WHERE collection_id = ? ORDER BY source", (collection.id,)
        ).fetchall()
        return [_document(row) for row in rows]

    def put_document(
        self,
        collection: Collection,
        *,
        source: str,
        title: str,
        format: str,
        content_hash: str,
        pipeline: str,
        chunks: Sequence[Chunk],
    ) -> DocumentRecord:
        """Store a document and its chunks, replacing the chunks of an earlier version.

        A document keeps its id across versions; all of its old chunks are
        removed and the new ones inserted in one transaction.
        """
        now = self._now()
        with self.transaction():
            existing = self.document(collection, source)
            if existing:
                document_id = existing.id
                self._db.execute(
                    """
                    UPDATE documents
                    SET title = ?, format = ?, content_hash = ?, pipeline = ?, ingested_at = ?
                    WHERE id = ?
                    """,
                    (title, format, content_hash, pipeline, now, document_id),
                )
                self._db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            else:
                cursor = self._db.execute(
                    """
                    INSERT INTO documents
                        (collection_id, source, title, format, content_hash, pipeline, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (collection.id, source, title, format, content_hash, pipeline, now),
                )
                document_id = _lastrowid(cursor)
            self._db.executemany(
                """
                INSERT INTO chunks
                    (document_id, ordinal, text, heading_path, page, page_end, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        chunk.ordinal,
                        chunk.text,
                        json.dumps(list(chunk.heading_path), ensure_ascii=False),
                        chunk.page,
                        chunk.page_end,
                        chunk.token_count,
                    )
                    for chunk in chunks
                ],
            )
        record = self.document(collection, source)
        assert record is not None
        return record

    def delete_document(self, collection: Collection, source: str) -> bool:
        """Remove a document and its chunks. Returns whether it existed."""
        with self.transaction():
            cursor = self._db.execute(
                "DELETE FROM documents WHERE collection_id = ? AND source = ?",
                (collection.id, source),
            )
        return cursor.rowcount > 0

    # Chunks

    def chunks(self, document_id: int) -> list[StoredChunk]:
        """A document's chunks in order."""
        rows = self._db.execute(
            """
            SELECT k.*, d.source, d.title FROM chunks k JOIN documents d ON d.id = k.document_id
            WHERE k.document_id = ? ORDER BY k.ordinal
            """,
            (document_id,),
        ).fetchall()
        return [_stored_chunk(row) for row in rows]


def _lastrowid(cursor: sqlite3.Cursor) -> int:
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def _document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        collection_id=row["collection_id"],
        source=row["source"],
        title=row["title"],
        format=row["format"],
        content_hash=row["content_hash"],
        pipeline=row["pipeline"],
        ingested_at=row["ingested_at"],
    )


def _stored_chunk(row: sqlite3.Row) -> StoredChunk:
    return StoredChunk(
        id=row["id"],
        document_id=row["document_id"],
        source=row["source"],
        title=row["title"],
        chunk=Chunk(
            ordinal=row["ordinal"],
            text=row["text"],
            heading_path=tuple(json.loads(row["heading_path"])),
            page=row["page"],
            page_end=row["page_end"],
            token_count=row["token_count"],
        ),
    )

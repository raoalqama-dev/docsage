"""Ingesting files into a collection: discover, hash, load, chunk and store."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from docsage.ingest.chunking import ChunkingOptions, chunk_document
from docsage.ingest.loaders import LoaderError, format_for, load_document
from docsage.store import Store, content_hash

#: Bumped whenever a change to the loaders or the chunker changes their output,
#: so documents stored by an older pipeline are chunked again.
PIPELINE_VERSION = "1"


class Outcome(StrEnum):
    ADDED = "added"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FileResult:
    source: str
    outcome: Outcome
    chunks: int = 0
    error: str | None = None


@dataclass(slots=True)
class IngestReport:
    collection: str
    results: list[FileResult] = field(default_factory=list)

    def count(self, outcome: Outcome) -> int:
        return sum(1 for result in self.results if result.outcome is outcome)

    @property
    def failed(self) -> bool:
        return self.count(Outcome.FAILED) > 0


def pipeline_signature(options: ChunkingOptions) -> str:
    return f"{PIPELINE_VERSION}:{options.signature}"


def discover(paths: Iterable[Path]) -> tuple[list[tuple[Path, str]], list[FileResult]]:
    """Supported files under ``paths``, each with its source name, and the paths that failed.

    A file's source name is its path relative to the directory it was found
    in, or its file name when given directly, with ``/`` separators. Hidden
    files and directories are skipped.
    """
    found: list[tuple[Path, str]] = []
    errors: list[FileResult] = []
    seen: dict[str, Path] = {}

    def add(path: Path, source: str) -> None:
        if source in seen and seen[source] != path:
            errors.append(
                FileResult(source, Outcome.FAILED, error=f"same source name as {seen[source]}")
            )
            return
        if source not in seen:
            seen[source] = path
            found.append((path, source))

    for root in paths:
        if root.is_dir():
            under_root: list[tuple[str, Path]] = []
            for directory, subdirs, files in os.walk(root):
                subdirs[:] = [name for name in subdirs if not name.startswith(".")]
                for name in files:
                    path = Path(directory, name)
                    if not name.startswith(".") and format_for(path):
                        under_root.append((path.relative_to(root).as_posix(), path))
            for source, path in sorted(under_root):
                add(path, source)
        elif root.is_file():
            if format_for(root):
                add(root, root.name)
            else:
                errors.append(FileResult(root.name, Outcome.FAILED, error="unsupported file type"))
        else:
            errors.append(FileResult(str(root), Outcome.FAILED, error="no such file or directory"))
    return found, errors


def ingest(
    store: Store,
    collection_name: str,
    paths: Iterable[Path],
    options: ChunkingOptions | None = None,
) -> IngestReport:
    """Ingest files and directories into a collection, skipping unchanged files."""
    options = options or ChunkingOptions()
    pipeline = pipeline_signature(options)
    collection = store.ensure_collection(collection_name)
    report = IngestReport(collection.name)
    files, errors = discover(paths)
    report.results.extend(errors)
    for path, source in files:
        try:
            data = path.read_bytes()
        except OSError as exc:
            report.results.append(FileResult(source, Outcome.FAILED, error=exc.strerror))
            continue
        digest = content_hash(data)
        existing = store.document(collection, source)
        if existing and existing.content_hash == digest and existing.pipeline == pipeline:
            report.results.append(FileResult(source, Outcome.UNCHANGED))
            continue
        try:
            document = load_document(path, data)
        except LoaderError as exc:
            report.results.append(FileResult(source, Outcome.FAILED, error=str(exc)))
            continue
        chunks = chunk_document(document, options)
        store.put_document(
            collection,
            source=source,
            title=document.title,
            format=document.format,
            content_hash=digest,
            pipeline=pipeline,
            chunks=chunks,
        )
        outcome = Outcome.UPDATED if existing else Outcome.ADDED
        report.results.append(FileResult(source, outcome, chunks=len(chunks)))
    return report

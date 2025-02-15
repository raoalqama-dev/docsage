from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from docsage.ingest.chunking import ChunkingOptions
from docsage.ingest.pipeline import (
    PIPELINE_VERSION,
    Outcome,
    discover,
    ingest,
    pipeline_signature,
)
from docsage.store import Collection, Store
from fixtures import make_pdf, prose


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "docsage.db") as opened:
        yield opened


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "guides").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "guides" / "raft.md").write_text(
        "# Raft\n\nRaft is a consensus algorithm.\n\n## Elections\n\n" + prose(30),
        encoding="utf-8",
    )
    (root / "notes.txt").write_text("Plain notes.\n\nSecond paragraph.", encoding="utf-8")
    (root / "page.html").write_text(
        "<title>Paxos</title><h1>Paxos</h1><p>Paxos is older.</p>", encoding="utf-8"
    )
    (root / "paper.pdf").write_bytes(
        make_pdf(
            [[["Page one text."]], [["Page two text."]]],
            outline=[("Intro", 0, 1), ("Body", 1, 1)],
            title="A paper",
        )
    )
    (root / "image.png").write_bytes(b"\x89PNG")
    (root / ".hidden.md").write_text("# secret", encoding="utf-8")
    (root / ".git" / "config.md").write_text("# not a doc", encoding="utf-8")
    return root


def outcomes(report_results: object) -> dict[str, Outcome]:
    assert isinstance(report_results, list)
    return {result.source: result.outcome for result in report_results}


def collection(store: Store, name: str = "docs") -> Collection:
    found = store.collection(name)
    assert found is not None
    return found


class TestDiscover:
    def test_walks_directories_and_names_sources_relative_to_them(self, corpus: Path) -> None:
        files, errors = discover([corpus])
        assert errors == []
        assert [source for _, source in files] == [
            "guides/raft.md",
            "notes.txt",
            "page.html",
            "paper.pdf",
        ]

    def test_a_file_argument_is_named_by_its_file_name(self, corpus: Path) -> None:
        files, _ = discover([corpus / "guides" / "raft.md"])
        assert [source for _, source in files] == ["raft.md"]

    def test_missing_and_unsupported_paths(self, corpus: Path) -> None:
        files, errors = discover([corpus / "nope.md", corpus / "image.png"])
        assert files == []
        assert [(error.source, error.error) for error in errors] == [
            (str(corpus / "nope.md"), "no such file or directory"),
            ("image.png", "unsupported file type"),
        ]

    def test_two_files_with_one_source_name(self, corpus: Path, tmp_path: Path) -> None:
        other = tmp_path / "other"
        other.mkdir()
        (other / "notes.txt").write_text("different", encoding="utf-8")
        files, errors = discover([corpus, other])
        assert [path for path, source in files if source == "notes.txt"] == [corpus / "notes.txt"]
        assert len(errors) == 1
        assert errors[0].source == "notes.txt"
        assert errors[0].error == f"same source name as {corpus / 'notes.txt'}"

    def test_the_same_file_twice_is_ingested_once(self, corpus: Path) -> None:
        files, errors = discover([corpus, corpus])
        assert errors == []
        assert len(files) == 4


class TestIngest:
    def test_first_ingest_adds_every_file_with_metadata(self, store: Store, corpus: Path) -> None:
        report = ingest(store, "docs", [corpus])
        assert report.collection == "docs"
        assert outcomes(report.results) == {
            "guides/raft.md": Outcome.ADDED,
            "notes.txt": Outcome.ADDED,
            "page.html": Outcome.ADDED,
            "paper.pdf": Outcome.ADDED,
        }
        assert not report.failed
        docs = {doc.source: doc for doc in store.documents(collection(store))}
        assert {source: (doc.title, doc.format) for source, doc in docs.items()} == {
            "guides/raft.md": ("Raft", "markdown"),
            "notes.txt": ("notes", "text"),
            "page.html": ("Paxos", "html"),
            "paper.pdf": ("A paper", "pdf"),
        }
        assert all(doc.pipeline == f"{PIPELINE_VERSION}:400/60" for doc in docs.values())

        pdf_chunks = [item.chunk for item in store.chunks(docs["paper.pdf"].id)]
        assert [(c.heading_path, c.page, c.page_end, c.text) for c in pdf_chunks] == [
            (("Intro",), 1, 1, "Page one text."),
            (("Body",), 2, 2, "Page two text."),
        ]
        raft = [item.chunk for item in store.chunks(docs["guides/raft.md"].id)]
        assert raft[0].heading_path == ("Raft",)
        assert raft[0].text == "Raft is a consensus algorithm."
        assert len(raft) > 2
        assert all(chunk.heading_path == ("Raft", "Elections") for chunk in raft[1:])
        assert all(chunk.token_count <= 400 for chunk in raft)

    def test_reingesting_unchanged_files_is_a_no_op(self, store: Store, corpus: Path) -> None:
        ingest(store, "docs", [corpus])
        before = {
            doc.source: (doc.id, doc.ingested_at, [(c.id, c.chunk) for c in store.chunks(doc.id)])
            for doc in store.documents(collection(store))
        }
        report = ingest(store, "docs", [corpus])
        assert set(outcomes(report.results).values()) == {Outcome.UNCHANGED}
        after = {
            doc.source: (doc.id, doc.ingested_at, [(c.id, c.chunk) for c in store.chunks(doc.id)])
            for doc in store.documents(collection(store))
        }
        assert after == before

    def test_chunks_are_stable_across_stores(self, tmp_path: Path, corpus: Path) -> None:
        def snapshot(path: Path) -> list[object]:
            with Store(path) as opened:
                ingest(opened, "docs", [corpus])
                return [
                    (doc.source, [item.chunk for item in opened.chunks(doc.id)])
                    for doc in opened.documents(collection(opened))
                ]

        assert snapshot(tmp_path / "one.db") == snapshot(tmp_path / "two.db")

    def test_a_changed_file_replaces_its_chunks(self, store: Store, corpus: Path) -> None:
        ingest(store, "docs", [corpus])
        notes = store.document(collection(store), "notes.txt")
        assert notes is not None
        (corpus / "notes.txt").write_text("Rewritten notes.", encoding="utf-8")
        report = ingest(store, "docs", [corpus])
        assert outcomes(report.results)["notes.txt"] is Outcome.UPDATED
        assert [r.outcome for r in report.results if r.source != "notes.txt"] == [
            Outcome.UNCHANGED
        ] * 3
        updated = store.document(collection(store), "notes.txt")
        assert updated is not None
        assert updated.id == notes.id
        assert updated.content_hash != notes.content_hash
        assert [item.chunk.text for item in store.chunks(notes.id)] == ["Rewritten notes."]

    def test_new_chunking_options_rechunk_everything(self, store: Store, corpus: Path) -> None:
        ingest(store, "docs", [corpus])
        options = ChunkingOptions(max_tokens=50, overlap_tokens=10)
        report = ingest(store, "docs", [corpus], options)
        assert set(outcomes(report.results).values()) == {Outcome.UPDATED}
        raft = store.document(collection(store), "guides/raft.md")
        assert raft is not None
        assert raft.pipeline == pipeline_signature(options)
        assert all(item.chunk.token_count <= 50 for item in store.chunks(raft.id))

    def test_failures_are_reported_and_the_rest_ingested(self, store: Store, corpus: Path) -> None:
        (corpus / "broken.pdf").write_bytes(b"%PDF-1.4 truncated")
        report = ingest(store, "docs", [corpus, corpus / "missing.txt"])
        results = {result.source: result for result in report.results}
        assert report.failed
        assert results["broken.pdf"].outcome is Outcome.FAILED
        assert (results["broken.pdf"].error or "").startswith("cannot read the PDF")
        assert results[str(corpus / "missing.txt")].outcome is Outcome.FAILED
        assert report.count(Outcome.ADDED) == 4
        assert store.document(collection(store), "broken.pdf") is None

    def test_unreadable_file(
        self, store: Store, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(self: Path) -> bytes:
            raise PermissionError(13, "Permission denied", str(self))

        monkeypatch.setattr(Path, "read_bytes", refuse)
        report = ingest(store, "docs", [corpus / "notes.txt"])
        assert [(r.outcome, r.error) for r in report.results] == [
            (Outcome.FAILED, "Permission denied")
        ]

    def test_empty_files_are_stored_without_chunks(self, store: Store, tmp_path: Path) -> None:
        (tmp_path / "empty.md").write_text("", encoding="utf-8")
        report = ingest(store, "docs", [tmp_path / "empty.md"])
        assert [(r.outcome, r.chunks) for r in report.results] == [(Outcome.ADDED, 0)]
        assert outcomes(ingest(store, "docs", [tmp_path / "empty.md"]).results) == {
            "empty.md": Outcome.UNCHANGED
        }

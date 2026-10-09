from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from docsage.models import Chunk
from docsage.retrieval.bm25 import MAX_TERMS, bm25_search, build_match, query_terms
from docsage.store import Collection, Store


def chunk(ordinal: int, text: str, heading: tuple[str, ...] = ()) -> Chunk:
    return Chunk(ordinal, text, heading, None, None, len(text.split()))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "test.db") as opened:
        yield opened


def put(store: Store, collection: Collection, source: str, *chunks: Chunk) -> list[int]:
    record = store.put_document(
        collection,
        source=source,
        title=source,
        format="text",
        content_hash=source,
        pipeline="1",
        chunks=chunks,
    )
    return [stored.id for stored in store.chunks(record.id)]


class TestQueryConstruction:
    def test_words_are_lowercased_and_deduplicated_in_order(self) -> None:
        assert query_terms("Raft raft LEADER, Election; leader") == ["raft", "leader", "election"]

    def test_punctuation_and_underscores_separate_words(self) -> None:
        assert query_terms("foo_bar-baz.qux (x)") == ["foo", "bar", "baz", "qux", "x"]

    def test_other_scripts_and_accents_are_words(self) -> None:
        assert query_terms("Größe 日本語 café") == ["größe", "日本語", "café"]

    def test_the_number_of_terms_is_capped(self) -> None:
        terms = query_terms(" ".join(f"w{n}" for n in range(MAX_TERMS + 20)))
        assert len(terms) == MAX_TERMS
        assert terms[-1] == f"w{MAX_TERMS - 1}"

    @pytest.mark.parametrize("query", ["", "   ", "!!! ??? ---", "()", "_"])
    def test_a_query_without_words_has_no_match_expression(self, query: str) -> None:
        assert build_match(query) is None

    def test_every_word_is_quoted(self) -> None:
        assert build_match("raft AND NOT leader*") == '"raft" OR "and" OR "not" OR "leader"'

    @pytest.mark.parametrize(
        "query",
        [
            '"unbalanced',
            'say "hello',
            "a AND",
            "OR b",
            "NOT",
            "NEAR(a b, 2)",
            "raft*",
            "-raft",
            "+raft",
            "title:raft",
            "{text heading_path}: raft",
            "^raft",
            ")(",
            "a AND (b OR",
            "'; DROP TABLE chunks; --",
            '"" ""',
            "text : *",
        ],
    )
    def test_expressions_contain_only_quoted_words(self, query: str) -> None:
        match = build_match(query)
        assert match is None or all(
            part.startswith('"') and part.endswith('"') and part.count('"') == 2
            for part in match.split(" OR ")
        )


class TestSearch:
    @pytest.fixture
    def docs(self, store: Store) -> Collection:
        collection = store.ensure_collection("docs")
        put(
            store,
            collection,
            "raft.md",
            chunk(0, "Raft elects a leader. The leader replicates the log.", ("Raft", "Leader")),
            chunk(1, "Followers vote in elections. A candidate needs a majority."),
        )
        put(store, collection, "paxos.md", chunk(0, "Paxos is a family of protocols."))
        put(store, collection, "cooking.md", chunk(0, "Whisk the eggs. Fold in the flour."))
        return collection

    @pytest.mark.parametrize(
        "query",
        [
            '"unbalanced',
            "leader AND",
            "NEAR(leader log)",
            "leader*",
            "-leader",
            "text:leader",
            "heading_path: Raft",
            ")(",
            "'; DROP TABLE chunks; --",
            "OR",
            "AND OR NOT NEAR",
        ],
    )
    def test_fts_syntax_in_a_query_cannot_break_the_search(
        self, store: Store, docs: Collection, query: str
    ) -> None:
        bm25_search(store, docs, query, 10)  # does not raise
        assert sum(row.chunks for row in store.collection_stats()) == 4

    def test_operators_are_searched_as_words(self, store: Store, docs: Collection) -> None:
        ids = put(store, docs, "ops.md", chunk(0, "Combine filters with AND, OR and NOT."))
        assert [i for i, _ in bm25_search(store, docs, "NOT", 10)] == ids
        assert bm25_search(store, docs, "leader AND", 10) != []

    def test_a_star_is_not_a_prefix_wildcard(self, store: Store, docs: Collection) -> None:
        assert bm25_search(store, docs, "lead*", 10) == []
        assert bm25_search(store, docs, "lead", 10) == []

    def test_a_column_filter_is_not_applied(self, store: Store, docs: Collection) -> None:
        # "heading_path:" would restrict the match to headings if it reached FTS5.
        by_filter = bm25_search(store, docs, "heading_path:log", 10)
        plain = bm25_search(store, docs, "heading path log", 10)
        assert by_filter == plain != []

    def test_matches_are_ranked_by_bm25(self, store: Store, docs: Collection) -> None:
        results = bm25_search(store, docs, "leader log", 10)
        chunks = store.chunks_by_id([i for i, _ in results])
        assert [chunks[i].chunk.text[:10] for i, _ in results] == ["Raft elect"]
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        assert all(score > 0 for score in scores)

    def test_more_occurrences_rank_higher(self, store: Store, docs: Collection) -> None:
        ids = put(
            store,
            docs,
            "many.md",
            chunk(0, "leader leader leader election"),
            chunk(1, "the leader waits"),
        )
        results = bm25_search(store, docs, "leader", 10)
        assert [i for i, _ in results][:2] == [ids[0], ids[1]]

    def test_words_are_stemmed_and_accents_folded(self, store: Store, docs: Collection) -> None:
        [garden] = put(store, docs, "g.md", chunk(0, "Gardens keep growing"))
        [school] = put(store, docs, "fr.md", chunk(0, "Les élèves apprennent"))
        assert [i for i, _ in bm25_search(store, docs, "GARDEN", 10)] == [garden]
        assert [i for i, _ in bm25_search(store, docs, "grow", 10)] == [garden]
        assert [i for i, _ in bm25_search(store, docs, "eleves", 10)] == [school]
        assert [i for i, _ in bm25_search(store, docs, "élève", 10)] == [school]

    def test_headings_are_searchable_but_weigh_less_than_text(
        self, store: Store, docs: Collection
    ) -> None:
        put(store, docs, "h.md", chunk(0, "Unrelated words.", ("Snapshots",)))
        put(store, docs, "t.md", chunk(0, "Snapshots compact the log."))
        results = bm25_search(store, docs, "snapshots", 10)
        chunks = store.chunks_by_id([i for i, _ in results])
        assert [chunks[i].source for i, _ in results] == ["t.md", "h.md"]

    def test_the_json_punctuation_of_the_heading_path_is_not_a_word(
        self, store: Store, docs: Collection
    ) -> None:
        assert bm25_search(store, docs, "[ ] ,", 10) == []

    def test_limit(self, store: Store, docs: Collection) -> None:
        for number in range(5):
            put(store, docs, f"d{number}.md", chunk(0, "common word"))
        assert len(bm25_search(store, docs, "common", 3)) == 3
        assert bm25_search(store, docs, "common", 0) == []

    def test_empty_query_and_no_matches(self, store: Store, docs: Collection) -> None:
        assert bm25_search(store, docs, "", 10) == []
        assert bm25_search(store, docs, "kubernetes", 10) == []

    def test_searches_are_scoped_to_a_collection(self, store: Store, docs: Collection) -> None:
        other = store.ensure_collection("other")
        put(store, other, "x.md", chunk(0, "The leader of another collection."))
        mine = {i for i, _ in bm25_search(store, docs, "leader", 10)}
        theirs = {i for i, _ in bm25_search(store, other, "leader", 10)}
        assert mine
        assert len(theirs) == 1
        assert mine.isdisjoint(theirs)

    def test_ties_are_broken_by_chunk_id(self, store: Store, docs: Collection) -> None:
        ids = [put(store, docs, f"t{n}.md", chunk(0, "identical text"))[0] for n in range(3)]
        assert [i for i, _ in bm25_search(store, docs, "identical", 10)] == ids


class TestIndexStaysInSync:
    def test_replacing_a_document_replaces_its_entries(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        put(store, collection, "a.md", chunk(0, "old zebra text"))
        assert len(bm25_search(store, collection, "zebra", 10)) == 1
        put(store, collection, "a.md", chunk(0, "new giraffe text"))
        assert bm25_search(store, collection, "zebra", 10) == []
        assert len(bm25_search(store, collection, "giraffe", 10)) == 1

    def test_deleting_a_document_removes_its_entries(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        put(store, collection, "a.md", chunk(0, "zebra"), chunk(1, "zebra again"))
        store.delete_document(collection, "a.md")
        assert bm25_search(store, collection, "zebra", 10) == []

    def test_updating_a_chunk_reindexes_it(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        [chunk_id] = put(store, collection, "a.md", chunk(0, "zebra"))
        store._db.execute("UPDATE chunks SET text = 'giraffe' WHERE id = ?", (chunk_id,))
        assert bm25_search(store, collection, "zebra", 10) == []
        assert [i for i, _ in bm25_search(store, collection, "giraffe", 10)] == [chunk_id]

    def test_the_index_passes_its_own_integrity_check(self, store: Store) -> None:
        collection = store.ensure_collection("docs")
        put(store, collection, "a.md", chunk(0, "alpha"), chunk(1, "beta"))
        put(store, collection, "a.md", chunk(0, "gamma"))
        store.delete_document(collection, "a.md")
        put(store, collection, "b.md", chunk(0, "delta"))
        store._db.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('integrity-check')")

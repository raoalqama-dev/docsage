from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from docsage.ingest.pipeline import ingest
from docsage.retrieval.dense import HashingEmbedder
from docsage.retrieval.search import Hit, Mode, search
from docsage.store import Collection, Store
from embedders import ConceptEmbedder


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "docsage.db") as opened:
        yield opened


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two documents a search must find by different means, among unrelated ones.

    ``network.md`` has a rare token that no embedding model knows, and shares no
    concept with the queries below. ``dealership.md`` shares no word with them
    but is about the same thing as "car purchase". ``cats.md`` likewise for "sleep".
    """
    root = tmp_path / "corpus"
    root.mkdir()
    files = {
        "network.md": (
            "# Troubleshooting\n\nThe worker logs ECONNRESET whenever the peer closes "
            "the socket early. Retry with backoff."
        ),
        "dealership.md": "# Showroom\n\nOur dealership sells automobiles to local customers.",
        "cats.md": "# Pets\n\nKittens nap all afternoon in the sun.",
        "cooking.md": "# Pasta\n\nBoil water, add salt, then stir the noodles gently.",
        "garden.md": "# Roses\n\nPrune the roses in early spring and water them weekly.",
        "raft.md": "# Raft\n\nA leader replicates its log to the followers of the cluster.",
    }
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def docs(store: Store, corpus: Path) -> Collection:
    ingest(store, "docs", [corpus])
    collection = store.collection("docs")
    assert collection is not None
    return collection


def sources(result_hits: list[Hit]) -> list[str]:
    return [hit.chunk.source for hit in result_hits]


QUERY = "ECONNRESET car purchase"


class TestFixtureCorpus:
    """The feature's acceptance test: hybrid finds what only one retriever can."""

    def test_bm25_finds_the_keyword_match_but_not_the_paraphrase(
        self, store: Store, docs: Collection
    ) -> None:
        found = sources(search(store, docs, QUERY, mode=Mode.BM25).hits)
        assert found == ["network.md"]

    def test_dense_finds_the_paraphrase_but_not_the_keyword_match(
        self, store: Store, docs: Collection
    ) -> None:
        found = sources(
            search(store, docs, QUERY, mode=Mode.DENSE, embedder=ConceptEmbedder()).hits
        )
        assert found == ["dealership.md"]

    def test_hybrid_finds_both(self, store: Store, docs: Collection) -> None:
        result = search(store, docs, QUERY, mode=Mode.HYBRID, k=2, embedder=ConceptEmbedder())
        assert set(sources(result.hits)) == {"network.md", "dealership.md"}

    def test_a_keyword_only_query_in_hybrid_mode(self, store: Store, docs: Collection) -> None:
        hits = search(store, docs, "ECONNRESET", k=3, embedder=ConceptEmbedder()).hits
        assert sources(hits) == ["network.md"]
        assert (hits[0].bm25_rank, hits[0].dense_rank) == (1, None)

    def test_a_paraphrase_only_query_in_hybrid_mode(self, store: Store, docs: Collection) -> None:
        hits = search(store, docs, "slumber", k=3, embedder=ConceptEmbedder()).hits
        assert sources(hits) == ["cats.md"]
        assert (hits[0].bm25_rank, hits[0].dense_rank) == (None, 1)

    def test_the_real_hashing_embedder_also_finds_keyword_matches_in_dense_mode(
        self, store: Store, docs: Collection
    ) -> None:
        hits = search(store, docs, "leader log", mode=Mode.DENSE, embedder=HashingEmbedder()).hits
        assert sources(hits)[0] == "raft.md"


class TestHits:
    def test_hybrid_hits_carry_both_ranks_and_the_fused_score(
        self, store: Store, docs: Collection
    ) -> None:
        # "car purchase" and "buy" both match dealership by concept; network by token.
        hits = search(store, docs, "ECONNRESET buy cars", embedder=ConceptEmbedder()).hits
        by_source = {hit.chunk.source: hit for hit in hits}
        assert by_source["network.md"].bm25_rank == 1
        assert by_source["network.md"].dense_rank is None
        assert by_source["dealership.md"].bm25_rank is None
        assert by_source["dealership.md"].dense_rank == 1
        assert by_source["network.md"].score == pytest.approx(1 / 61)
        assert by_source["dealership.md"].score == pytest.approx(1 / 61)

    def test_a_chunk_found_by_both_scores_the_sum(self, store: Store, docs: Collection) -> None:
        hits = search(store, docs, "kittens nap", embedder=ConceptEmbedder()).hits
        assert sources(hits) == ["cats.md"]
        assert (hits[0].bm25_rank, hits[0].dense_rank) == (1, 1)
        assert hits[0].score == pytest.approx(2 / 61)

    def test_bm25_and_dense_modes_report_their_own_scores(
        self, store: Store, docs: Collection
    ) -> None:
        embedder = ConceptEmbedder()
        [bm25] = search(store, docs, "kittens", mode=Mode.BM25).hits
        [dense] = search(store, docs, "kittens", mode=Mode.DENSE, embedder=embedder).hits
        assert bm25.score > 0
        assert (bm25.bm25_rank, bm25.dense_rank) == (1, None)
        assert dense.score == pytest.approx(0.5**0.5)  # kittens: 1 of the 2 concepts in the chunk
        assert (dense.bm25_rank, dense.dense_rank) == (None, 1)

    def test_a_hit_carries_its_chunk(self, store: Store, docs: Collection) -> None:
        [hit] = search(store, docs, "ECONNRESET", mode=Mode.BM25).hits
        assert hit.chunk.source == "network.md"
        assert hit.chunk.chunk.heading_path == ("Troubleshooting",)
        assert "ECONNRESET" in hit.chunk.chunk.text


class TestOptions:
    def test_k_limits_the_results(self, store: Store, docs: Collection) -> None:
        hits = search(store, docs, "the", mode=Mode.BM25, k=2).hits
        assert len(hits) == 2
        assert len(search(store, docs, "the", mode=Mode.BM25, k=10).hits) > 2

    def test_weights_decide_which_retriever_wins_a_tie(
        self, store: Store, docs: Collection
    ) -> None:
        embedder = ConceptEmbedder()
        query = "ECONNRESET sleep"
        even = sources(search(store, docs, query, k=2, embedder=embedder).hits)
        assert even == ["network.md", "cats.md"]  # equal scores, the first list wins
        dense_heavy = search(store, docs, query, k=2, embedder=embedder, dense_weight=3.0)
        assert sources(dense_heavy.hits) == ["cats.md", "network.md"]
        assert dense_heavy.hits[0].score == pytest.approx(3 / 61)
        bm25_heavy = search(store, docs, query, k=2, embedder=embedder, bm25_weight=3.0)
        assert sources(bm25_heavy.hits) == ["network.md", "cats.md"]

    def test_a_zero_weight_leaves_a_retriever_out(self, store: Store, docs: Collection) -> None:
        embedder = ConceptEmbedder()
        assert sources(search(store, docs, QUERY, embedder=embedder, dense_weight=0).hits) == [
            "network.md"
        ]
        assert sources(search(store, docs, QUERY, embedder=embedder, bm25_weight=0).hits) == [
            "dealership.md"
        ]
        assert (
            search(store, docs, QUERY, embedder=embedder, bm25_weight=0, dense_weight=0).hits == []
        )

    def test_a_bm25_only_hybrid_search_needs_no_embedder(
        self, store: Store, docs: Collection
    ) -> None:
        result = search(store, docs, "ECONNRESET", dense_weight=0)
        assert sources(result.hits) == ["network.md"]
        assert result.embedded == 0

    def test_rrf_k_sets_the_score_scale(self, store: Store, docs: Collection) -> None:
        [hit] = search(store, docs, "ECONNRESET", embedder=ConceptEmbedder(), rrf_k=0).hits
        assert hit.score == pytest.approx(1.0)

    def test_dense_and_hybrid_need_an_embedder(self, store: Store, docs: Collection) -> None:
        for mode in (Mode.DENSE, Mode.HYBRID):
            with pytest.raises(ValueError, match=f"{mode.value} search needs an embedder"):
                search(store, docs, "x", mode=mode)

    def test_k_must_be_positive(self, store: Store, docs: Collection) -> None:
        with pytest.raises(ValueError, match="k must be at least 1"):
            search(store, docs, "x", k=0)


class TestEmbeddingOnDemand:
    def test_chunks_are_embedded_by_the_first_search_only(
        self, store: Store, docs: Collection
    ) -> None:
        embedder = ConceptEmbedder()
        first = search(store, docs, "cats", embedder=embedder)
        assert first.embedded == 6
        second = search(store, docs, "cats", embedder=embedder)
        assert second.embedded == 0
        assert embedder.document_calls == 1

    def test_new_documents_are_embedded_when_next_searched(
        self, store: Store, docs: Collection, corpus: Path
    ) -> None:
        embedder = ConceptEmbedder()
        search(store, docs, "cats", embedder=embedder)
        (corpus / "more.md").write_text("Felines sleep a lot.", encoding="utf-8")
        ingest(store, "docs", [corpus])
        result = search(store, docs, "cats sleep", embedder=embedder)
        assert result.embedded == 1
        assert sources(result.hits)[0] == "more.md"

    def test_bm25_mode_embeds_nothing(self, store: Store, docs: Collection) -> None:
        assert search(store, docs, "cats", mode=Mode.BM25).embedded == 0

    def test_collections_are_searched_separately(
        self, store: Store, docs: Collection, tmp_path: Path
    ) -> None:
        other_root = tmp_path / "other"
        other_root.mkdir()
        (other_root / "x.md").write_text("Kittens elsewhere.", encoding="utf-8")
        ingest(store, "other", [other_root])
        other = store.collection("other")
        assert other is not None
        embedder = ConceptEmbedder()
        assert sources(search(store, other, "kittens", embedder=embedder).hits) == ["x.md"]
        assert sources(search(store, docs, "kittens", embedder=embedder).hits) == ["cats.md"]


class TestEmptyAndOddQueries:
    @pytest.mark.parametrize("mode", list(Mode))
    @pytest.mark.parametrize("query", ["", "   ", "!!! ???", "()"])
    def test_a_query_without_words_finds_nothing(
        self, store: Store, docs: Collection, mode: Mode, query: str
    ) -> None:
        assert search(store, docs, query, mode=mode, embedder=ConceptEmbedder()).hits == []

    @pytest.mark.parametrize("mode", list(Mode))
    def test_fts_syntax_cannot_break_any_mode(
        self, store: Store, docs: Collection, mode: Mode
    ) -> None:
        for query in ('"unbalanced', "NEAR(a b)", "title: x*", "a AND (", "-x OR"):
            search(store, docs, query, mode=mode, embedder=ConceptEmbedder())

    def test_an_empty_collection(self, store: Store) -> None:
        empty = store.ensure_collection("empty")
        for mode in Mode:
            result = search(store, empty, "anything", mode=mode, embedder=ConceptEmbedder())
            assert result.hits == []

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from docsage.ingest.pipeline import ingest
from docsage.retrieval.retrieve import MIN_RERANKED, retrieve
from docsage.retrieval.search import Hit, Mode, search
from docsage.store import Collection, Store
from embedders import ConceptEmbedder
from rerankers import FixedReranker, OverlapReranker


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "docsage.db") as opened:
        yield opened


@pytest.fixture
def docs(store: Store, tmp_path: Path) -> Collection:
    """Three pieces about sleeping cats that say nearly the same, and two other topics."""
    root = tmp_path / "corpus"
    root.mkdir()
    files = {
        "cats1.md": "# Cats\n\nKittens nap in the sun all afternoon.",
        "cats2.md": "# Kittens\n\nKittens nap in the sun all afternoon long.",
        "cats3.md": "# Felines\n\nA cat naps in the sun all afternoon.",
        "cars.md": "# Showroom\n\nOur dealership sells automobiles to local customers.",
        "raft.md": "# Raft\n\nA leader replicates its log to the followers of the cluster.",
    }
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    ingest(store, "docs", [root])
    collection = store.collection("docs")
    assert collection is not None
    return collection


def sources(hits: list[Hit]) -> list[str]:
    return [hit.chunk.source for hit in hits]


QUERY = "kittens nap automobiles"


class TestWithoutLaterStages:
    def test_is_the_search(self, store: Store, docs: Collection) -> None:
        plain = search(store, docs, QUERY, mode=Mode.BM25, k=3)
        result = retrieve(store, docs, QUERY, mode=Mode.BM25, k=3)
        assert result.hits == plain.hits
        assert (result.candidates, result.below_threshold) == (3, 0)

    def test_hybrid_arguments_are_passed_on(self, store: Store, docs: Collection) -> None:
        result = retrieve(
            store, docs, QUERY, k=2, embedder=ConceptEmbedder(), bm25_weight=0, dense_weight=1
        )
        assert all(hit.bm25_rank is None for hit in result.hits)
        assert result.embedded == 5


class TestRerank:
    def test_hits_are_reordered_by_the_reranker(self, store: Store, docs: Collection) -> None:
        first_stage = search(store, docs, "sells nap sun afternoon", mode=Mode.BM25, k=5).hits
        result = retrieve(
            store,
            docs,
            "sells nap sun afternoon",
            mode=Mode.BM25,
            k=5,
            reranker=OverlapReranker(),
        )
        assert sorted(sources(result.hits)) == sorted(sources(first_stage))
        scores = [hit.score for hit in result.hits]
        assert scores == sorted(scores, reverse=True)
        assert {hit.first_rank for hit in result.hits} == set(range(1, len(first_stage) + 1))

    def test_the_reranker_can_promote_what_the_search_ranked_low(
        self, store: Store, docs: Collection
    ) -> None:
        # BM25 ranks the pieces with "nap" and "sun" above the one with the rarer "sells".
        query = "sells nap sun"
        searched = search(store, docs, query, mode=Mode.BM25, k=5).hits
        scores = [1.0 if hit.chunk.source == "cars.md" else 0.0 for hit in searched]
        result = retrieve(store, docs, query, mode=Mode.BM25, k=5, reranker=FixedReranker(scores))
        assert sources(result.hits)[0] == "cars.md"
        assert result.hits[0].first_rank == sources(searched).index("cars.md") + 1

    def test_the_reranker_sees_only_the_candidates(self, store: Store, docs: Collection) -> None:
        reranker = OverlapReranker()
        result = retrieve(
            store, docs, "nap sun", mode=Mode.BM25, k=1, reranker=reranker, candidates=2
        )
        assert len(reranker.calls[0][1]) == 2
        assert len(result.hits) == 1
        assert result.candidates == 2

    def test_candidates_default_to_three_times_k_and_at_least_30(
        self, store: Store, tmp_path: Path
    ) -> None:
        root = tmp_path / "many"
        root.mkdir()
        for number in range(40):
            (root / f"d{number}.md").write_text(
                f"# D{number}\n\nThe word nap, number {number}.", "utf-8"
            )
        ingest(store, "many", [root])
        many = store.collection("many")
        assert many is not None
        for k, expected in [(1, MIN_RERANKED), (20, 40)]:
            reranker = OverlapReranker()
            result = retrieve(store, many, "nap", mode=Mode.BM25, k=k, reranker=reranker)
            assert len(reranker.calls[0][1]) == expected
            assert len(result.hits) == k


class TestMinimumScore:
    def test_the_rerankers_score_is_compared(self, store: Store, docs: Collection) -> None:
        result = retrieve(
            store,
            docs,
            "kittens nap sun",
            mode=Mode.BM25,
            k=5,
            reranker=OverlapReranker(),
            min_score=3,
        )
        assert sorted(sources(result.hits)) == ["cats1.md", "cats2.md"]
        assert all(hit.score >= 3 for hit in result.hits)
        assert result.below_threshold == result.candidates - 2

    def test_the_searchs_score_is_compared_without_a_reranker(
        self, store: Store, docs: Collection
    ) -> None:
        everything = retrieve(store, docs, "nap sun leader", mode=Mode.BM25, k=5).hits
        cut = everything[1].score
        result = retrieve(store, docs, "nap sun leader", mode=Mode.BM25, k=5, min_score=cut)
        assert len(result.hits) == len([h for h in everything if h.score >= cut])

    def test_a_score_equal_to_the_minimum_is_kept(self, store: Store, docs: Collection) -> None:
        result = retrieve(
            store,
            docs,
            "nap",
            mode=Mode.BM25,
            k=5,
            reranker=FixedReranker([1.0] * 3),
            min_score=1.0,
        )
        assert len(result.hits) == 3

    def test_nothing_may_pass(self, store: Store, docs: Collection) -> None:
        result = retrieve(
            store, docs, "nap", mode=Mode.BM25, k=5, reranker=OverlapReranker(), min_score=99
        )
        assert result.hits == []
        assert result.below_threshold == 3


class TestMmr:
    def test_it_swaps_a_duplicate_for_something_different(
        self, store: Store, docs: Collection
    ) -> None:
        embedder = ConceptEmbedder()
        plain = retrieve(store, docs, "cat nap car", mode=Mode.DENSE, k=2, embedder=embedder)
        diverse = retrieve(
            store, docs, "cat nap car", mode=Mode.DENSE, k=2, embedder=embedder, mmr_lambda=0.3
        )
        assert set(sources(plain.hits)) <= {"cats1.md", "cats2.md", "cats3.md"}
        assert "cars.md" in sources(diverse.hits)
        assert sources(diverse.hits)[0] == sources(plain.hits)[0]

    def test_lambda_1_keeps_the_ranking(self, store: Store, docs: Collection) -> None:
        embedder = ConceptEmbedder()
        plain = retrieve(store, docs, "cat nap car", mode=Mode.DENSE, k=3, embedder=embedder)
        kept = retrieve(
            store, docs, "cat nap car", mode=Mode.DENSE, k=3, embedder=embedder, mmr_lambda=1.0
        )
        assert kept.hits == plain.hits[:3]

    def test_it_works_on_a_bm25_search_with_an_embedder(
        self, store: Store, docs: Collection
    ) -> None:
        embedder = ConceptEmbedder()
        result = retrieve(
            store, docs, "nap sun afternoon", mode=Mode.BM25, k=2, embedder=embedder, mmr_lambda=0.5
        )
        assert len(result.hits) == 2
        assert result.embedded == 5

    def test_it_needs_an_embedder(self, store: Store, docs: Collection) -> None:
        with pytest.raises(ValueError, match="MMR needs an embedder"):
            retrieve(store, docs, "nap", mode=Mode.BM25, mmr_lambda=0.5)

    def test_it_follows_the_rerankers_scores(self, store: Store, docs: Collection) -> None:
        searched = search(store, docs, "nap sun", mode=Mode.BM25, k=5).hits
        scores = [1.0 if hit.chunk.source == "cats3.md" else 0.5 for hit in searched]
        result = retrieve(
            store,
            docs,
            "nap sun",
            mode=Mode.BM25,
            k=1,
            embedder=ConceptEmbedder(),
            reranker=FixedReranker(scores),
            mmr_lambda=0.7,
        )
        assert sources(result.hits) == ["cats3.md"]

    def test_no_hits_to_diversify(self, store: Store, docs: Collection) -> None:
        result = retrieve(
            store, docs, "zzz", mode=Mode.BM25, k=2, embedder=ConceptEmbedder(), mmr_lambda=0.5
        )
        assert result.hits == []


class TestValidation:
    def test_k(self, store: Store, docs: Collection) -> None:
        with pytest.raises(ValueError, match="k must be at least 1"):
            retrieve(store, docs, "nap", mode=Mode.BM25, k=0)

    def test_candidates_must_cover_k(self, store: Store, docs: Collection) -> None:
        with pytest.raises(ValueError, match=r"candidates \(2\) must be at least k \(3\)"):
            retrieve(store, docs, "nap", mode=Mode.BM25, k=3, candidates=2)

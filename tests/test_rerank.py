from __future__ import annotations

import inspect
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from docsage.ingest.pipeline import ingest
from docsage.retrieval.dense import Vectors
from docsage.retrieval.rerank import (
    DEFAULT_RERANK_MODEL,
    CrossEncoderReranker,
    RerankerError,
    create_reranker,
    mmr,
    passage_text,
    rerank,
)
from docsage.retrieval.search import Hit, Mode, search
from docsage.store import Store
from rerankers import FakeCrossEncoderModel, FixedReranker, OverlapReranker


@pytest.fixture
def hits(tmp_path: Path) -> Iterator[list[Hit]]:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.md").write_text("# Raft\n\n## Elections\n\nA candidate asks for votes.", "utf-8")
    (root / "b.md").write_text("# Pasta\n\nBoil the noodles, then drain the votes.", "utf-8")
    (root / "c.md").write_text("# Cats\n\nKittens nap.", "utf-8")
    with Store(tmp_path / "docsage.db") as store:
        ingest(store, "docs", [root])
        collection = store.collection("docs")
        assert collection is not None
        yield search(store, collection, "votes candidate nap", mode=Mode.BM25, k=3).hits


class TestRerank:
    def test_orders_by_the_rerankers_score_and_remembers_the_first_rank(
        self, hits: list[Hit]
    ) -> None:
        assert len(hits) == 3
        reranked = rerank(FixedReranker([0.1, 0.9, 0.5]), "q", hits)
        assert [hit.chunk.id for hit in reranked] == [
            hits[1].chunk.id,
            hits[2].chunk.id,
            hits[0].chunk.id,
        ]
        assert [hit.score for hit in reranked] == [0.9, 0.5, 0.1]
        assert [hit.first_rank for hit in reranked] == [2, 3, 1]

    def test_keeps_the_other_ranks(self, hits: list[Hit]) -> None:
        reranked = rerank(FixedReranker([0.0, 1.0, 0.0]), "q", hits)
        assert reranked[0].bm25_rank == hits[1].bm25_rank

    def test_equal_scores_keep_their_order(self, hits: list[Hit]) -> None:
        reranked = rerank(FixedReranker([1.0, 1.0, 1.0]), "q", hits)
        assert [hit.chunk.id for hit in reranked] == [hit.chunk.id for hit in hits]

    def test_the_reranker_reads_the_headings_and_the_text(self, hits: list[Hit]) -> None:
        reranker = OverlapReranker()
        rerank(reranker, "votes", hits)
        query, passages = reranker.calls[0]
        assert query == "votes"
        assert passages == [passage_text(hit) for hit in hits]
        assert any(p == "Raft\nElections\nA candidate asks for votes." for p in passages)

    def test_a_real_scorer_changes_the_order(self, hits: list[Hit]) -> None:
        reranked = rerank(OverlapReranker(), "kittens nap", hits)
        assert reranked[0].chunk.source == "c.md"

    def test_no_hits_is_not_scored(self) -> None:
        reranker = OverlapReranker()
        assert rerank(reranker, "q", []) == []
        assert reranker.calls == []

    def test_a_wrong_number_of_scores_is_an_error(self, hits: list[Hit]) -> None:
        with pytest.raises(RerankerError, match="fixed-test returned 1 scores for 3 hits"):
            rerank(FixedReranker([1.0]), "q", hits)


class TestCrossEncoderReranker:
    def test_scores_come_from_the_model(self) -> None:
        model = FakeCrossEncoderModel()
        reranker = CrossEncoderReranker("some/model", model=model)
        assert reranker.score("q", ["a", "abc"]) == [1.0, 3.0]
        assert model.calls == [("q", ["a", "abc"])]
        assert reranker.name == "cross-encoder-some/model"

    def test_no_passages_need_no_model(self) -> None:
        model = FakeCrossEncoderModel()
        assert CrossEncoderReranker(model=model).score("q", []) == []
        assert model.calls == []

    def test_the_model_is_created_on_first_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created: list[str] = []

        class TextCrossEncoder(FakeCrossEncoderModel):
            def __init__(self, model_name: str) -> None:
                super().__init__()
                created.append(model_name)

        module = types.ModuleType("fastembed.rerank.cross_encoder")
        module.TextCrossEncoder = TextCrossEncoder  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", module)
        reranker = CrossEncoderReranker("org/small")
        assert created == []
        reranker.score("q", ["x"])
        reranker.score("q", ["y"])
        assert created == ["org/small"]

    def test_a_missing_extra_is_explained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", None)
        with pytest.raises(RerankerError, match=r"pip install 'docsage\[rerank\]'"):
            CrossEncoderReranker().score("q", ["x"])

    def test_a_model_that_cannot_load_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(model_name: str) -> None:
            raise ValueError(f"model {model_name} is not supported")

        module = types.ModuleType("fastembed.rerank.cross_encoder")
        module.TextCrossEncoder = broken  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", module)
        with pytest.raises(RerankerError, match="cannot load nope: model nope is not supported"):
            CrossEncoderReranker("nope").score("q", ["x"])

    def test_create_reranker(self) -> None:
        assert create_reranker().name == f"cross-encoder-{DEFAULT_RERANK_MODEL}"
        assert create_reranker("org/m").name == "cross-encoder-org/m"


class TestRealFastEmbed:
    """Checks the installed fastembed against what the reranker expects; no model is loaded."""

    def test_the_library_offers_the_api_the_reranker_uses(self) -> None:
        cross_encoder = pytest.importorskip("fastembed.rerank.cross_encoder")
        encoder = cross_encoder.TextCrossEncoder
        assert callable(encoder.rerank)
        assert "model_name" in inspect.signature(encoder).parameters
        supported = {model["model"] for model in encoder.list_supported_models()}
        assert DEFAULT_RERANK_MODEL in supported


def unit(*rows: tuple[float, ...]) -> Vectors:
    matrix = np.array(rows, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    scaled: Vectors = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return scaled


class TestMmr:
    #: Items 0 and 1 are the same direction; item 2 is orthogonal to them.
    vectors = unit((1, 0), (1, 0), (0, 1))

    def test_lambda_1_is_plain_ranking(self) -> None:
        assert mmr([3.0, 2.0, 1.0], self.vectors, 3, 1.0) == [0, 1, 2]

    def test_lower_lambda_prefers_variety_over_a_duplicate(self) -> None:
        # Item 1 is almost as relevant as item 0 but identical to it.
        assert mmr([1.0, 0.9, 0.5], self.vectors, 2, 0.5) == [0, 2]

    def test_the_first_pick_is_always_the_most_relevant(self) -> None:
        assert mmr([0.2, 0.9, 0.1], self.vectors, 1, 0.0) == [1]

    def test_hand_computed_values(self) -> None:
        # Scaled relevance is [1, 0.8, 0]; after item 0, item 1 is worth
        # 0.5 * 0.8 - 0.5 * 1 = -0.1 and item 2 is worth 0.5 * 0 - 0 = 0.
        assert mmr([1.0, 0.9, 0.5], self.vectors, 3, 0.5) == [0, 2, 1]
        # With lambda 0.7: item 1 is 0.7 * 0.8 - 0.3 = 0.26 and item 2 is 0.
        assert mmr([1.0, 0.9, 0.5], self.vectors, 3, 0.7) == [0, 1, 2]

    def test_scores_of_any_scale_behave_alike(self) -> None:
        base = mmr([1.0, 0.9, 0.5], self.vectors, 3, 0.5)
        assert mmr([100.0, 90.0, 50.0], self.vectors, 3, 0.5) == base
        assert mmr([-1.0, -1.1, -1.5], self.vectors, 3, 0.5) == base

    def test_negative_similarity_is_no_bonus(self) -> None:
        opposite = unit((1, 0), (-1, 0), (0, 1))
        assert mmr([1.0, 0.9, 0.0], opposite, 3, 0.5) == [0, 1, 2]

    def test_equal_relevance_picks_the_first_and_then_the_most_different(self) -> None:
        assert mmr([1.0, 1.0, 1.0], self.vectors, 3, 0.5) == [0, 2, 1]

    def test_fewer_items_than_k(self) -> None:
        assert mmr([1.0, 0.5], unit((1, 0), (0, 1)), 5) == [0, 1]

    def test_no_items(self) -> None:
        assert mmr([], np.zeros((0, 2), dtype=np.float32), 3) == []

    @pytest.mark.parametrize("lambda_", [-0.1, 1.1])
    def test_lambda_must_be_a_fraction(self, lambda_: float) -> None:
        with pytest.raises(ValueError, match="lambda must be between 0 and 1"):
            mmr([1.0], unit((1, 0)), 1, lambda_)

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="k must be at least 1"):
            mmr([1.0], unit((1, 0)), 0)

    def test_one_vector_per_item(self) -> None:
        with pytest.raises(ValueError, match="1 vectors for 2 items"):
            mmr([1.0, 2.0], unit((1, 0)), 1)

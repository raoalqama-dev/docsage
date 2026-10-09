from __future__ import annotations

import pytest

from docsage.retrieval.fusion import DEFAULT_K, reciprocal_rank_fusion


def test_default_k_is_60() -> None:
    assert DEFAULT_K == 60


def test_scores_match_hand_computation() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [item for item, _ in fused] == ["a", "c", "b", "d"]


def test_k_changes_the_scores() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b"]], k=0))
    assert fused == {"a": pytest.approx(1.0), "b": pytest.approx(0.5)}


def test_weights_scale_each_list() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], [3.0, 1.0])
    assert [item for item, _ in fused] == ["a", "b"]
    assert dict(fused)["a"] == pytest.approx(3 / 61 + 1 / 62)
    flipped = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], [1.0, 3.0])
    assert [item for item, _ in flipped] == ["b", "a"]


def test_a_zero_weight_adds_nothing_but_still_lists_the_items() -> None:
    fused = reciprocal_rank_fusion([["a"], ["b"]], [1.0, 0.0])
    assert fused == [("a", pytest.approx(1 / 61)), ("b", 0.0)]


def test_ties_go_to_the_best_rank_then_to_the_first_seen() -> None:
    # a and b both score 1/61 + 1/62; c and d score the same as each other.
    assert [i for i, _ in reciprocal_rank_fusion([["a", "b"], ["b", "a"]])] == ["a", "b"]
    assert [i for i, _ in reciprocal_rank_fusion([["x", "c"], ["d", "y"]])] == ["x", "d", "c", "y"]


def test_a_repeated_item_counts_at_its_first_position() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "a"]]))
    assert fused["a"] == pytest.approx(1 / 61)


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_ids_may_be_ints() -> None:
    assert [i for i, _ in reciprocal_rank_fusion([[3, 1], [1]])] == [1, 3]


@pytest.mark.parametrize(
    ("rankings", "weights", "k", "message"),
    [
        ([["a"]], [1.0, 2.0], 60, "2 weights for 1 rankings"),
        ([["a"]], [-1.0], 60, "weights must not be negative"),
        ([["a"]], None, -1, "k must not be negative"),
    ],
)
def test_invalid_arguments(
    rankings: list[list[str]], weights: list[float] | None, k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion(rankings, weights, k=k)

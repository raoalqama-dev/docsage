"""Reciprocal rank fusion of ranked lists."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

#: The constant from the original paper; larger values flatten the weight of top ranks.
DEFAULT_K = 60

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]],
    weights: Sequence[float] | None = None,
    *,
    k: int = DEFAULT_K,
) -> list[tuple[T, float]]:
    """Merge ranked lists into one: an item scores ``sum(weight / (k + rank))``.

    Ranks start at 1. Only positions matter, so scores from different
    retrievers (BM25, cosine similarity) need no calibration. An item missing
    from a list gains nothing from it, and an item repeated within a list
    counts at its first position. Ties go to the item that ranks best in any
    list, then to the one that appeared first.
    """
    if k < 0:
        raise ValueError(f"k must not be negative, not {k}")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"{len(weights)} weights for {len(rankings)} rankings")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must not be negative")
    scores: dict[T, float] = {}
    best: dict[T, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        counted: set[T] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in counted:
                continue
            counted.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best[item] = min(best.get(item, rank), rank)
    order = {item: position for position, item in enumerate(scores)}
    ranked = sorted(scores, key=lambda item: (-scores[item], best[item], order[item]))
    return [(item, scores[item]) for item in ranked]

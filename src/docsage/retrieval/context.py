"""Assembling retrieved chunks into context for a language model, within a token budget.

The chunks are taken in rank order. A chunk that nearly repeats one already
taken is collapsed into it, and one too big for what is left of the budget is
skipped. Any budget still unspent goes to the chunks next to the ones taken, so
a passage is not cut off mid-thought. Chunks that follow one another in a
document are then merged into a single passage, with the text the chunker
repeated between them removed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from docsage.ingest.chunking import count_tokens
from docsage.retrieval.search import Hit
from docsage.store import Store, StoredChunk

#: Chunks sharing at least this fraction of the shorter one's word triples are duplicates.
DEFAULT_DUPLICATE_THRESHOLD = 0.9

_WORD_RE = re.compile(r"[^\W_]+")
_SHINGLE_WORDS = 3
_MIN_OVERLAP = 12

Shingles = frozenset[tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class Passage:
    """One or more consecutive chunks of a document, as the model will read them."""

    source: str
    title: str
    heading_path: tuple[str, ...]
    page: int | None
    page_end: int | None
    text: str
    chunk_ids: tuple[int, ...]
    """The chunks it is made of, in document order."""
    hit_ids: tuple[int, ...]
    """Which of those the search found, the others being added as neighbours."""
    tokens: int


@dataclass(frozen=True, slots=True)
class Context:
    passages: list[Passage]
    """Best first: ordered by the best-ranked hit each contains."""
    tokens: int
    budget: int
    duplicates: int
    """Hits dropped for nearly repeating a better-ranked one."""
    skipped: int
    """Hits dropped because they did not fit in what was left of the budget."""


def _shingles(text: str) -> Shingles:
    words = _WORD_RE.findall(text.lower())
    if len(words) <= _SHINGLE_WORDS:
        return frozenset({tuple(words)})
    return frozenset(
        tuple(words[start : start + _SHINGLE_WORDS])
        for start in range(len(words) - _SHINGLE_WORDS + 1)
    )


def _similarity(a: Shingles, b: Shingles) -> float:
    """The share of the smaller set that the other contains."""
    return len(a & b) / min(len(a), len(b))


def _join(previous: str, following: str) -> str:
    """Two consecutive chunks as one text, without the sentences the second repeats.

    The chunker starts a chunk with the end of the one before it. That repeat
    is found as the longest start of ``following`` that ends ``previous`` on
    word boundaries on both sides, if it is at least ``_MIN_OVERLAP`` characters:
    a shorter match is more likely a word that happens to end one chunk and
    start the next. When there is none the chunks are joined as paragraphs.
    """
    for length in range(min(len(previous), len(following)), _MIN_OVERLAP - 1, -1):
        if (
            previous.endswith(following[:length])
            and (length == len(previous) or previous[-length - 1].isspace())
            and (length == len(following) or following[length].isspace())
        ):
            return previous + following[length:]
    return previous + "\n\n" + following


def _runs(chunks: Sequence[StoredChunk]) -> list[list[StoredChunk]]:
    """Group a document's chunks into runs of consecutive chunks of one section."""
    runs: list[list[StoredChunk]] = []
    for chunk in sorted(chunks, key=lambda c: c.chunk.ordinal):
        last = runs[-1][-1] if runs else None
        if (
            last is not None
            and chunk.chunk.ordinal == last.chunk.ordinal + 1
            and chunk.chunk.heading_path == last.chunk.heading_path
        ):
            runs[-1].append(chunk)
        else:
            runs.append([chunk])
    return runs


def _passage(chunks: Sequence[StoredChunk], hit_ids: set[int]) -> Passage:
    text = chunks[0].chunk.text
    for chunk in chunks[1:]:
        text = _join(text, chunk.chunk.text)
    pages = [c.chunk.page for c in chunks if c.chunk.page is not None]
    page_ends = [c.chunk.page_end for c in chunks if c.chunk.page_end is not None]
    first = chunks[0]
    return Passage(
        source=first.source,
        title=first.title,
        heading_path=first.chunk.heading_path,
        page=min(pages) if pages else None,
        page_end=max(page_ends) if page_ends else None,
        text=text,
        chunk_ids=tuple(c.id for c in chunks),
        hit_ids=tuple(c.id for c in chunks if c.id in hit_ids),
        tokens=count_tokens(text),
    )


def assemble_context(
    store: Store,
    hits: Sequence[Hit],
    budget: int,
    *,
    neighbours: int = 1,
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> Context:
    """The passages to show a model for ``hits``, in at most ``budget`` tokens.

    Up to ``neighbours`` chunks on each side of a hit are added while budget
    remains, nearest first and best-ranked hit first, but only chunks of the
    same section, so a passage never crosses a heading. Chunks that follow
    each other in a document overlap by design and are merged, never collapsed
    as duplicates of one another; any other pair that shares at least
    ``duplicate_threshold`` of the shorter one's word triples is a duplicate,
    and only the better-ranked is kept.

    The budget is spent chunk by chunk on the chunks' own token counts. Merging
    removes repeated text, so a passage can come out shorter than that, never
    longer: ``Context.tokens`` is counted on the final text and is at most
    ``budget``.
    """
    if budget < 1:
        raise ValueError(f"the budget must be at least 1 token, not {budget}")
    if neighbours < 0:
        raise ValueError(f"neighbours must not be negative, not {neighbours}")
    if not 0.0 < duplicate_threshold <= 1.0:
        raise ValueError(f"the duplicate threshold must be in (0, 1], not {duplicate_threshold}")

    taken: dict[int, StoredChunk] = {}
    shingles: dict[int, Shingles] = {}
    rank: dict[int, int] = {}
    spent = 0
    duplicates = skipped = 0

    def is_duplicate(chunk: StoredChunk, own: Shingles) -> bool:
        for other in taken.values():
            # Chunks that follow each other overlap by design; they are merged instead.
            adjacent = (
                other.document_id == chunk.document_id
                and abs(other.chunk.ordinal - chunk.chunk.ordinal) == 1
            )
            if not adjacent and _similarity(own, shingles[other.id]) >= duplicate_threshold:
                return True
        return False

    for position, hit in enumerate(hits):
        chunk = hit.chunk
        if chunk.id in taken:
            continue
        own = _shingles(chunk.chunk.text)
        if is_duplicate(chunk, own):
            duplicates += 1
        elif spent + chunk.chunk.token_count > budget:
            skipped += 1
        else:
            taken[chunk.id] = chunk
            shingles[chunk.id] = own
            rank[chunk.id] = position
            spent += chunk.chunk.token_count
    hit_ids = set(taken)

    document_chunks: dict[int, dict[int, StoredChunk]] = {}

    def chunks_of(document_id: int) -> dict[int, StoredChunk]:
        if document_id not in document_chunks:
            document_chunks[document_id] = {c.chunk.ordinal: c for c in store.chunks(document_id)}
        return document_chunks[document_id]

    for _ in range(neighbours):
        for hit_id in sorted(hit_ids, key=rank.__getitem__):
            anchor = taken[hit_id]
            siblings = chunks_of(anchor.document_id)
            for step in (-1, 1):
                # Walk outwards over what is already taken, to the first chunk that is not.
                ordinal = anchor.chunk.ordinal + step
                while ordinal in siblings and siblings[ordinal].id in taken:
                    ordinal += step
                candidate = siblings.get(ordinal)
                if (
                    candidate is None
                    or candidate.chunk.heading_path != anchor.chunk.heading_path
                    or spent + candidate.chunk.token_count > budget
                ):
                    continue
                own = _shingles(candidate.chunk.text)
                if is_duplicate(candidate, own):
                    continue
                taken[candidate.id] = candidate
                shingles[candidate.id] = own
                spent += candidate.chunk.token_count

    passages: list[tuple[int, Passage]] = []
    by_document: dict[int, list[StoredChunk]] = {}
    for chunk in taken.values():
        by_document.setdefault(chunk.document_id, []).append(chunk)
    for chunks in by_document.values():
        for run in _runs(chunks):
            best = min((rank[c.id] for c in run if c.id in rank), default=len(hits))
            passages.append((best, _passage(run, hit_ids)))
    passages.sort(key=lambda item: item[0])
    ordered = [passage for _, passage in passages]
    return Context(ordered, sum(p.tokens for p in ordered), budget, duplicates, skipped)

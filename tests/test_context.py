from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from docsage.ingest.chunking import ChunkingOptions, count_tokens
from docsage.ingest.pipeline import ingest
from docsage.retrieval.context import Context, assemble_context
from docsage.retrieval.search import Hit
from docsage.store import Collection, Store, StoredChunk
from fixtures import prose

#: Chunks of about 62 tokens that start with the last two sentences of the one before.
OPTIONS = ChunkingOptions(max_tokens=80, overlap_tokens=30)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "docsage.db") as opened:
        yield opened


def build(store: Store, tmp_path: Path, files: dict[str, str]) -> Collection:
    root = tmp_path / "docs"
    root.mkdir(exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")
    ingest(store, "docs", [root], OPTIONS)
    collection = store.collection("docs")
    assert collection is not None
    return collection


def chunks_of(store: Store, collection: Collection, source: str) -> list[StoredChunk]:
    record = store.document(collection, source)
    assert record is not None
    return store.chunks(record.id)


def as_hits(*chunks: StoredChunk) -> list[Hit]:
    return [Hit(chunk, 1.0 / (rank + 1)) for rank, chunk in enumerate(chunks)]


def sentence_numbers(text: str) -> list[int]:
    return [int(number) for number in re.findall(r"Sentence (\d+) says", text)]


@pytest.fixture
def long_doc(store: Store, tmp_path: Path) -> list[StoredChunk]:
    """One section of 24 numbered sentences: ten chunks, each repeating the previous one's end."""
    collection = build(store, tmp_path, {"doc.md": "# Doc\n\n" + prose(24)})
    chunks = chunks_of(store, collection, "doc.md")
    assert len(chunks) == 10
    return chunks


class TestNeighbours:
    def test_a_hit_is_widened_with_the_chunks_around_it(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        context = assemble_context(store, as_hits(long_doc[4]), 1000)
        (passage,) = context.passages
        assert passage.chunk_ids == tuple(c.id for c in long_doc[3:6])
        assert passage.hit_ids == (long_doc[4].id,)

    def test_the_repeated_overlap_is_removed_when_chunks_are_merged(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        context = assemble_context(store, as_hits(long_doc[4]), 1000, neighbours=2)
        (passage,) = context.passages
        numbers = sentence_numbers(passage.text)
        assert numbers == sorted(set(numbers))
        assert numbers == list(range(numbers[0], numbers[-1] + 1))
        assert passage.text.startswith(long_doc[2].chunk.text)
        assert passage.text.endswith(long_doc[6].chunk.text)

    def test_neighbours_zero_adds_nothing(self, store: Store, long_doc: list[StoredChunk]) -> None:
        context = assemble_context(store, as_hits(long_doc[4]), 1000, neighbours=0)
        assert [p.chunk_ids for p in context.passages] == [(long_doc[4].id,)]
        assert context.passages[0].text == long_doc[4].chunk.text

    def test_the_ends_of_a_document_have_one_neighbour(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        context = assemble_context(store, as_hits(long_doc[0], long_doc[9]), 1000)
        assert [p.chunk_ids for p in context.passages] == [
            (long_doc[0].id, long_doc[1].id),
            (long_doc[8].id, long_doc[9].id),
        ]

    def test_neighbours_only_fill_the_budget_left_after_the_hits(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        hit = long_doc[4]
        tight = hit.chunk.token_count + long_doc[3].chunk.token_count
        context = assemble_context(store, as_hits(hit), tight)
        (passage,) = context.passages
        # Room for one neighbour: the earlier one is tried first.
        assert passage.chunk_ids == (long_doc[3].id, hit.id)

    def test_hits_come_before_the_neighbours_of_better_hits(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        first, second = long_doc[1], long_doc[7]
        budget = first.chunk.token_count + second.chunk.token_count
        context = assemble_context(store, as_hits(first, second), budget)
        assert [p.hit_ids for p in context.passages] == [(first.id,), (second.id,)]
        assert all(len(p.chunk_ids) == 1 for p in context.passages)

    def test_a_passage_does_not_cross_a_heading(self, store: Store, tmp_path: Path) -> None:
        text = "# Doc\n\n## One\n\n" + prose(4) + "\n\n## Two\n\n" + prose(4, start=100)
        collection = build(store, tmp_path, {"doc.md": text})
        chunks = chunks_of(store, collection, "doc.md")
        last_of_one = max(c.chunk.ordinal for c in chunks if c.chunk.heading_path[-1] == "One")
        context = assemble_context(store, as_hits(chunks[last_of_one]), 1000, neighbours=3)
        second = {c.id for c in chunks if c.chunk.heading_path[-1] == "Two"}
        assert second
        (passage,) = context.passages
        assert not second & set(passage.chunk_ids)
        assert passage.heading_path == ("Doc", "One")
        assert passage.chunk_ids[-1] == chunks[last_of_one].id

    def test_hits_in_different_sections_stay_separate_passages(
        self, store: Store, tmp_path: Path
    ) -> None:
        text = "# Doc\n\n## One\n\nAlpha beta gamma.\n\n## Two\n\nDelta epsilon zeta."
        collection = build(store, tmp_path, {"doc.md": text})
        one, two = chunks_of(store, collection, "doc.md")
        context = assemble_context(store, as_hits(one, two), 1000)
        assert [p.heading_path for p in context.passages] == [
            ("Doc", "One"),
            ("Doc", "Two"),
        ]

    def test_chunks_without_overlap_are_joined_as_paragraphs(
        self, store: Store, tmp_path: Path
    ) -> None:
        # Each paragraph nearly fills a chunk, so the chunker carries nothing over.
        paragraphs = [prose(1, start=n, words=25) for n in range(3)]
        collection = build(store, tmp_path, {"doc.md": "# Doc\n\n" + "\n\n".join(paragraphs)})
        chunks = chunks_of(store, collection, "doc.md")
        assert [c.chunk.text for c in chunks] == paragraphs
        context = assemble_context(store, as_hits(chunks[0], chunks[1]), 1000, neighbours=0)
        (passage,) = context.passages
        assert passage.text == paragraphs[0] + "\n\n" + paragraphs[1]

    def test_text_that_merely_ends_like_the_next_chunk_starts_is_kept(
        self, store: Store, tmp_path: Path
    ) -> None:
        # "alpha" ends the first paragraph and starts the second: a coincidence, not the overlap.
        paragraphs = [
            prose(1, start=0, words=25) + " alpha",
            "alpha " + prose(1, start=1, words=25),
        ]
        collection = build(store, tmp_path, {"doc.md": "# Doc\n\n" + "\n\n".join(paragraphs)})
        chunks = chunks_of(store, collection, "doc.md")
        context = assemble_context(store, as_hits(*chunks), 1000, neighbours=0)
        assert context.passages[0].text.count("alpha") == 2


class TestOrderAndMetadata:
    def test_passages_are_ordered_by_their_best_hit(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        context = assemble_context(store, as_hits(long_doc[9], long_doc[1], long_doc[5]), 1000)
        assert [p.hit_ids for p in context.passages] == [
            (long_doc[9].id,),
            (long_doc[1].id,),
            (long_doc[5].id,),
        ]

    def test_adjacent_hits_make_one_passage(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        context = assemble_context(store, as_hits(long_doc[5], long_doc[4]), 1000, neighbours=0)
        (passage,) = context.passages
        assert passage.chunk_ids == (long_doc[4].id, long_doc[5].id)
        assert passage.hit_ids == passage.chunk_ids
        assert sentence_numbers(passage.text) == sorted(set(sentence_numbers(passage.text)))

    def test_source_title_and_headings(self, store: Store, long_doc: list[StoredChunk]) -> None:
        (passage,) = assemble_context(store, as_hits(long_doc[2]), 1000).passages
        assert (passage.source, passage.title) == ("doc.md", "Doc")
        assert passage.heading_path == ("Doc",)
        assert (passage.page, passage.page_end) == (None, None)

    def test_pages_cover_the_whole_passage(self, store: Store, tmp_path: Path) -> None:
        from fixtures import make_pdf

        pages = [[[f"Page {n} sentence {i} about replicas."] for i in range(3)] for n in range(3)]
        root = tmp_path / "pdfs"
        root.mkdir()
        (root / "paper.pdf").write_bytes(make_pdf(pages))
        ingest(store, "pdfs", [root], ChunkingOptions(max_tokens=40, overlap_tokens=0))
        collection = store.collection("pdfs")
        assert collection is not None
        chunks = chunks_of(store, collection, "paper.pdf")
        assert len(chunks) >= 3
        context = assemble_context(store, as_hits(chunks[1]), 1000, neighbours=1)
        (passage,) = context.passages
        covered = [c for c in chunks if c.id in passage.chunk_ids]
        assert passage.page == min(c.chunk.page or 0 for c in covered)
        assert passage.page_end == max(c.chunk.page_end or 0 for c in covered)

    def test_no_hits(self, store: Store) -> None:
        context = assemble_context(store, [], 100)
        assert context == Context([], 0, 100, 0, 0)

    def test_the_same_hit_twice_counts_once(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        context = assemble_context(store, as_hits(long_doc[4], long_doc[4]), 1000, neighbours=0)
        assert len(context.passages) == 1
        assert context.duplicates == 0


class TestDuplicates:
    """The feature's acceptance test, with the budget below."""

    @pytest.fixture
    def copies(self, store: Store, tmp_path: Path) -> list[StoredChunk]:
        base = (
            "Servers compact their log by writing a snapshot of the state machine and "
            "discarding the entries it covers. A follower that is far behind receives the "
            "snapshot from the leader instead of the individual entries."
        )
        collection = build(
            store,
            tmp_path,
            {
                "a.md": f"# Raft\n\n{base}",
                "b.md": f"# Mirror\n\n{base.replace('individual', 'separate')} See the paper.",
                "c.md": "# Elections\n\nA candidate asks the other servers for votes in its term.",
                "d.md": f"# Copy\n\n{base.upper()}",
            },
        )
        return [chunks_of(store, collection, name)[0] for name in ("a.md", "b.md", "c.md", "d.md")]

    def test_near_duplicate_chunks_are_collapsed_into_the_best_ranked(
        self, store: Store, copies: list[StoredChunk]
    ) -> None:
        a, b, c, d = copies
        context = assemble_context(store, as_hits(a, b, c, d), 1000, neighbours=0)
        assert [p.source for p in context.passages] == ["a.md", "c.md"]
        assert context.duplicates == 2

    def test_the_better_ranked_copy_is_the_one_kept(
        self, store: Store, copies: list[StoredChunk]
    ) -> None:
        a, b, c, _ = copies
        context = assemble_context(store, as_hits(c, b, a), 1000, neighbours=0)
        assert [p.source for p in context.passages] == ["c.md", "b.md"]

    def test_a_stricter_threshold_keeps_more(self, store: Store, copies: list[StoredChunk]) -> None:
        a, b, *_ = copies
        loose = assemble_context(store, as_hits(a, b), 1000, neighbours=0, duplicate_threshold=0.5)
        assert len(loose.passages) == 1
        strict = assemble_context(store, as_hits(a, b), 1000, neighbours=0, duplicate_threshold=1)
        assert len(strict.passages) == 2

    def test_a_chunk_inside_a_longer_one_is_a_duplicate(self, store: Store, tmp_path: Path) -> None:
        words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        collection = build(
            store,
            tmp_path,
            {"long.md": f"# L\n\n{words} mu nu xi omicron.", "short.md": f"# S\n\n{words}."},
        )
        long = chunks_of(store, collection, "long.md")[0]
        short = chunks_of(store, collection, "short.md")[0]
        context = assemble_context(store, as_hits(long, short), 1000, neighbours=0)
        assert [p.source for p in context.passages] == ["long.md"]

    def test_unrelated_chunks_are_not_duplicates(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        context = assemble_context(
            store, as_hits(long_doc[1], long_doc[5], long_doc[9]), 1000, neighbours=0
        )
        assert context.duplicates == 0
        assert len(context.passages) == 3

    def test_consecutive_chunks_overlap_by_design_and_are_not_collapsed(
        self, store: Store, tmp_path: Path
    ) -> None:
        # The second chunk is mostly the end of the first, yet both belong in the context.
        collection = build(store, tmp_path, {"doc.md": "# Doc\n\n" + prose(3, words=12)})
        options = ChunkingOptions(max_tokens=40, overlap_tokens=30)
        ingest(store, "tight", [tmp_path / "docs"], options)
        tight = store.collection("tight")
        assert tight is not None
        chunks = chunks_of(store, tight, "doc.md")
        assert len(chunks) >= 2
        context = assemble_context(store, as_hits(*chunks[:2]), 1000, neighbours=0)
        assert context.duplicates == 0
        assert len(context.passages) == 1
        assert collection.id != tight.id

    def test_a_neighbour_that_repeats_another_chunk_is_not_added(
        self, store: Store, tmp_path: Path
    ) -> None:
        base = "one two three four five six seven eight nine ten eleven twelve"
        text = f"# Doc\n\n{base}.\n\n{prose(4)}"
        collection = build(store, tmp_path, {"a.md": text, "b.md": f"# Other\n\n{base}."})
        copy = chunks_of(store, collection, "b.md")[0]
        mine = chunks_of(store, collection, "a.md")
        context = assemble_context(store, as_hits(copy, mine[-1]), 1000)
        assert mine[0].id not in [i for p in context.passages for i in p.chunk_ids]


class TestBudget:
    """The feature's acceptance test: the assembled context never exceeds its budget."""

    @pytest.mark.parametrize("budget", [1, 30, 61, 62, 63, 100, 124, 190, 250, 400, 635, 1000])
    @pytest.mark.parametrize("neighbours", [0, 1, 3])
    def test_the_context_never_exceeds_the_budget(
        self, store: Store, long_doc: list[StoredChunk], budget: int, neighbours: int
    ) -> None:
        hits = as_hits(long_doc[7], long_doc[2], long_doc[4], long_doc[9], long_doc[0])
        context = assemble_context(store, hits, budget, neighbours=neighbours)
        assert context.budget == budget
        assert context.tokens <= budget
        assert sum(count_tokens(p.text) for p in context.passages) <= budget
        assert context.tokens == sum(p.tokens for p in context.passages)

    def test_merging_never_counts_more_than_the_chunks_cost(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        context = assemble_context(store, as_hits(long_doc[4]), 1000, neighbours=3)
        (passage,) = context.passages
        by_id = {c.id: c for c in long_doc}
        assert passage.tokens < sum(by_id[i].chunk.token_count for i in passage.chunk_ids)

    def test_a_hit_too_big_for_what_is_left_is_skipped_and_counted(
        self, store: Store, long_doc: list[StoredChunk]
    ) -> None:
        first, second = long_doc[2], long_doc[6]
        budget = first.chunk.token_count + second.chunk.token_count - 1
        context = assemble_context(store, as_hits(first, second), budget, neighbours=0)
        assert [p.hit_ids for p in context.passages] == [(first.id,)]
        assert context.skipped == 1

    def test_a_smaller_hit_still_fits_after_a_big_one_was_skipped(
        self, store: Store, tmp_path: Path
    ) -> None:
        collection = build(
            store,
            tmp_path,
            {"big.md": "# Big\n\n" + prose(4), "small.md": "# Small\n\nJust a few words here."},
        )
        big = chunks_of(store, collection, "big.md")[0]
        small = chunks_of(store, collection, "small.md")[0]
        context = assemble_context(store, as_hits(big, small), small.chunk.token_count + 2)
        assert [p.source for p in context.passages] == ["small.md"]
        assert context.skipped == 1

    def test_nothing_fits(self, store: Store, long_doc: list[StoredChunk]) -> None:
        context = assemble_context(store, as_hits(long_doc[0]), 5)
        assert context.passages == []
        assert (context.tokens, context.skipped) == (0, 1)


class TestValidation:
    def test_budget(self, store: Store) -> None:
        with pytest.raises(ValueError, match="budget must be at least 1"):
            assemble_context(store, [], 0)

    def test_neighbours(self, store: Store) -> None:
        with pytest.raises(ValueError, match="neighbours must not be negative"):
            assemble_context(store, [], 10, neighbours=-1)

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5])
    def test_duplicate_threshold(self, store: Store, threshold: float) -> None:
        with pytest.raises(ValueError, match="duplicate threshold must be in"):
            assemble_context(store, [], 10, duplicate_threshold=threshold)

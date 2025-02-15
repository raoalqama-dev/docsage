from __future__ import annotations

import itertools
import random

import pytest

from docsage.ingest.chunking import ChunkingOptions, chunk_document, count_tokens
from docsage.ingest.loaders import load_markdown
from docsage.models import Block, Chunk, Document
from fixtures import prose


def document(*blocks: Block) -> Document:
    return Document(title="t", format="text", blocks=blocks)


def assert_within_budget(chunks: list[Chunk], options: ChunkingOptions) -> None:
    for chunk in chunks:
        assert chunk.token_count == count_tokens(chunk.text)
        assert 0 < chunk.token_count <= options.max_tokens
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


class TestCountTokens:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("", 0),
            ("   \n", 0),
            ("the cat", 2),
            ("chunking", 2),  # four characters per token
            ("retrieval", 3),
            ("Hello, world!", 6),  # errs high: BPE makes 4
            ("x=1;", 4),
            ("café", 1),
        ],
    )
    def test_counts(self, text: str, expected: int) -> None:
        assert count_tokens(text) == expected

    def test_counts_add_up_across_whitespace(self) -> None:
        parts = ["Raft elects a leader.", "It (the leader) replicates", "entries—quickly."]
        assert count_tokens(" ".join(parts)) == sum(count_tokens(part) for part in parts)
        assert count_tokens("\n\n".join(parts)) == sum(count_tokens(part) for part in parts)


class TestOptions:
    def test_defaults(self) -> None:
        options = ChunkingOptions()
        assert (options.max_tokens, options.overlap_tokens) == (400, 60)
        assert options.signature == "400/60"

    @pytest.mark.parametrize(
        ("max_tokens", "overlap", "message"),
        [
            (0, 0, "at least 1 token"),
            (100, 100, "between 0 and the chunk size"),
            (100, -1, "between 0 and the chunk size"),
        ],
    )
    def test_invalid(self, max_tokens: int, overlap: int, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            ChunkingOptions(max_tokens, overlap)


class TestChunking:
    def test_empty_document(self) -> None:
        assert chunk_document(document()) == []

    def test_small_sections_are_one_chunk_each(self) -> None:
        source = (
            b"# Guide\n\nIntro one.\n\nIntro two.\n\n## Setup\n\nInstall it.\n\n## Use\n\nRun it."
        )
        chunks = chunk_document(load_markdown(source, "guide"))
        assert [(chunk.heading_path, chunk.text) for chunk in chunks] == [
            (("Guide",), "Intro one.\n\nIntro two."),
            (("Guide", "Setup"), "Install it."),
            (("Guide", "Use"), "Run it."),
        ]
        assert all(chunk.page is None and chunk.page_end is None for chunk in chunks)

    def test_a_paragraph_that_fits_a_chunk_is_never_split(self) -> None:
        options = ChunkingOptions(max_tokens=100, overlap_tokens=0)
        first, second = prose(4), prose(4, start=4)
        assert count_tokens(first) + count_tokens(second) > 100
        assert count_tokens(second) <= 100
        chunks = chunk_document(document(Block(first), Block(second)), options)
        assert [chunk.text for chunk in chunks] == [first, second]

    def test_long_paragraph_splits_at_sentences_with_overlap(self) -> None:
        options = ChunkingOptions(max_tokens=60, overlap_tokens=25)
        text = prose(12)
        chunks = chunk_document(document(Block(text)), options)
        assert len(chunks) > 1
        assert_within_budget(chunks, options)
        sentence_ids = []
        for chunk in chunks:
            # Every chunk is whole sentences.
            assert chunk.text.startswith("Sentence ")
            assert chunk.text.endswith(".")
            sentence_ids.append(
                [int(part.split()[0]) for part in chunk.text.split("Sentence ")[1:]]
            )
        for previous, current in itertools.pairwise(sentence_ids):
            # Each chunk starts by repeating the previous chunk's last sentence.
            assert current[0] == previous[-1]
        assert sentence_ids[-1][-1] == 11
        # Nothing is lost: every sentence appears, in order.
        assert sorted({n for ids in sentence_ids for n in ids}) == list(range(12))

    def test_overlap_never_crosses_a_heading(self) -> None:
        options = ChunkingOptions(max_tokens=60, overlap_tokens=25)
        chunks = chunk_document(
            document(Block(prose(6), ("A",)), Block(prose(6, start=6), ("B",))), options
        )
        for chunk in chunks:
            ids = [int(part.split()[0]) for part in chunk.text.split("Sentence ")[1:]]
            expected_section = ("A",) if ids[0] < 6 else ("B",)
            assert chunk.heading_path == expected_section
            assert all((n < 6) == (ids[0] < 6) for n in ids)

    def test_zero_overlap_repeats_nothing(self) -> None:
        options = ChunkingOptions(max_tokens=60, overlap_tokens=0)
        chunks = chunk_document(document(Block(prose(12))), options)
        assert " ".join(chunk.text for chunk in chunks) == prose(12)

    def test_sentence_longer_than_the_budget_is_cut_between_words(self) -> None:
        options = ChunkingOptions(max_tokens=20, overlap_tokens=0)
        words = [f"w{n}" for n in range(50)]
        chunks = chunk_document(document(Block(" ".join(words))), options)
        assert_within_budget(chunks, options)
        assert " ".join(chunk.text for chunk in chunks).split() == words

    def test_a_word_longer_than_the_budget_is_cut_anyway(self) -> None:
        options = ChunkingOptions(max_tokens=5, overlap_tokens=0)
        chunks = chunk_document(document(Block("a" * 40 + " end")), options)
        assert_within_budget(chunks, options)
        assert "".join(chunk.text.replace(" ", "") for chunk in chunks) == "a" * 40 + "end"

    def test_abbreviations_do_not_end_sentences(self) -> None:
        options = ChunkingOptions(max_tokens=16, overlap_tokens=0)
        text = "Use a quorum, e.g. three nodes. Then elect."
        chunks = chunk_document(document(Block(text)), options)
        assert [chunk.text for chunk in chunks] == [
            "Use a quorum, e.g. three nodes.",
            "Then elect.",
        ]

    def test_lines_and_indentation_are_kept(self) -> None:
        options = ChunkingOptions(max_tokens=14, overlap_tokens=0)
        code = "def apply(entry):\n    log.append(entry)\n    return entry.index"
        chunks = chunk_document(document(Block(code)), options)
        assert "\n".join(chunk.text for chunk in chunks) == code

    def test_pages_cover_the_blocks_in_a_chunk(self) -> None:
        options = ChunkingOptions(max_tokens=400, overlap_tokens=0)
        chunks = chunk_document(
            document(Block("one.", page=2), Block("two.", page=3), Block("three.", page=3)),
            options,
        )
        assert [(chunk.page, chunk.page_end) for chunk in chunks] == [(2, 3)]

    def test_overlap_pages_come_from_the_repeated_sentences(self) -> None:
        options = ChunkingOptions(max_tokens=70, overlap_tokens=30)
        chunks = chunk_document(
            document(Block(prose(2), page=1), Block(prose(2, start=2), page=2)), options
        )
        # The second chunk repeats sentence 1, from page 1, before page 2's paragraph.
        assert chunks[1].text.startswith("Sentence 1 ")
        assert [(chunk.page, chunk.page_end) for chunk in chunks] == [(1, 1), (1, 2)]

    def test_random_documents_respect_the_budget_and_are_stable(self) -> None:
        rng = random.Random(7)
        for _ in range(40):
            options = ChunkingOptions(rng.randint(8, 120), 0)
            options = ChunkingOptions(options.max_tokens, rng.randint(0, options.max_tokens - 1))
            blocks = []
            for _ in range(rng.randint(1, 12)):
                sentences = rng.randint(1, 15)
                text = prose(sentences, start=rng.randint(0, 50), words=rng.randint(1, 30))
                if rng.random() < 0.2:
                    text = text.replace(". ", ".\n")
                heading = (f"H{rng.randint(0, 3)}",)
                blocks.append(Block(text, heading, rng.choice([None, 1, 2])))
            doc = document(*blocks)
            chunks = chunk_document(doc, options)
            assert_within_budget(chunks, options)
            assert chunks == chunk_document(doc, options)
            # Every word of the document is in some chunk of its section.
            for block in blocks:
                section_text = " ".join(
                    chunk.text for chunk in chunks if chunk.heading_path == block.heading_path
                )
                assert set(block.text.split()) <= set(section_text.split())

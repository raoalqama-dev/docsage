"""Splitting documents into chunks that respect headings, paragraphs and sentences.

A chunk never crosses a heading, so its heading path describes all of it.
Within a section, whole paragraphs are packed into chunks up to the token
budget; a paragraph too long for a chunk of its own is split at line breaks,
then at sentences, and only a sentence longer than the budget is cut between
words. Each chunk after the first in a section starts with the last sentences
of the previous one, up to the overlap budget.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import groupby

from docsage.models import Block, Chunk, Document

_TOKEN_RE = re.compile(r"\w{1,4}|[^\w\s]")
_SENTENCE_END_RE = re.compile(r"[.!?\u2026]+[\"'\u201d\u2019)\]]*\s+")


def count_tokens(text: str) -> int:
    """Estimate how many tokens a language model's tokenizer would make of ``text``.

    Every punctuation mark is a token, and a word is a token per four
    characters. For English prose this errs high against BPE tokenizers, so a
    budget counted this way fits the model's. Counts add up: two texts joined
    by whitespace count as the sum of their counts.
    """
    return sum(1 for _ in _TOKEN_RE.finditer(text))


@dataclass(frozen=True, slots=True)
class ChunkingOptions:
    """The token budget for each chunk, and how much of it to repeat from the previous one."""

    max_tokens: int = 400
    overlap_tokens: int = 60

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"the chunk size must be at least 1 token, not {self.max_tokens}")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError(
                f"the overlap must be between 0 and the chunk size ({self.max_tokens}) "
                f"less one, not {self.overlap_tokens}"
            )

    @property
    def signature(self) -> str:
        """Identifies these options, so stored chunks made with others can be found."""
        return f"{self.max_tokens}/{self.overlap_tokens}"


@dataclass(frozen=True, slots=True)
class _Unit:
    """The smallest piece the chunker moves around: a paragraph, line, sentence or word run."""

    text: str
    tokens: int
    page: int | None
    # How the unit joins the one before it in a chunk.
    separator: str


def _sentences(text: str) -> Iterator[str]:
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        following = text[match.end() : match.end() + 1]
        # "e.g. this" is not a sentence break; "end. Next" and "end. 42" are.
        if following and (following.isupper() or following.isdigit() or following in "\"'(["):
            yield text[start : match.end()].strip()
            start = match.end()
    if text[start:].strip():
        yield text[start:].strip()


def _word_runs(text: str, max_tokens: int) -> Iterator[str]:
    """Cut text into runs of at most ``max_tokens``, between words where possible."""
    tokens = list(_TOKEN_RE.finditer(text))
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        if end < len(tokens):
            # Back up to the start of a word, unless that would leave the run empty.
            cut = end
            while cut > start + 1 and not text[tokens[cut].start() - 1].isspace():
                cut -= 1
            if cut > start + 1 or text[tokens[cut].start() - 1].isspace():
                end = cut
        yield text[tokens[start].start() : tokens[end - 1].end()]
        start = end


def _split(text: str, max_tokens: int) -> Iterator[tuple[str, str]]:
    """A paragraph's sentences as ``(text, separator)``, each within ``max_tokens``.

    Line breaks (list items, code) and a line's indentation are kept; a
    sentence longer than the budget is cut into runs of words.
    """
    lines = [line.rstrip() for line in text.split("\n") if line.strip()]
    for line_number, line in enumerate(lines):
        indent = line[: len(line) - len(line.lstrip())]
        separator = "\n" if line_number else ""
        for sentence in _sentences(line):
            for run in _word_runs(sentence, max_tokens):
                yield indent + run, separator
                indent, separator = "", " "


def _paragraph_units(block: Block, max_tokens: int) -> list[_Unit]:
    units = [
        _Unit(piece, count_tokens(piece), block.page, separator)
        for piece, separator in _split(block.text, max_tokens)
    ]
    # Paragraphs are separated by a blank line.
    return [_Unit(units[0].text, units[0].tokens, units[0].page, "\n\n"), *units[1:]]


def _tail(units: Sequence[_Unit], budget: int) -> list[_Unit]:
    """The last units of a chunk that fit in ``budget``, to repeat in the next chunk."""
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.tokens > budget:
            break
        tail.append(unit)
        total += unit.tokens
    tail.reverse()
    return tail


class _Packer:
    def __init__(self, heading_path: tuple[str, ...], options: ChunkingOptions) -> None:
        self.heading_path = heading_path
        self.options = options
        self.units: list[_Unit] = []
        self.tokens = 0
        # Units at the front of ``units`` repeated from the previous chunk.
        self.carried = 0
        self.chunks: list[tuple[list[_Unit], tuple[str, ...]]] = []

    def add(self, unit: _Unit) -> None:
        self.units.append(unit)
        self.tokens += unit.tokens

    def fits(self, tokens: int) -> bool:
        return self.tokens + tokens <= self.options.max_tokens

    def emit(self, next_tokens: int) -> None:
        """Close the chunk, carrying its tail into the next one if ``next_tokens`` leaves room."""
        if len(self.units) > self.carried:
            self.chunks.append((self.units, self.heading_path))
        budget = min(self.options.overlap_tokens, self.options.max_tokens - next_tokens)
        self.units = _tail(self.units, budget) if budget > 0 else []
        self.tokens = sum(unit.tokens for unit in self.units)
        self.carried = len(self.units)

    def paragraph(self, units: list[_Unit]) -> None:
        total = sum(unit.tokens for unit in units)
        if self.fits(total):
            for unit in units:
                self.add(unit)
        elif total <= self.options.max_tokens:
            # The paragraph fits a chunk of its own: start one rather than split it.
            self.emit(total)
            for unit in units:
                self.add(unit)
        else:
            for unit in units:
                if not self.fits(unit.tokens):
                    self.emit(unit.tokens)
                self.add(unit)

    def finish(self) -> list[tuple[list[_Unit], tuple[str, ...]]]:
        self.emit(self.options.max_tokens)
        return self.chunks


def _render(units: Sequence[_Unit]) -> str:
    return "".join(
        (unit.separator if index else "") + unit.text for index, unit in enumerate(units)
    )


def _sections(blocks: Iterable[Block]) -> Iterator[tuple[tuple[str, ...], list[Block]]]:
    for heading_path, group in groupby(blocks, key=lambda block: block.heading_path):
        yield heading_path, list(group)


def chunk_document(document: Document, options: ChunkingOptions | None = None) -> list[Chunk]:
    """Split a document into chunks of at most ``options.max_tokens`` tokens each."""
    options = options or ChunkingOptions()
    chunks: list[Chunk] = []
    for heading_path, blocks in _sections(document.blocks):
        packer = _Packer(heading_path, options)
        for block in blocks:
            packer.paragraph(_paragraph_units(block, options.max_tokens))
        for units, path in packer.finish():
            pages = [unit.page for unit in units if unit.page is not None]
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    text=_render(units),
                    heading_path=path,
                    page=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    token_count=sum(unit.tokens for unit in units),
                )
            )
    return chunks

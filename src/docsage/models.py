"""Documents as loaded from files, and the chunks they are split into."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Block:
    """A paragraph-level run of text: a paragraph, a list, a table or a code block.

    ``heading_path`` holds the headings the block sits under, outermost first.
    ``page`` is the 1-based page of a PDF, and ``None`` for other formats.
    """

    text: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None


@dataclass(frozen=True, slots=True)
class Document:
    """A loaded file: its title and format, and its text as blocks in reading order."""

    title: str
    format: str
    blocks: tuple[Block, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable piece of a document.

    A chunk never spans two sections: every block in it shares ``heading_path``.
    ``page`` and ``page_end`` are the first and last PDF pages it covers.
    """

    ordinal: int
    text: str
    heading_path: tuple[str, ...]
    page: int | None
    page_end: int | None
    token_count: int

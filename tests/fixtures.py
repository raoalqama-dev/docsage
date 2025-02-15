"""Builders for fixture documents, PDFs among them, made in the tests themselves."""

from __future__ import annotations

import io
from collections.abc import Sequence

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

#: A page is a list of paragraphs, and a paragraph a list of lines.
Page = Sequence[Sequence[str]]


def _escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(
    pages: Sequence[Page],
    *,
    outline: Sequence[tuple[str, int, int]] = (),
    title: str | None = None,
) -> bytes:
    """A PDF with one Helvetica text page per entry in ``pages``.

    Lines are 14 points apart, and paragraphs have an empty line between them,
    as a typeset document has. ``outline`` entries are ``(title, page index,
    level)``; each level-2 entry nests under the level-1 entry before it.
    """
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for paragraphs in pages:
        page = writer.add_blank_page(612, 792)
        operators = ["BT /F1 11 Tf 14 TL 72 720 Td"]
        for number, paragraph in enumerate(paragraphs):
            if number:
                operators.append("T*")
            operators.extend(f"({_escape(line)}) Tj T*" for line in paragraph)
        operators.append("ET")
        content = DecodedStreamObject()
        content.set_data("\n".join(operators).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(content)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
    parent = None
    for heading, page_index, level in outline:
        item = writer.add_outline_item(heading, page_index, parent=parent if level > 1 else None)
        if level == 1:
            parent = item
    if title is not None:
        writer.add_metadata({"/Title": title})
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def prose(sentences: int, *, start: int = 0, words: int = 9) -> str:
    """Deterministic prose: numbered sentences of ``words`` words each."""
    vocabulary = [
        "replica", "leader", "term", "log", "entry", "commit", "index", "quorum", "vote",
        "election", "heartbeat", "snapshot", "follower", "candidate", "cluster", "message",
        "timeout", "state", "machine", "apply",
    ]  # fmt: skip
    out = []
    for number in range(start, start + sentences):
        body = " ".join(vocabulary[(number * 7 + k) % len(vocabulary)] for k in range(words - 1))
        out.append(f"Sentence {number} says {body}.")
    return " ".join(out)

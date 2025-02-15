"""Loaders that turn Markdown, plain text, HTML and PDF files into blocks of text.

Each loader keeps the structure retrieval and citations need: the headings a
block sits under, and for PDFs the page it came from.
"""

from __future__ import annotations

import io
import itertools
import re
import statistics
from collections.abc import Callable, Iterator, Sequence
from html.parser import HTMLParser
from pathlib import PurePath
from typing import Any

from pypdf import PageObject, PdfReader
from pypdf.generic import Destination

from docsage.models import Block, Document

FORMATS: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


class LoaderError(Exception):
    """A file that cannot be read as the format its extension claims."""


def format_for(path: PurePath) -> str | None:
    """The format of a file, from its extension, or ``None`` if it is not supported."""
    return FORMATS.get(path.suffix.lower())


def load_document(path: PurePath, data: bytes) -> Document:
    """Parse the contents of the file at ``path`` according to its extension."""
    fmt = format_for(path)
    if fmt is None:
        raise LoaderError(f"unsupported file type {path.suffix or '(no extension)'!r}")
    return _LOADERS[fmt](data, path.stem)


def _decode(data: bytes) -> str:
    # Undecodable bytes become U+FFFD rather than failing the whole file.
    return data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


_SPACE_RE = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


# Plain text


def load_text(data: bytes, stem: str) -> Document:
    """Paragraphs are separated by blank lines; hard-wrapped lines are rejoined."""
    blocks = tuple(
        Block(_squash(paragraph))
        for paragraph in re.split(r"\n[ \t]*\n", _decode(data))
        if paragraph.strip()
    )
    return Document(title=stem, format="text", blocks=blocks)


# Markdown

_ATX_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_THEMATIC_RE = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_LINE_START_RE = re.compile(r"^\s*(?:[-*+][ \t]|\d{1,9}[.)][ \t]|\||>)")
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_AUTOLINK_RE = re.compile(r"<((?:https?|mailto):[^>\s]+)>")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_FRONT_MATTER_TITLE_RE = re.compile(r"^title:[ \t]*(.+?)[ \t]*$", re.MULTILINE)


def _inline(text: str) -> str:
    """Reduce links and images to their text, which is what a reader sees."""
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    return _AUTOLINK_RE.sub(r"\1", text)


def _heading_text(text: str) -> str:
    return _squash(_inline(text).replace("`", "").strip("*_ "))


def _split_front_matter(text: str) -> tuple[str | None, str]:
    """Strip YAML front matter, returning its ``title`` if it has a plain one."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() in ("---", "..."):
            header = "\n".join(lines[1:end])
            match = _FRONT_MATTER_TITLE_RE.search(header)
            title = match.group(1).strip("'\"") if match else None
            return title or None, "\n".join(lines[end + 1 :])
    return None, text


class _Sections:
    """The heading stack, and the blocks collected under it."""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []
        self.blocks: list[Block] = []
        self.first_h1: str | None = None

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(title for _, title in self.stack)

    def heading(self, level: int, title: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        if title:
            self.stack.append((level, title))
            if level == 1 and self.first_h1 is None:
                self.first_h1 = title

    def add(self, text: str, page: int | None = None) -> None:
        if text.strip():
            self.blocks.append(Block(text.strip(), self.path, page))


def _join_markdown_lines(lines: Sequence[str]) -> str:
    """Rejoin soft-wrapped lines, but keep list items, table rows and quotes on their own."""
    out: list[str] = []
    for raw in lines:
        if _TABLE_RULE_RE.match(raw) and "-" in raw:
            continue
        line = _squash(_inline(raw))
        if out and not _LINE_START_RE.match(raw):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out)


def load_markdown(data: bytes, stem: str) -> Document:
    """CommonMark-style ATX and setext headings, fenced code, paragraphs and lists."""
    front_title, text = _split_front_matter(_COMMENT_RE.sub("", _decode(data)))
    sections = _Sections()
    paragraph: list[str] = []
    fence: str | None = None
    code: list[str] = []

    def flush() -> None:
        sections.add(_join_markdown_lines(paragraph))
        paragraph.clear()

    for line in text.split("\n"):
        if fence is not None:
            stripped = line.strip()
            if stripped.startswith(fence) and stripped.strip(fence[0]) == "":
                sections.add("\n".join(code))
                fence = None
                code.clear()
            else:
                code.append(line.rstrip())
            continue
        if match := _FENCE_RE.match(line):
            flush()
            fence = match.group(1)
            continue
        if match := _ATX_RE.match(line):
            flush()
            sections.heading(len(match.group(1)), _heading_text(match.group(2) or ""))
            continue
        if paragraph and (match := _SETEXT_RE.match(line)):
            title = _heading_text(" ".join(part.strip() for part in paragraph))
            paragraph.clear()
            sections.heading(1 if match.group(1)[0] == "=" else 2, title)
            continue
        if not line.strip() or _THEMATIC_RE.match(line):
            flush()
            continue
        paragraph.append(line)
    flush()
    if fence is not None:
        # An unclosed fence runs to the end of the document, as in CommonMark.
        sections.add("\n".join(code))

    title = front_title or sections.first_h1 or stem
    return Document(title=title, format="markdown", blocks=tuple(sections.blocks))


# HTML

_SKIPPED_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "canvas", "iframe", "nav"}
)
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "body", "br", "dd", "details", "dialog",
        "div", "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form", "header",
        "hr", "li", "main", "ol", "p", "pre", "section", "summary", "table", "tbody", "tfoot",
        "thead", "tr", "ul",
    }
)  # fmt: skip
_CELL_TAGS = frozenset({"td", "th"})
_HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}


class _HTMLBlocks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections = _Sections()
        self.title: str | None = None
        self._skip_depth = 0
        self._in_head = False
        self._in_title = False
        self._pre_depth = 0
        self._heading: int | None = None
        self._text: list[str] = []

    def _flush(self) -> None:
        text = "".join(self._text)
        self._text.clear()
        if self._heading is not None:
            self.sections.heading(self._heading, _squash(text))
            self._heading = None
        elif self._pre_depth:
            self.sections.add(text.strip("\n"))
        else:
            self.sections.add(_squash(text))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
            self.title = ""
            return
        # </head> is optional in HTML, so <body> ends the head as well.
        if tag in ("head", "body"):
            self._in_head = tag == "head"
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth or self._in_head:
            return
        if tag in _HEADING_TAGS:
            self._flush()
            self._heading = _HEADING_TAGS[tag]
        elif tag in _BLOCK_TAGS and self._heading is None:
            self._flush()
            if tag == "pre":
                self._pre_depth += 1
        elif tag in _CELL_TAGS and self._text:
            self._text.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag == "head":
            self._in_head = False
            return
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth or self._in_head:
            return
        if tag in _HEADING_TAGS and self._heading is not None:
            self._flush()
        elif tag in _BLOCK_TAGS and self._heading is None:
            self._flush()
            if tag == "pre":
                self._pre_depth = max(0, self._pre_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title = (self.title or "") + data
        elif not self._skip_depth and not self._in_head:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def load_html(data: bytes, stem: str) -> Document:
    """Headings ``h1``-``h6`` form the heading path; scripts, styles and navigation are dropped."""
    parser = _HTMLBlocks()
    parser.feed(_decode(data))
    parser.close()
    title = _squash(parser.title or "") or parser.sections.first_h1 or stem
    return Document(title=title, format="html", blocks=tuple(parser.sections.blocks))


# PDF

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(?=[a-z])")


def _outline(reader: PdfReader) -> list[tuple[int, int, str]]:
    """The document outline (bookmarks) as ``(page index, level, title)``, in page order."""
    entries: list[tuple[int, int, str]] = []

    def walk(items: Sequence[Any], level: int) -> Iterator[tuple[int, int, str]]:
        for item in items:
            if isinstance(item, list):
                yield from walk(item, level + 1)
            elif isinstance(item, Destination):
                page = reader.get_destination_page_number(item)
                title = _squash(str(item.title or ""))
                if page is not None and page >= 0 and title:
                    yield page, level, title

    entries.extend(walk(reader.outline, 1))
    # A stable sort keeps the outline's own order for entries on the same page.
    entries.sort(key=lambda entry: entry[0])
    return entries


def _page_text(page: PageObject) -> str:
    """A page's text, with a blank line wherever the gap between lines is unusually large.

    pypdf reports lines but not paragraphs. Paragraphs set with space between
    them show up as line gaps well above the page's median line spacing.
    """
    fragments: list[tuple[str, float]] = []

    def visit(text: str, cm: list[float], tm: list[float], *_: Any) -> None:
        if text:
            # The baseline's height on the page: the text matrix through the CTM.
            fragments.append((text, tm[4] * cm[1] + tm[5] * cm[3] + cm[5]))

    text = page.extract_text(visitor_text=visit)
    if "".join(fragment for fragment, _ in fragments) != text:
        return text
    line_starts: list[int] = []
    for index in range(len(fragments)):
        if index == 0 or fragments[index - 1][0].endswith("\n"):
            line_starts.append(index)
    gaps = [
        fragments[earlier][1] - fragments[later][1]
        for earlier, later in itertools.pairwise(line_starts)
    ]
    positive = sorted(gap for gap in gaps if gap > 0)
    if not positive:
        return text
    spacing = statistics.median(positive)
    breaks = {
        later
        for (earlier, later), gap in zip(itertools.pairwise(line_starts), gaps, strict=True)
        if gap > 1.5 * spacing
    }
    return "".join(
        ("\n" if index in breaks else "") + fragment
        for index, (fragment, _) in enumerate(fragments)
    )


def _pdf_paragraphs(text: str) -> list[str]:
    text = _HYPHEN_BREAK_RE.sub(r"\1", text.replace("\r\n", "\n").replace("\r", "\n"))
    return [_squash(part) for part in re.split(r"\n[ \t]*\n", text) if part.strip()]


def load_pdf(data: bytes, stem: str) -> Document:
    """One block per paragraph, each with its page; the outline supplies the headings.

    Paragraphs are found from the spacing between lines (see ``_page_text``),
    and a page takes the headings of the last outline entry that starts on or
    before it, so a section that begins mid-page is credited from that page on.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise LoaderError("the PDF is encrypted")
        outline = _outline(reader)
        sections = _Sections()
        next_entry = 0
        for index, page in enumerate(reader.pages):
            while next_entry < len(outline) and outline[next_entry][0] <= index:
                _, level, title = outline[next_entry]
                sections.heading(level, title)
                next_entry += 1
            for paragraph in _pdf_paragraphs(_page_text(page)):
                sections.add(paragraph, page=index + 1)
        metadata_title = _squash(str(reader.metadata.title or "")) if reader.metadata else ""
    except LoaderError:
        raise
    except Exception as exc:
        # Malformed PDFs fail in many ways inside pypdf; one bad file must not
        # stop a whole ingestion run, so every failure becomes a LoaderError.
        raise LoaderError(f"cannot read the PDF: {exc}") from exc
    return Document(title=metadata_title or stem, format="pdf", blocks=tuple(sections.blocks))


_LOADERS: dict[str, Callable[[bytes, str], Document]] = {
    "markdown": load_markdown,
    "text": load_text,
    "html": load_html,
    "pdf": load_pdf,
}

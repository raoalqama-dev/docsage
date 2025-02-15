from __future__ import annotations

from pathlib import PurePath

import pytest

from docsage.ingest.loaders import (
    LoaderError,
    format_for,
    load_document,
    load_html,
    load_markdown,
    load_pdf,
    load_text,
)
from docsage.models import Block
from fixtures import make_pdf


def texts(blocks: tuple[Block, ...]) -> list[str]:
    return [block.text for block in blocks]


class TestFormats:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("a.md", "markdown"),
            ("a.MARKDOWN", "markdown"),
            ("a.txt", "text"),
            ("a.text", "text"),
            ("a.html", "html"),
            ("a.HTM", "html"),
            ("a.pdf", "pdf"),
            ("a.docx", None),
            ("README", None),
        ],
    )
    def test_format_from_extension(self, name: str, expected: str | None) -> None:
        assert format_for(PurePath(name)) == expected

    def test_load_document_dispatches_on_extension(self) -> None:
        document = load_document(PurePath("notes/guide.md"), b"# Guide\n\nHello.")
        assert (document.format, document.title) == ("markdown", "Guide")

    def test_unsupported_extension_is_an_error(self) -> None:
        with pytest.raises(LoaderError, match=r"unsupported file type '\.docx'"):
            load_document(PurePath("a.docx"), b"")
        with pytest.raises(LoaderError, match=r"\(no extension\)"):
            load_document(PurePath("Makefile"), b"")


class TestText:
    def test_paragraphs_are_separated_by_blank_lines_and_unwrapped(self) -> None:
        data = b"First line\nwrapped here.\n\n  \nSecond paragraph.\r\n\r\nThird.\n"
        document = load_text(data, "notes")
        assert texts(document.blocks) == [
            "First line wrapped here.",
            "Second paragraph.",
            "Third.",
        ]
        assert document.title == "notes"
        assert all(block.heading_path == () and block.page is None for block in document.blocks)

    def test_bom_and_invalid_utf8(self) -> None:
        document = load_text(b"\xef\xbb\xbfcaf\xc3\xa9 \xff ok", "x")
        assert texts(document.blocks) == ["café � ok"]

    def test_empty_file_has_no_blocks(self) -> None:
        assert load_text(b"\n \n", "x").blocks == ()


MARKDOWN = b"""---
title: "Raft notes"
tags: [consensus]
---

Preface before any heading.

# Consensus

Raft elects a *leader*, which
replicates the [log](https://raft.github.io) to ![a diagram](d.png) followers.

## Elections
### Timeouts ###

A follower that hears nothing becomes a candidate.

- first item
  wraps here
- second item
1. numbered

| a | b |
|---|:-:|
| 1 | 2 |

> quoted

Log replication
---------------

```python
# not a heading
def apply(entry):
    return entry
```

Setext title
============

***
After a rule. <!-- hidden
comment --> Visible. See <https://example.com>.
"""


class TestMarkdown:
    def test_blocks_and_heading_paths(self) -> None:
        document = load_markdown(MARKDOWN, "raft")
        got = [(block.heading_path, block.text) for block in document.blocks]
        assert got == [
            ((), "Preface before any heading."),
            (
                ("Consensus",),
                "Raft elects a *leader*, which replicates the log to a diagram followers.",
            ),
            (
                ("Consensus", "Elections", "Timeouts"),
                "A follower that hears nothing becomes a candidate.",
            ),
            (
                ("Consensus", "Elections", "Timeouts"),
                "- first item wraps here\n- second item\n1. numbered",
            ),
            (("Consensus", "Elections", "Timeouts"), "| a | b |\n| 1 | 2 |"),
            (("Consensus", "Elections", "Timeouts"), "> quoted"),
            (
                ("Consensus", "Log replication"),
                "# not a heading\ndef apply(entry):\n    return entry",
            ),
            (("Setext title",), "After a rule. Visible. See https://example.com."),
        ]
        assert all(block.page is None for block in document.blocks)

    def test_front_matter_title_wins(self) -> None:
        assert load_markdown(MARKDOWN, "raft").title == "Raft notes"

    def test_title_falls_back_to_first_h1_then_file_name(self) -> None:
        assert load_markdown(b"## Sub\n\n# Main `api`\n\ntext", "x").title == "Main api"
        assert load_markdown(b"no headings", "file-name").title == "file-name"

    def test_front_matter_without_title_or_closing(self) -> None:
        assert load_markdown(b"---\nauthor: me\n---\n# Doc\n", "x").title == "Doc"
        unclosed = load_markdown(b"---\nnot front matter\n", "x")
        assert texts(unclosed.blocks) == ["not front matter"]

    def test_unclosed_fence_runs_to_the_end(self) -> None:
        document = load_markdown(b"text\n~~~\ncode line\n\nmore code", "x")
        assert texts(document.blocks) == ["text", "code line\n\nmore code"]

    def test_empty_heading_closes_deeper_sections(self) -> None:
        document = load_markdown(b"# A\n## B\nunder b\n#\nafter", "x")
        assert [block.heading_path for block in document.blocks] == [("A", "B"), ()]


HTML = b"""<!DOCTYPE html>
<html><head><title> Raft &amp; Paxos </title>
<style>p { color: red }</style><script>var x = "<p>no</p>";</script>
<meta charset="utf-8"></head>
<body>
<nav><a href="/">Home</a> <a href="/docs">Docs</a></nav>
<p>Intro &mdash; before
   any heading.</p>
<h1>Consensus <small>protocols</small></h1>
<p>Raft has a <b>leader</b>.<br>Paxos does not need one.</p>
<h2><span>Safety</span></h2>
<ul><li>one</li><li>two</li></ul>
<table><tr><th>term</th><td>3</td></tr></table>
<pre>
def apply(entry):
    return entry
</pre>
<h3><div>Nested</div> block</h3>
<p>Under nested.
<p>Unclosed paragraph.
<noscript>enable js</noscript>
<template><p>never</p></template>
<h2>Liveness</h2>
Trailing text
"""


class TestHtml:
    def test_blocks_and_heading_paths(self) -> None:
        document = load_html(HTML, "page")
        got = [(block.heading_path, block.text) for block in document.blocks]
        assert got == [
            ((), "Intro — before any heading."),
            (("Consensus protocols",), "Raft has a leader."),
            (("Consensus protocols",), "Paxos does not need one."),
            (("Consensus protocols", "Safety"), "one"),
            (("Consensus protocols", "Safety"), "two"),
            (("Consensus protocols", "Safety"), "term | 3"),
            (("Consensus protocols", "Safety"), "def apply(entry):\n    return entry"),
            (("Consensus protocols", "Safety", "Nested block"), "Under nested."),
            (("Consensus protocols", "Safety", "Nested block"), "Unclosed paragraph."),
            (("Consensus protocols", "Liveness"), "Trailing text"),
        ]

    def test_title_from_title_tag_then_h1_then_file_name(self) -> None:
        assert load_html(HTML, "page").title == "Raft & Paxos"
        assert load_html(b"<h1>Heading</h1><p>x</p>", "page").title == "Heading"
        assert load_html(b"<title>  </title><p>x</p>", "page").title == "page"

    def test_head_without_closing_tag(self) -> None:
        document = load_html(b"<html><head><title>T</title><body><p>Body text.</p>", "x")
        assert texts(document.blocks) == ["Body text."]

    def test_stray_closing_tags_are_ignored(self) -> None:
        document = load_html(b"</script></pre></h2><p>still here</p>", "x")
        assert texts(document.blocks) == ["still here"]


class TestPdf:
    def test_paragraphs_pages_and_outline_headings(self) -> None:
        data = make_pdf(
            [
                [["Raft keeps a replicated", "log of commands."], ["A leader is elect-", "ed."]],
                [["Followers apply (in order)", "what is committed."]],
                [["Snapshots compact the log."]],
            ],
            outline=[("Overview", 0, 1), ("Log", 0, 2), ("Safety", 1, 1)],
            title="Raft in brief",
        )
        document = load_pdf(data, "raft")
        assert document.title == "Raft in brief"
        assert document.format == "pdf"
        assert [(b.page, b.heading_path, b.text) for b in document.blocks] == [
            (1, ("Overview", "Log"), "Raft keeps a replicated log of commands."),
            (1, ("Overview", "Log"), "A leader is elected."),
            (2, ("Safety",), "Followers apply (in order) what is committed."),
            (3, ("Safety",), "Snapshots compact the log."),
        ]

    def test_without_outline_or_title(self) -> None:
        document = load_pdf(make_pdf([[["Only text."]], []]), "scan")
        assert document.title == "scan"
        assert [(b.page, b.heading_path, b.text) for b in document.blocks] == [
            (1, (), "Only text.")
        ]

    def test_evenly_spaced_lines_form_one_paragraph(self) -> None:
        document = load_pdf(make_pdf([[["one", "two", "three"]]]), "x")
        assert texts(document.blocks) == ["one two three"]

    def test_not_a_pdf(self) -> None:
        with pytest.raises(LoaderError, match="cannot read the PDF"):
            load_pdf(b"%PDF-1.7 this is not really a PDF", "x")

    def test_encrypted_pdf(self) -> None:
        from io import BytesIO

        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter(clone_from=PdfReader(BytesIO(make_pdf([[["secret"]]]))))
        writer.encrypt(user_password="pw", algorithm="AES-128")
        buffer = BytesIO()
        writer.write(buffer)
        with pytest.raises(LoaderError, match="the PDF is encrypted"):
            load_pdf(buffer.getvalue(), "x")

    def test_encrypted_with_an_empty_password_is_readable(self) -> None:
        from io import BytesIO

        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter(clone_from=PdfReader(BytesIO(make_pdf([[["open text"]]]))))
        writer.encrypt(user_password="", owner_password="owner", algorithm="AES-128")
        buffer = BytesIO()
        writer.write(buffer)
        assert texts(load_pdf(buffer.getvalue(), "x").blocks) == ["open text"]

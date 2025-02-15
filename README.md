# docsage

Document question answering you can measure: hybrid retrieval, reranking, cited
answers and an evaluation suite, for any OpenAI-compatible LLM endpoint.

docsage answers questions over your documents with numbered citations, and ships
an evaluation suite that measures retrieval and answer quality instead of
guessing. It talks to any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio or
a hosted API) over plain HTTP with no vendor SDK, and runs its embedding and
reranking models locally.

**Status:** early development. Ingestion works: Markdown, text, HTML and PDF
files are split into chunks that follow their headings, and stored in a single
SQLite file. Search, answers and evaluation are next on the
[roadmap](#roadmap).

## Quick start

Python 3.11 or newer:

```sh
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e .
docsage ingest notes --collection projects
```

```text
added      changelog.md             4 chunks
added      papers/raft-summary.pdf  2 chunks
added      raftkv.md                37 chunks
added      slicesim.md              43 chunks
added      todo.txt                 1 chunk
collection 'projects': 5 added, 0 updated, 0 unchanged, 0 failed
```

Ingesting again only does work for files that changed:

```text
unchanged  changelog.md
unchanged  papers/raft-summary.pdf
unchanged  raftkv.md
unchanged  slicesim.md
updated    todo.txt                 1 chunk
collection 'projects': 0 added, 1 updated, 4 unchanged, 0 failed
```

```sh
docsage collections
```

```text
collection  documents   chunks     tokens
projects            5       87     24,517
```

`docsage ingest PATH...` takes files and directories; directories are searched
recursively, skipping hidden files and folders. `--collection` (or `-c`) names
the collection, created on first use, and defaults to `default`. A file that
cannot be read is reported and the rest are still ingested; the command then
exits with status 1.

## Ingestion

### Formats

| Format | Extensions | Structure kept |
| ------ | ---------- | -------------- |
| Markdown | `.md`, `.markdown` | ATX and setext headings, paragraphs, lists, tables and fenced code; links reduced to their text; the title from front matter or the first `#` heading |
| Plain text | `.txt`, `.text` | paragraphs separated by blank lines |
| HTML | `.html`, `.htm` | `h1`-`h6` headings and block elements; scripts, styles, navigation and the `<head>` dropped; the title from `<title>` or the first `h1` |
| PDF | `.pdf` | page numbers, headings from the outline (bookmarks), paragraphs from the spacing between lines, the title from the metadata |

HTML is parsed with the standard library's `html.parser` and PDFs with
[pypdf](https://pypdf.readthedocs.io/), including encrypted PDFs that open
without a password. Text is decoded as UTF-8; bytes that are not valid UTF-8
become U+FFFD instead of failing the file.

Every block of text keeps its **heading path**, the headings it sits under from
the outermost in, so `["raftkv", "HTTP API"]` for a paragraph under
`## HTTP API`. For PDFs, a page takes the headings of the last outline entry
that starts on or before it, and a paragraph starts wherever the gap between two
lines is more than 1.5 times the page's median line spacing.

### Chunking

Documents are split into chunks of at most 400 tokens (`--chunk-tokens`), and
each chunk starts with up to 60 tokens of the one before it (`--overlap`):

- **A chunk never crosses a heading**, so its heading path describes all of it,
  and overlap never carries text into another section.
- **Paragraphs stay whole when they can.** Paragraphs are packed into a chunk
  until the next one does not fit; a paragraph that fits a chunk of its own
  starts one rather than being split.
- **Only longer paragraphs are split**, at line breaks and sentences. A sentence
  longer than the whole budget is cut between words.
- **Overlap is whole sentences**: the last sentences of the previous chunk that
  fit in the overlap budget.
- **List items, table rows and code keep their lines**, and code its indentation.

Tokens are estimated without a tokenizer: every punctuation mark is a token, and
a word one token per four characters. For English prose this counts more tokens
than BPE tokenizers do, so a budget that fits this count fits a model's.
Counts add up, so the count stored for a chunk is the sum of its sentences'.

### The store

A collection lives in a SQLite file, `docsage.db` unless `--db` or
`DOCSAGE_DB` says otherwise, with three tables: `collections`, `documents` and
`chunks`. The schema is versioned with `PRAGMA user_version`, and a newer
docsage upgrades an older file when it opens it.

A document is identified by its **source**: its path relative to the directory
that was ingested, or its file name when the file was named directly. A
collection therefore survives moving or cloning the folder it came from. Two
different files with the same source in one run are reported, and only the
first is ingested.

Each document records the SHA-256 of its file and the pipeline that chunked it
(the chunking version and options). Ingesting a file again:

- **unchanged** (same hash, same options): the file is hashed but not parsed,
  and nothing is written;
- **changed**, or chunked with other options: its chunks are replaced in one
  transaction, and it keeps its document id.

## Configuration

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `DOCSAGE_DB` | `docsage.db` | the SQLite database |
| `DOCSAGE_CHUNK_TOKENS` | `400` | the token budget of a chunk |
| `DOCSAGE_CHUNK_OVERLAP` | `60` | tokens each chunk repeats from the previous one; less than the budget |

Command-line options take precedence over the environment. An invalid value is
reported with its variable's name, and the command exits with status 2.

## Roadmap

- [x] Ingestion: loaders, chunking, the SQLite store and `docsage ingest`
- [ ] Hybrid retrieval: BM25 through SQLite FTS5, dense vectors, reciprocal rank fusion and `docsage search`
- [ ] Reranking with a cross-encoder, MMR and context assembly
- [ ] Answers with citations from any OpenAI-compatible endpoint: `docsage ask`
- [ ] An evaluation suite: retrieval and answer metrics, and `docsage eval`
- [ ] An HTTP API with FastAPI, and Docker
- [ ] A Next.js web UI, and v0.1.0

## Development

```sh
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy
pytest --cov
```

CI runs the same checks, the tests on Python 3.11 to 3.14, and the tests again
against the oldest dependency versions `pyproject.toml` allows. The tests build
their fixture documents, PDFs included, as they run.

# docsage

Document question answering you can measure: hybrid retrieval, reranking, cited
answers and an evaluation suite, for any OpenAI-compatible LLM endpoint.

docsage answers questions over your documents with numbered citations, and ships
an evaluation suite that measures retrieval and answer quality instead of
guessing. It talks to any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio or
a hosted API) over plain HTTP with no vendor SDK, and runs its embedding and
reranking models locally.

**Status:** early development. Ingestion and retrieval work: Markdown, text,
HTML and PDF files are split into chunks that follow their headings and stored
in a single SQLite file, and `docsage search` finds them by BM25, by dense
vectors or by both fused, can rerank them with a cross-encoder, drop weak and
repeated ones, and assemble what is left into a context of limited size.
Answers and evaluation are next on the [roadmap](#roadmap).

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

```sh
docsage search "ECONNRESET snapshot" --collection projects -k 3
```

```text
 1  0.0325  raft.md > Raft notes > Snapshots  [bm25 #1, dense #2]
      Servers compact their log by writing a snapshot of the state machine and discarding the entries it covers.
 2  0.0325  troubleshooting.md > Troubleshooting > Connection errors  [bm25 #2, dense #1]
      The client logs ECONNRESET when the peer closes the socket before the response is complete. Retry idempotent requests with exponential backoff.
 3  0.0159  raft.md > Raft notes > Elections  [dense #3]
      A follower that hears nothing from a leader before its election timeout becomes a candidate, increments its term and asks the other servers for votes. A candidate that gathers votes from a majority...
```

(That search ran against a folder of three small notes, `raft.md`,
`troubleshooting.md` and `todo.txt`, and the first run also printed `embedded 6
chunks with hashing-512` on standard error.)

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
`DOCSAGE_DB` says otherwise, with the tables `collections`, `documents`,
`chunks` and `embeddings`, and the FTS5 index `chunks_fts`. The schema is versioned with `PRAGMA user_version`, and a newer
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

## Search

```sh
docsage search QUERY [--collection NAME] [--mode bm25|dense|hybrid] [-k 10]
```

(Reranking, MMR and context assembly are options of the same command, described
[below](#reranking-and-context).)

Each result shows its rank, score, source with heading path and PDF pages, and
the start of the chunk. In hybrid mode it also shows the position the chunk had
in each retriever's list, so you can see which one found it. Scores mean
different things per mode (BM25 score, cosine similarity, fused score) and are
only comparable within one search.

### BM25

Lexical search uses SQLite's FTS5 with the `porter unicode61` tokenizer, so
case, accents and word endings do not matter (`elections` finds `election`).
The text and the heading path are indexed, and a word in the text counts twice
as much as one in a heading. The index is kept in step with the chunks by
triggers, so there is nothing to rebuild after ingesting.

**User text never reaches FTS5 syntax.** A query is cut into words (letters and
digits in any script), and each is sent as a double-quoted string, joined with
`OR`. `AND`, `NOT`, `NEAR`, `*`, `-`, `column:` filters, parentheses and
unbalanced quotes are therefore ordinary punctuation or ordinary words:

```sh
docsage search 'snapshot* AND ("unbalanced' -c projects --mode bm25 -k 1
```

```text
 1  2.1116  raft.md > Raft notes > Snapshots
      Servers compact their log by writing a snapshot of the state machine and discarding the entries it covers.
```

Because a query is an `OR` of its words, a chunk needs only one of them, and
BM25 ranks the chunks with the most and the rarest words first. A word found in
more than half of the chunks of a small collection scores about zero, so a
query made of such words returns chunks with scores near 0.

### Dense

Chunks are embedded by an `Embedder`, and a query is compared with every vector
of the collection as a dot product of unit vectors (cosine similarity),
exactly, with numpy. Vectors are stored in the database as float32 blobs, keyed
by chunk and by the embedder's name, so several embedders can share a database,
and a changed document loses its vectors along with its chunks. **The first
dense or hybrid search embeds whatever has no vector yet**, and is the only
slow one.

Two embedders are included, chosen with `--embedder` or `DOCSAGE_EMBEDDER`:

- `hashing` (the default): hashed words and character trigrams in 512 buckets.
  It needs no model and no download, is fully deterministic, and finds chunks
  that share words or word fragments with the query. **It knows nothing about
  meaning**, so it does not match paraphrases; it is there so that dense search
  works anywhere, and for tests.
- `fastembed`: a local ONNX model through
  [fastembed](https://qdrant.github.io/fastembed/), `BAAI/bge-small-en-v1.5`
  unless `DOCSAGE_EMBED_MODEL` names another. Install it with
  `pip install 'docsage[embed]'`; the model is downloaded on first use. This is
  the embedder meant for paraphrases.

Anything with `name`, `dimension`, `embed_documents` and `embed_query` is an
`Embedder`.

### Hybrid

Hybrid search takes the top candidates of BM25 and of dense retrieval (at least
50 each) and fuses their *ranks* with reciprocal rank fusion: a chunk scores the
sum over the lists of `weight / (k + rank)`, with `k = 60`. Only positions
matter, so BM25 scores and cosine similarities need no calibration against each
other. `--bm25-weight` and `--dense-weight` scale the lists (0 leaves one out
altogether), and `--rrf-k` sets `k`. Ties go to the chunk with the better rank
in either list, then to the one BM25 ranked first.

In the example above, `ECONNRESET` is a token no embedding model would know, so
only BM25 finds the troubleshooting note by it, while the dense list adds the
election chunk, which BM25 never returned. The tests check both directions on a
fixture corpus with a test embedder that maps synonyms to the same dimension: a
keyword-only match that dense retrieval misses, and a paraphrase-only match
with no word in common with the query that BM25 misses. Hybrid returns both.

## Reranking and context

First-stage retrieval is cheap and approximate. Three optional stages follow it,
all part of `docsage search`, and a fourth turns the results into text for a
model. Each is also a plain function: `retrieve()` runs the stages up to MMR and
`assemble_context()` builds the context.

### Reranking

```sh
pip install 'docsage[rerank]'
docsage search "how do servers compact the log" -c notes -k 3 --rerank
```

```text
 1  8.8316  raft-mirror.md > Raft summary  [bm25 #2, dense #1, first #1]
      Servers compact their log by writing a snapshot of the state machine and discarding the entries it covers. The snapshot records the index and term of the last entry it replaces, so the log can cont...
 2  8.5260  raft.md > Raft notes > Snapshots  [bm25 #3, dense #2, first #3]
      Servers compact their log by writing a snapshot of the state machine and discarding the entries it covers. The snapshot records the index and term of the last entry it replaces, so the log can cont...
 3  5.3156  raft.md > Raft notes > Snapshots  [bm25 #1, dense #3, first #2]
      Compaction runs independently on each server, once the log passes a size threshold.
```

(Run against three small notes ingested with `--chunk-tokens 60 --overlap 15`,
`raft.md`, `raft-mirror.md` and `troubleshooting.md`, with the hashing embedder
and the default cross-encoder.)

A `Reranker` scores passages against a query by reading both together, which is
far more accurate than comparing precomputed vectors and far slower, so it only
sees the best candidates of the first stage: `--candidates N`, by default three
times `-k` and at least 30. The reranker's score replaces the search's score,
and `first #N` shows where the search had put the result. The reranker reads
each chunk's heading path followed by its text.

The included implementation is a local ONNX cross-encoder through fastembed
(any model on fastembed's cross-encoder list), `Xenova/ms-marco-MiniLM-L-6-v2` unless
`DOCSAGE_RERANK_MODEL` names another. The model is downloaded on first use.
Cross-encoder scores are raw model outputs: they can be negative, and their
scale differs between models, so they only mean something within one search.
Anything with a `name` and a `score(query, passages)` method is a `Reranker`.

### Minimum score

`--min-score S` drops results scoring below `S`, so a query the collection
cannot answer returns nothing rather than its least bad chunks. It compares the
reranker's score when `--rerank` is given and the search's own score otherwise
(a BM25 score, a cosine similarity or a fused score, depending on `--mode`),
so pick the value by looking at the scores of your model on your collection.

```sh
docsage search "how do servers compact the log" -c notes -k 5 --rerank --min-score 6
```

```text
 1  8.8316  raft-mirror.md > Raft summary  [bm25 #2, dense #1, first #1]
      Servers compact their log by writing a snapshot of the state machine and discarding the entries it covers. The snapshot records the index and term of the last entry it replaces, so the log can cont...
 2  8.5260  raft.md > Raft notes > Snapshots  [bm25 #3, dense #2, first #3]
      Servers compact their log by writing a snapshot of the state machine and discarding the entries it covers. The snapshot records the index and term of the last entry it replaces, so the log can cont...
```

### Diversity: MMR

`--mmr` picks the results one by one by maximal marginal relevance: each pick
maximises `λ * relevance - (1 - λ) * redundancy`, where redundancy is the
highest cosine similarity between the candidate's vector and the vectors of the
results already picked. Relevance is rescaled to 0..1 across the candidates
first, so it can be weighed against a similarity whatever the scale of the
scores. `--mmr-lambda` (default 0.7) sets `λ`: 1 is plain ranking, lower values
favour variety. The first pick is always the most relevant. MMR compares chunks
by their vectors, so it embeds with `--embedder` even in a `bm25` search.

### Context assembly

`--context-tokens N` assembles the results into passages of at most `N` tokens
(counted as ingestion counts them) and prints them after the results:

```sh
docsage search "how do servers compact the log" -c notes -k 3 --mmr --context-tokens 200
```

```text
context: 2 passages, 168 of 200 tokens (1 duplicate dropped)

[1] raft-mirror.md > Raft summary
Servers compact their log by writing a snapshot of the state machine and discarding the entries it covers. The snapshot records the index and term of the last entry it replaces, so the log can continue from there.

A follower that is far behind receives the leader's snapshot in place of the individual entries.

[2] raft.md > Raft notes > Snapshots
A follower that is far behind receives the leader's snapshot instead of the individual entries. Installing a snapshot replaces the follower's state machine and drops any log entries the snapshot covers.

Compaction runs independently on each server, once the log passes a size threshold.
```

(The three results printed before the context are omitted here.) The results are taken
in rank order, then:

- **Near-duplicates are collapsed.** A chunk is a duplicate of a better-ranked
  one when they share at least 90% of the shorter one's word triples
  (`--duplicate-threshold`), which catches the same passage in two files, a
  copy with an edit, and a chunk contained in a longer one. The better-ranked
  copy stays; the report counts the others. In the example above,
  `raft-mirror.md` repeats the first chunk of the Snapshots section, and only
  the mirror is kept.
- **Chunks that follow each other overlap on purpose and are never duplicates
  of one another.** They are merged into one passage instead, with the repeated
  sentences removed: the overlap is found as the longest start of the second
  chunk that ends the first on a word boundary (at least 12 characters, so a
  word that happens to end one chunk and begin the next is left alone).
- **A chunk that does not fit in the budget left is skipped** and counted, and a
  smaller one further down can still use the room.
- **Neighbours fill what is left.** While budget remains, the chunks before and
  after each result are added (`--neighbours N`, default 1 on each side), nearest
  first and best result first, so a passage is not cut off mid-thought. Only
  chunks of the same section qualify, so a passage never crosses a heading.

The budget is spent chunk by chunk on the chunks' stored token counts. Merging
only removes text, so the passages can total less than the budget, never more:
the tokens printed are counted on the final text, and the tests check the bound
for twelve budgets from 1 to 1000 tokens. Tokens are estimates (see
[Chunking](#chunking)), not a model's tokenizer.

## Configuration

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `DOCSAGE_DB` | `docsage.db` | the SQLite database |
| `DOCSAGE_CHUNK_TOKENS` | `400` | the token budget of a chunk |
| `DOCSAGE_CHUNK_OVERLAP` | `60` | tokens each chunk repeats from the previous one; less than the budget |
| `DOCSAGE_EMBEDDER` | `hashing` | `hashing` or `fastembed` |
| `DOCSAGE_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | the fastembed model |
| `DOCSAGE_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | the fastembed cross-encoder used by `--rerank` |

Command-line options take precedence over the environment. An invalid value is
reported with its variable's name, and the command exits with status 2.

## Roadmap

- [x] Ingestion: loaders, chunking, the SQLite store and `docsage ingest`
- [x] Hybrid retrieval: BM25 through SQLite FTS5, dense vectors, reciprocal rank fusion and `docsage search`
- [x] Reranking with a cross-encoder, MMR and context assembly
- [ ] Answers with citations from any OpenAI-compatible endpoint: `docsage ask`
- [ ] An evaluation suite: retrieval and answer metrics, and `docsage eval`
- [ ] An HTTP API with FastAPI, and Docker
- [ ] A Next.js web UI, and v0.1.0

## Development

```sh
pip install -e ".[dev,embed,rerank]"
ruff check . && ruff format --check .
mypy
pytest --cov
```

CI runs the same checks, the tests on Python 3.11 to 3.14 with every extra
installed, and the tests again against the oldest dependency versions
`pyproject.toml` allows, once for the core and once for each extra. The tests
build their fixture documents, PDFs included, as they run. Embedders and
rerankers in the tests are fakes, so no model is downloaded; two tests only
check that the installed fastembed offers the methods docsage calls.

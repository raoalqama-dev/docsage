"""The ``docsage`` command."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from docsage import __version__
from docsage.config import ConfigError, Settings, load_settings
from docsage.ingest.chunking import ChunkingOptions
from docsage.ingest.pipeline import Outcome, ingest
from docsage.retrieval.context import DEFAULT_DUPLICATE_THRESHOLD, Context, assemble_context
from docsage.retrieval.dense import EmbedderError, create_embedder
from docsage.retrieval.fusion import DEFAULT_K
from docsage.retrieval.rerank import DEFAULT_MMR_LAMBDA, RerankerError, create_reranker
from docsage.retrieval.retrieve import retrieve
from docsage.retrieval.search import Hit, Mode
from docsage.store import Store, StoreError

EXIT_OK = 0
EXIT_FAILED = 1
"""The command ran, but some of its work failed, such as files that could not be ingested."""
EXIT_USAGE = 2
"""Bad arguments or settings, the same code argparse uses."""

DEFAULT_COLLECTION = "default"


class CommandError(Exception):
    """A problem with the request that is reported without a traceback."""


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return value


def _weight(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite number, not negative")
    return value


def _score(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a finite number")
    return value


def _unit_interval(text: str) -> float:
    value = _score(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def _threshold(text: str) -> float:
    value = _unit_interval(text)
    if value == 0.0:
        raise argparse.ArgumentTypeError("must be above 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docsage",
        description="Document question answering with hybrid retrieval and citations.",
    )
    parser.add_argument("--version", action="version", version=f"docsage {__version__}")
    parser.add_argument(
        "--db",
        type=Path,
        metavar="PATH",
        help="the SQLite database (default: $DOCSAGE_DB or docsage.db)",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    ingest_parser = commands.add_parser(
        "ingest",
        help="add files to a collection",
        description=(
            "Load Markdown, text, HTML and PDF files, split them into chunks and store "
            "them in a collection. Directories are searched recursively. Unchanged files "
            "are skipped, and changed ones replace their earlier chunks."
        ),
    )
    ingest_parser.add_argument("paths", nargs="+", type=Path, metavar="PATH")
    ingest_parser.add_argument(
        "--collection",
        "-c",
        default=DEFAULT_COLLECTION,
        metavar="NAME",
        help=f"the collection to add to, created if needed (default: {DEFAULT_COLLECTION})",
    )
    ingest_parser.add_argument(
        "--chunk-tokens",
        type=_positive_int,
        metavar="N",
        help="the token budget of a chunk (default: $DOCSAGE_CHUNK_TOKENS or 400)",
    )
    ingest_parser.add_argument(
        "--overlap",
        type=_non_negative_int,
        metavar="N",
        help="tokens repeated from the previous chunk (default: $DOCSAGE_CHUNK_OVERLAP or 60)",
    )

    search_parser = commands.add_parser(
        "search",
        help="find the chunks that match a query",
        description=(
            "Search a collection with BM25, dense vectors, or both fused with reciprocal "
            "rank fusion. Dense and hybrid searches embed any chunks that have no vector "
            "yet, which is slow only the first time. The best candidates can then be "
            "reranked with a cross-encoder, filtered by score, diversified with MMR and "
            "assembled into a context of limited size."
        ),
    )
    search_parser.add_argument("query", metavar="QUERY")
    search_parser.add_argument(
        "--collection",
        "-c",
        default=DEFAULT_COLLECTION,
        metavar="NAME",
        help=f"the collection to search (default: {DEFAULT_COLLECTION})",
    )
    search_parser.add_argument(
        "--mode",
        "-m",
        type=Mode,
        choices=list(Mode),
        default=Mode.HYBRID,
        metavar="{bm25,dense,hybrid}",
        help="how to search (default: hybrid)",
    )
    search_parser.add_argument(
        "-k", type=_positive_int, default=10, metavar="N", help="results to show (default: 10)"
    )
    search_parser.add_argument(
        "--embedder",
        choices=["hashing", "fastembed"],
        help="the embedder for dense and hybrid searches (default: $DOCSAGE_EMBEDDER or hashing)",
    )
    search_parser.add_argument(
        "--bm25-weight",
        type=_weight,
        default=1.0,
        metavar="W",
        help="the weight of BM25 in a hybrid search (default: 1)",
    )
    search_parser.add_argument(
        "--dense-weight",
        type=_weight,
        default=1.0,
        metavar="W",
        help="the weight of dense retrieval in a hybrid search (default: 1)",
    )
    search_parser.add_argument(
        "--rrf-k",
        type=_non_negative_int,
        default=DEFAULT_K,
        metavar="K",
        help=f"the rank constant of reciprocal rank fusion (default: {DEFAULT_K})",
    )
    search_parser.add_argument(
        "--rerank",
        action="store_true",
        help="rerank the candidates with a cross-encoder (needs the rerank extra)",
    )
    search_parser.add_argument(
        "--candidates",
        type=_positive_int,
        metavar="N",
        help="candidates to rerank, filter and diversify (default: 3 x k, at least 30)",
    )
    search_parser.add_argument(
        "--min-score",
        type=_score,
        metavar="S",
        help="drop results scoring below S (the reranker's score with --rerank)",
    )
    search_parser.add_argument(
        "--mmr",
        action="store_true",
        help="diversify the results with maximal marginal relevance",
    )
    search_parser.add_argument(
        "--mmr-lambda",
        type=_unit_interval,
        default=DEFAULT_MMR_LAMBDA,
        metavar="L",
        help=f"MMR's weight of relevance against variety, 0 to 1 (default: {DEFAULT_MMR_LAMBDA})",
    )
    search_parser.add_argument(
        "--context-tokens",
        type=_positive_int,
        metavar="N",
        help="also assemble the results into a context of at most N tokens",
    )
    search_parser.add_argument(
        "--neighbours",
        type=_non_negative_int,
        default=1,
        metavar="N",
        help="chunks to add on each side of a result while the context has room (default: 1)",
    )
    search_parser.add_argument(
        "--duplicate-threshold",
        type=_threshold,
        default=DEFAULT_DUPLICATE_THRESHOLD,
        metavar="T",
        help=(
            "the share of word triples that makes two chunks duplicates "
            f"(default: {DEFAULT_DUPLICATE_THRESHOLD})"
        ),
    )

    commands.add_parser(
        "collections",
        help="list collections",
        description="List the collections with their documents, chunks and tokens.",
    )
    return parser


def _db_path(args: argparse.Namespace, settings: Settings) -> Path:
    path: Path = args.db if args.db is not None else settings.db_path
    return path


def cmd_ingest(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    try:
        options = ChunkingOptions(
            max_tokens=args.chunk_tokens or settings.chunk_tokens,
            overlap_tokens=settings.chunk_overlap if args.overlap is None else args.overlap,
        )
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    with Store(_db_path(args, settings)) as store:
        report = ingest(store, args.collection, args.paths, options)

    width = max((len(result.source) for result in report.results), default=0)
    for result in report.results:
        if result.outcome is Outcome.FAILED:
            detail = result.error or ""
        elif result.outcome is Outcome.UNCHANGED:
            detail = ""
        else:
            detail = f"{result.chunks} chunk{'' if result.chunks == 1 else 's'}"
        out.write(f"{result.outcome.value:<9}  {result.source:<{width}}  {detail}".rstrip() + "\n")
    counts = ", ".join(f"{report.count(outcome)} {outcome.value}" for outcome in Outcome)
    out.write(f"collection {report.collection!r}: {counts}\n")
    return EXIT_FAILED if report.failed else EXIT_OK


SNIPPET_CHARS = 200


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 3].rstrip() + "..."


def _location(hit: Hit) -> str:
    chunk = hit.chunk.chunk
    parts = [" > ".join((hit.chunk.source, *chunk.heading_path))]
    if chunk.page is not None:
        same = chunk.page == chunk.page_end
        pages = str(chunk.page) if same else f"{chunk.page}-{chunk.page_end}"
        parts.append(f"p.{pages}")
    return "  ".join(parts)


def _ranks(hit: Hit) -> str:
    ranks = [("bm25", hit.bm25_rank), ("dense", hit.dense_rank), ("first", hit.first_rank)]
    return ", ".join(f"{name} #{rank}" for name, rank in ranks if rank is not None)


def cmd_search(args: argparse.Namespace, settings: Settings, out: TextIO, err: TextIO) -> int:
    path = _db_path(args, settings)
    if not path.exists():
        raise CommandError(f"no database at {path}; run docsage ingest first")
    embedder = None
    if args.mode is not Mode.BM25 or args.mmr:
        embedder = create_embedder(args.embedder or settings.embedder, settings.embed_model)
    reranker = create_reranker(settings.rerank_model) if args.rerank else None
    with Store(path) as store:
        collection = store.collection(args.collection)
        if collection is None:
            raise CommandError(f"no collection {args.collection!r}")
        try:
            result = retrieve(
                store,
                collection,
                args.query,
                mode=args.mode,
                k=args.k,
                embedder=embedder,
                bm25_weight=args.bm25_weight,
                dense_weight=args.dense_weight,
                rrf_k=args.rrf_k,
                reranker=reranker,
                candidates=args.candidates,
                min_score=args.min_score,
                mmr_lambda=args.mmr_lambda if args.mmr else None,
            )
        except ValueError as exc:
            raise CommandError(str(exc)) from None
        context = None
        if args.context_tokens and result.hits:
            context = assemble_context(
                store,
                result.hits,
                args.context_tokens,
                neighbours=args.neighbours,
                duplicate_threshold=args.duplicate_threshold,
            )
    if embedder and result.embedded:
        err.write(f"embedded {result.embedded} chunks with {embedder.name}\n")
    if not result.hits:
        out.write("no results\n")
        return EXIT_OK
    show_ranks = args.mode is Mode.HYBRID or reranker is not None
    for position, hit in enumerate(result.hits, start=1):
        detail = f"  [{_ranks(hit)}]" if show_ranks else ""
        out.write(f"{position:>2}  {hit.score:.4f}  {_location(hit)}{detail}\n")
        out.write(f"      {_snippet(hit.chunk.chunk.text)}\n")
    if context is not None:
        _write_context(context, out)
    return EXIT_OK


def _write_context(context: Context, out: TextIO) -> None:
    count = len(context.passages)
    notes = []
    if context.duplicates:
        notes.append(f"{context.duplicates} duplicate{'' if context.duplicates == 1 else 's'}")
    if context.skipped:
        notes.append(f"{context.skipped} too big for the budget")
    extra = f" ({', '.join(notes)} dropped)" if notes else ""
    out.write(
        f"\ncontext: {count} passage{'' if count == 1 else 's'}, "
        f"{context.tokens} of {context.budget} tokens{extra}\n"
    )
    for number, passage in enumerate(context.passages, start=1):
        pages = ""
        if passage.page is not None:
            same = passage.page == passage.page_end
            pages = f"  p.{passage.page}" if same else f"  p.{passage.page}-{passage.page_end}"
        path = " > ".join((passage.source, *passage.heading_path))
        out.write(f"\n[{number}] {path}{pages}\n{passage.text}\n")


def cmd_collections(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    path = _db_path(args, settings)
    if not path.exists():
        raise CommandError(f"no database at {path}; run docsage ingest first")
    with Store(path) as store:
        stats = store.collection_stats()
    if not stats:
        out.write("no collections\n")
        return EXIT_OK
    width = max(len("collection"), *(len(row.name) for row in stats))
    out.write(f"{'collection':<{width}}  {'documents':>9}  {'chunks':>7}  {'tokens':>9}\n")
    for row in stats:
        out.write(f"{row.name:<{width}}  {row.documents:>9,}  {row.chunks:>7,}  {row.tokens:>9,}\n")
    return EXIT_OK


def _with_err(
    command: Callable[[argparse.Namespace, Settings, TextIO], int],
) -> Callable[[argparse.Namespace, Settings, TextIO, TextIO], int]:
    return lambda args, settings, out, err: command(args, settings, out)


COMMANDS: dict[str, Callable[[argparse.Namespace, Settings, TextIO, TextIO], int]] = {
    "ingest": _with_err(cmd_ingest),
    "search": cmd_search,
    "collections": _with_err(cmd_collections),
}


def main(
    argv: Sequence[str] | None = None, out: TextIO | None = None, err: TextIO | None = None
) -> int:
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings()
        return COMMANDS[args.command](args, settings, out, err)
    except (CommandError, ConfigError, EmbedderError, RerankerError, StoreError) as exc:
        err.write(f"docsage: {exc}\n")
        return EXIT_USAGE

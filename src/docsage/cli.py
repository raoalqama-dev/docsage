"""The ``docsage`` command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from docsage import __version__
from docsage.config import ConfigError, Settings, load_settings
from docsage.ingest.chunking import ChunkingOptions
from docsage.ingest.pipeline import Outcome, ingest
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


COMMANDS: dict[str, Callable[[argparse.Namespace, Settings, TextIO], int]] = {
    "ingest": cmd_ingest,
    "collections": cmd_collections,
}


def main(
    argv: Sequence[str] | None = None, out: TextIO | None = None, err: TextIO | None = None
) -> int:
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings()
        return COMMANDS[args.command](args, settings, out)
    except (CommandError, ConfigError, StoreError) as exc:
        err.write(f"docsage: {exc}\n")
        return EXIT_USAGE

"""BM25 search over the SQLite FTS5 index.

User text never reaches FTS5 syntax. Queries are cut into words, and every
word is sent as a quoted string, so ``AND``, ``NEAR``, ``*``, ``-``, ``:`` and
unbalanced quotes in a question are just punctuation or words to look up.
"""

from __future__ import annotations

import re

from docsage.store import Collection, Store

#: Words are runs of letters and digits in any script; "_" separates them.
_WORD_RE = re.compile(r"[^\W_]+")

#: More terms than this add cost without changing the ranking much.
MAX_TERMS = 64


def query_terms(query: str) -> list[str]:
    """The distinct words of ``query`` in order, lowercased and capped at :data:`MAX_TERMS`."""
    seen: dict[str, None] = {}
    for word in _WORD_RE.findall(query.lower()):
        seen.setdefault(word)
    return list(seen)[:MAX_TERMS]


def build_match(query: str) -> str | None:
    """An FTS5 expression matching any word of ``query``, or ``None`` if it has no words.

    Every word is double-quoted. A word cannot contain a quote, because words
    are letters and digits only, so nothing can escape the quoting.
    """
    terms = query_terms(query)
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


def bm25_search(
    store: Store, collection: Collection, query: str, limit: int
) -> list[tuple[int, float]]:
    """Chunk ids of ``collection`` with BM25 scores for ``query``, best first."""
    match = build_match(query)
    if match is None or limit < 1:
        return []
    return store.fts_search(collection, match, limit)

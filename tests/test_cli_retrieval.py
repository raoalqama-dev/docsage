"""``docsage search`` with reranking, score filtering, MMR and context assembly."""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pytest

from docsage.cli import EXIT_OK, EXIT_USAGE, main
from docsage.retrieval.rerank import Reranker
from fixtures import prose
from rerankers import OverlapReranker

DUPLICATE = (
    "Servers compact their log by writing a snapshot of the state machine and discarding "
    "the entries it covers. A follower that is far behind receives the snapshot instead."
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DOCSAGE_DB",
        "DOCSAGE_CHUNK_TOKENS",
        "DOCSAGE_CHUNK_OVERLAP",
        "DOCSAGE_EMBEDDER",
        "DOCSAGE_EMBED_MODEL",
        "DOCSAGE_RERANK_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out, err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def db(tmp_path: Path) -> str:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "raft.md").write_text("# Raft\n\n" + prose(24), encoding="utf-8")
    (root / "snapshots.md").write_text(f"# Snapshots\n\n{DUPLICATE}", encoding="utf-8")
    (root / "mirror.md").write_text(
        f"# Mirror\n\n{DUPLICATE.replace('instead', 'in its place')}", encoding="utf-8"
    )
    (root / "cats.md").write_text("# Cats\n\nKittens nap in the sun.", encoding="utf-8")
    db = str(tmp_path / "x.db")
    ingested = run(
        "--db", db, "ingest", str(root), "-c", "demo", "--chunk-tokens", "80", "--overlap", "30"
    )
    assert ingested[0] == EXIT_OK
    return db


@pytest.fixture
def fake_reranker(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replaces the cross-encoder with a fake; the list records the models asked for."""
    models: list[str] = []

    def create(model: str) -> Reranker:
        models.append(model)
        return OverlapReranker()

    monkeypatch.setattr("docsage.cli.create_reranker", create)
    return models


def result_sources(out: str) -> list[str]:
    return [line.split()[2] for line in result_lines(out)]


def result_lines(out: str) -> list[str]:
    """The numbered result lines of the output, before any context."""
    return [line for line in out.split("\ncontext:")[0].splitlines() if re.match(r" *\d+  ", line)]


class TestRerank:
    def test_results_show_where_the_search_had_put_them(
        self, db: str, fake_reranker: list[str]
    ) -> None:
        code, out, err = run(
            "--db", db, "search", "kittens nap", "-c", "demo", "-m", "bm25", "--rerank"
        )
        assert (code, err) == (EXIT_OK, "")
        first = result_lines(out)[0]
        assert first.startswith(" 1  2.0000  cats.md > Cats")
        assert first.endswith("[bm25 #1, first #1]")
        assert fake_reranker == ["Xenova/ms-marco-MiniLM-L-6-v2"]

    def test_the_model_comes_from_the_environment(
        self, db: str, fake_reranker: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOCSAGE_RERANK_MODEL", "org/small")
        run("--db", db, "search", "nap", "-c", "demo", "-m", "bm25", "--rerank")
        assert fake_reranker == ["org/small"]

    def test_hybrid_shows_every_rank(self, db: str, fake_reranker: list[str]) -> None:
        _, out, _ = run("--db", db, "search", "kittens nap", "-c", "demo", "--rerank")
        assert re.search(r"\[bm25 #\d+, dense #\d+, first #\d+\]", result_lines(out)[0])

    def test_a_missing_extra_is_explained(self, db: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", None)
        code, _, err = run("--db", db, "search", "nap", "-c", "demo", "-m", "bm25", "--rerank")
        assert code == EXIT_USAGE
        assert err == (
            "docsage: the cross-encoder reranker needs the rerank extra: "
            "pip install 'docsage[rerank]'\n"
        )

    def test_candidates(self, db: str, fake_reranker: list[str]) -> None:
        _, out, _ = run(
            "--db", db, "search", "the", "-c", "demo", "-m", "bm25", "--rerank", "-k", "1",
            "--candidates", "3",
        )  # fmt: skip
        assert len(result_lines(out)) == 1

    def test_candidates_below_k_are_rejected(self, db: str) -> None:
        code, _, err = run(
            "--db", db, "search", "the", "-c", "demo", "-m", "bm25", "-k", "5", "--candidates", "2"
        )
        assert code == EXIT_USAGE
        assert err == "docsage: candidates (2) must be at least k (5)\n"


class TestMinScore:
    def test_filters_by_the_rerankers_score(self, db: str, fake_reranker: list[str]) -> None:
        _, out, _ = run(
            "--db", db, "search", "kittens nap sun", "-c", "demo", "-m", "bm25", "--rerank",
            "--min-score", "3",
        )  # fmt: skip
        assert [line.split()[2] for line in result_lines(out)] == ["cats.md"]

    def test_everything_below_it_leaves_no_results(self, db: str) -> None:
        code, out, _ = run(
            "--db", db, "search", "nap", "-c", "demo", "-m", "bm25", "--min-score", "1000"
        )
        assert (code, out) == (EXIT_OK, "no results\n")


class TestMmr:
    def test_the_copies_of_a_passage_give_way_to_other_results(self, db: str) -> None:
        args = ["--db", db, "search", "snapshot compact", "-c", "demo", "-m", "bm25"]
        _, plain, _ = run(*args, "-k", "2")
        _, diverse, _ = run(*args, "-k", "2", "--mmr", "--mmr-lambda", "0.2")
        assert set(result_sources(plain)) == {"snapshots.md", "mirror.md"}
        assert "mirror.md" not in diverse
        assert "snapshots.md" in diverse

    def test_mmr_embeds_even_in_a_bm25_search(self, db: str) -> None:
        code, _, err = run("--db", db, "search", "nap", "-c", "demo", "-m", "bm25", "--mmr")
        assert code == EXIT_OK
        assert err == "embedded 13 chunks with hashing-512\n"

    @pytest.mark.parametrize("value", ["-0.1", "1.5", "nan", "x"])
    def test_lambda_must_be_a_fraction(
        self, value: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            main(["search", "x", "--mmr-lambda", value])
        assert "--mmr-lambda" in capsys.readouterr().err


class TestContext:
    def test_the_context_follows_the_results(self, db: str) -> None:
        code, out, _ = run(
            "--db", db, "search", "Sentence 12 says", "-c", "demo", "-m", "bm25", "-k", "1",
            "--context-tokens", "400",
        )  # fmt: skip
        assert code == EXIT_OK
        header = re.search(r"^context: 1 passage, (\d+) of 400 tokens$", out, re.MULTILINE)
        assert header is not None
        assert int(header.group(1)) <= 400
        assert "\n[1] raft.md > Raft\n" in out

    def test_neighbours(self, db: str) -> None:
        args = [
            "--db", db, "search", "Sentence 12 says", "-c", "demo", "-m", "bm25", "-k", "1",
            "--context-tokens", "1000",
        ]  # fmt: skip
        _, alone, _ = run(*args, "--neighbours", "0")
        _, widened, _ = run(*args)
        assert len(widened) > len(alone)
        assert widened.count("Sentence 12 says") == 2  # once in the result, once in the passage
        numbers = [int(n) for n in re.findall(r"Sentence (\d+) says", widened.split("\n[1]")[1])]
        assert numbers == sorted(set(numbers))

    def test_the_budget_is_respected_and_reported(self, db: str) -> None:
        _, out, _ = run(
            "--db", db, "search", "Sentence 12 says", "-c", "demo", "-m", "bm25", "-k", "1",
            "--context-tokens", "10",
        )  # fmt: skip
        assert "context: 0 passages, 0 of 10 tokens (1 too big for the budget dropped)" in out

    def test_duplicates_are_reported(self, db: str) -> None:
        _, out, _ = run(
            "--db", db, "search", "snapshot compact", "-c", "demo", "-m", "bm25", "-k", "2",
            "--context-tokens", "400", "--neighbours", "0",
        )  # fmt: skip
        assert "context: 1 passage, " in out
        assert "(1 duplicate dropped)" in out
        assert out.count("Servers compact their log") == 3  # two results, one passage

    def test_a_stricter_threshold_keeps_the_copy(self, db: str) -> None:
        _, out, _ = run(
            "--db", db, "search", "snapshot compact", "-c", "demo", "-m", "bm25", "-k", "2",
            "--context-tokens", "400", "--neighbours", "0", "--duplicate-threshold", "1",
        )  # fmt: skip
        assert "context: 2 passages, " in out
        assert "duplicate" not in out.split("context:")[1].splitlines()[0]

    def test_pdf_pages_are_shown(self, tmp_path: Path) -> None:
        from fixtures import make_pdf

        root = tmp_path / "pdfs"
        root.mkdir()
        (root / "paper.pdf").write_bytes(make_pdf([[["Page one."]], [["Quorums overlap."]]]))
        db = str(tmp_path / "p.db")
        run("--db", db, "ingest", str(root), "-c", "p")
        _, out, _ = run(
            "--db", db, "search", "quorums", "-c", "p", "-m", "bm25", "--context-tokens", "100"
        )
        assert "\n[1] paper.pdf  p.1-2\n" in out

    def test_no_results_means_no_context(self, db: str) -> None:
        _, out, _ = run(
            "--db",
            db,
            "search",
            "kubernetes",
            "-c",
            "demo",
            "-m",
            "bm25",
            "--context-tokens",
            "100",
        )
        assert out == "no results\n"

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--context-tokens", "0"], "must be at least 1"),
            (["--neighbours", "-1"], "must not be negative"),
            (["--duplicate-threshold", "0"], "must be above 0"),
            (["--duplicate-threshold", "1.5"], "must be between 0 and 1"),
            (["--min-score", "nan"], "must be a finite number"),
            (["--min-score", "x"], "not a number: 'x'"),
        ],
    )
    def test_bad_arguments(
        self, capsys: pytest.CaptureFixture[str], args: list[str], message: str
    ) -> None:
        with pytest.raises(SystemExit) as caught:
            main(["search", "x", *args])
        assert caught.value.code == EXIT_USAGE
        assert message in capsys.readouterr().err

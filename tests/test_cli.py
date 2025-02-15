from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from docsage import __version__
from docsage.cli import EXIT_FAILED, EXIT_OK, EXIT_USAGE, main
from docsage.store import Store


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DOCSAGE_DB", "DOCSAGE_CHUNK_TOKENS", "DOCSAGE_CHUNK_OVERLAP"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("# Alpha\n\nFirst document.", encoding="utf-8")
    (root / "b.txt").write_text("Second document.", encoding="utf-8")
    return root


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out, err)
    return code, out.getvalue(), err.getvalue()


class TestIngest:
    def test_ingest_then_again(self, tmp_path: Path, corpus: Path) -> None:
        db = str(tmp_path / "x.db")
        code, out, err = run("--db", db, "ingest", str(corpus), "--collection", "demo")
        assert (code, err) == (EXIT_OK, "")
        assert out.splitlines() == [
            "added      a.md   1 chunk",
            "added      b.txt  1 chunk",
            "collection 'demo': 2 added, 0 updated, 0 unchanged, 0 failed",
        ]
        code, out, _ = run("--db", db, "ingest", str(corpus), "-c", "demo")
        assert code == EXIT_OK
        assert out.splitlines() == [
            "unchanged  a.md",
            "unchanged  b.txt",
            "collection 'demo': 0 added, 0 updated, 2 unchanged, 0 failed",
        ]

    def test_default_collection_and_database_from_the_environment(
        self, tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOCSAGE_DB", str(tmp_path / "env.db"))
        code, out, _ = run("ingest", str(corpus / "a.md"))
        assert code == EXIT_OK
        assert out.splitlines()[-1].startswith("collection 'default': 1 added")
        with Store(tmp_path / "env.db") as store:
            assert store.collection("default") is not None

    def test_chunking_options(self, tmp_path: Path) -> None:
        (tmp_path / "long.txt").write_text(
            " ".join(f"Sentence {n} is here." for n in range(40)), encoding="utf-8"
        )
        db = str(tmp_path / "x.db")
        code, out, _ = run(
            "--db", db, "ingest", str(tmp_path / "long.txt"), "--chunk-tokens", "30",
            "--overlap", "0",
        )  # fmt: skip
        assert code == EXIT_OK
        assert out.startswith("added      long.txt  ")
        chunks = int(out.split()[2])
        assert chunks > 5

    def test_a_failed_file_exits_1(self, tmp_path: Path, corpus: Path) -> None:
        (corpus / "bad.pdf").write_bytes(b"not a pdf")
        code, out, _ = run("--db", str(tmp_path / "x.db"), "ingest", str(corpus))
        assert code == EXIT_FAILED
        assert "failed     bad.pdf  cannot read the PDF" in out
        assert "2 added, 0 updated, 0 unchanged, 1 failed" in out

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--chunk-tokens", "50", "--overlap", "50"], "the overlap must be between 0"),
            (["--collection", "bad name"], "invalid collection name 'bad name'"),
        ],
    )
    def test_invalid_requests_exit_2(
        self, tmp_path: Path, corpus: Path, args: list[str], message: str
    ) -> None:
        code, _, err = run("--db", str(tmp_path / "x.db"), "ingest", str(corpus), *args)
        assert code == EXIT_USAGE
        assert err.startswith("docsage: ")
        assert message in err

    def test_invalid_environment_exits_2(
        self, tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOCSAGE_CHUNK_TOKENS", "many")
        code, _, err = run("--db", str(tmp_path / "x.db"), "ingest", str(corpus))
        assert code == EXIT_USAGE
        assert "DOCSAGE_CHUNK_TOKENS: Input should be a valid integer" in err

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["--chunk-tokens", "0"], "must be at least 1"),
            (["--chunk-tokens", "x"], "not an integer: 'x'"),
            (["--overlap", "-1"], "must not be negative"),
            (["--overlap", "1.5"], "not an integer: '1.5'"),
        ],
    )
    def test_bad_arguments(
        self, corpus: Path, capsys: pytest.CaptureFixture[str], args: list[str], message: str
    ) -> None:
        with pytest.raises(SystemExit) as caught:
            main(["ingest", str(corpus), *args])
        assert caught.value.code == EXIT_USAGE
        assert message in capsys.readouterr().err


class TestCollections:
    def test_lists_collections(self, tmp_path: Path, corpus: Path) -> None:
        db = str(tmp_path / "x.db")
        run("--db", db, "ingest", str(corpus), "-c", "demo")
        run("--db", db, "ingest", str(corpus / "a.md"), "-c", "alpha")
        code, out, _ = run("--db", db, "collections")
        assert code == EXIT_OK
        assert out.splitlines() == [
            "collection  documents   chunks     tokens",
            "alpha               1        1          5",
            "demo                2        2         10",
        ]

    def test_empty_database(self, tmp_path: Path) -> None:
        Store(tmp_path / "x.db").close()
        assert run("--db", str(tmp_path / "x.db"), "collections") == (
            EXIT_OK,
            "no collections\n",
            "",
        )

    def test_missing_database_is_not_created(self, tmp_path: Path) -> None:
        code, _, err = run("--db", str(tmp_path / "none.db"), "collections")
        assert code == EXIT_USAGE
        assert "no database at" in err
        assert not (tmp_path / "none.db").exists()


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out == f"docsage {__version__}\n"


def test_python_dash_m(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "docsage", "--db", str(tmp_path / "x.db"), "collections"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == EXIT_USAGE
    assert "no database at" in result.stderr

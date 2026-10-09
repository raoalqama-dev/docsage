from __future__ import annotations

from pathlib import Path

import pytest

from docsage.config import ConfigError, Settings, load_settings


def test_defaults() -> None:
    assert load_settings({}) == Settings(
        db_path=Path("docsage.db"),
        chunk_tokens=400,
        chunk_overlap=60,
        embedder="hashing",
        embed_model="BAAI/bge-small-en-v1.5",
    )


def test_values_from_the_environment() -> None:
    settings = load_settings(
        {"DOCSAGE_DB": "data/x.db", "DOCSAGE_CHUNK_TOKENS": "256", "DOCSAGE_CHUNK_OVERLAP": "0"}
    )
    assert settings == Settings(db_path=Path("data/x.db"), chunk_tokens=256, chunk_overlap=0)


def test_embedder_settings() -> None:
    settings = load_settings({"DOCSAGE_EMBEDDER": "fastembed", "DOCSAGE_EMBED_MODEL": "org/m"})
    assert (settings.embedder, settings.embed_model) == ("fastembed", "org/m")


def test_an_unknown_embedder_is_named() -> None:
    with pytest.raises(ConfigError, match="DOCSAGE_EMBEDDER: Input should be 'hashing' or"):
        load_settings({"DOCSAGE_EMBEDDER": "openai"})


def test_empty_variables_are_unset() -> None:
    assert load_settings({"DOCSAGE_CHUNK_TOKENS": ""}).chunk_tokens == 400


def test_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCSAGE_DB", "from-env.db")
    assert load_settings().db_path == Path("from-env.db")


def test_every_problem_names_its_variable() -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings({"DOCSAGE_CHUNK_TOKENS": "lots", "DOCSAGE_CHUNK_OVERLAP": "-1"})
    message = str(caught.value)
    assert message.startswith("invalid settings:")
    assert "DOCSAGE_CHUNK_TOKENS: Input should be a valid integer" in message
    assert "DOCSAGE_CHUNK_OVERLAP: Input should be greater than or equal to 0" in message


def test_overlap_must_be_below_the_chunk_size() -> None:
    with pytest.raises(
        ConfigError,
        match=r"DOCSAGE_CHUNK_OVERLAP \(100\) must be smaller than DOCSAGE_CHUNK_TOKENS \(100\)",
    ):
        load_settings({"DOCSAGE_CHUNK_TOKENS": "100", "DOCSAGE_CHUNK_OVERLAP": "100"})


def test_settings_are_frozen() -> None:
    with pytest.raises(ValueError, match="frozen"):
        Settings().chunk_tokens = 5  # type: ignore[misc]

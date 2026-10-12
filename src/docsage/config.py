"""Settings read from ``DOCSAGE_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from docsage.retrieval.dense import DEFAULT_FASTEMBED_MODEL
from docsage.retrieval.rerank import DEFAULT_RERANK_MODEL

#: Each setting and the environment variable it is read from.
ENV_VARS: dict[str, str] = {
    "db_path": "DOCSAGE_DB",
    "chunk_tokens": "DOCSAGE_CHUNK_TOKENS",
    "chunk_overlap": "DOCSAGE_CHUNK_OVERLAP",
    "embedder": "DOCSAGE_EMBEDDER",
    "embed_model": "DOCSAGE_EMBED_MODEL",
    "rerank_model": "DOCSAGE_RERANK_MODEL",
}


class ConfigError(Exception):
    """Environment variables with invalid values; the message lists every problem."""


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    db_path: Path = Path("docsage.db")
    chunk_tokens: int = Field(default=400, ge=1)
    chunk_overlap: int = Field(default=60, ge=0)
    embedder: Literal["hashing", "fastembed"] = "hashing"
    embed_model: str = Field(default=DEFAULT_FASTEMBED_MODEL, min_length=1)
    """The fastembed model; the hashing embedder has no model."""
    rerank_model: str = Field(default=DEFAULT_RERANK_MODEL, min_length=1)
    """The fastembed cross-encoder used by ``--rerank``."""

    @model_validator(mode="after")
    def _overlap_below_chunk_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError(
                f"{ENV_VARS['chunk_overlap']} ({self.chunk_overlap}) must be smaller than "
                f"{ENV_VARS['chunk_tokens']} ({self.chunk_tokens})"
            )
        return self


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Settings from ``environ`` (the process environment by default); unset ones keep defaults."""
    environ = os.environ if environ is None else environ
    values = {field: environ[var] for field, var in ENV_VARS.items() if environ.get(var, "")}
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            field = str(error["loc"][0]) if error["loc"] else ""
            name = ENV_VARS.get(field)
            message = error["msg"].removeprefix("Value error, ")
            problems.append(f"{name}: {message}" if name else message)
        raise ConfigError("invalid settings:\n  " + "\n  ".join(problems)) from exc

"""Settings read from ``DOCSAGE_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

#: Each setting and the environment variable it is read from.
ENV_VARS: dict[str, str] = {
    "db_path": "DOCSAGE_DB",
    "chunk_tokens": "DOCSAGE_CHUNK_TOKENS",
    "chunk_overlap": "DOCSAGE_CHUNK_OVERLAP",
}


class ConfigError(Exception):
    """Environment variables with invalid values; the message lists every problem."""


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    db_path: Path = Path("docsage.db")
    chunk_tokens: int = Field(default=400, ge=1)
    chunk_overlap: int = Field(default=60, ge=0)

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

"""Карточка ВКР и внутренние результаты ранжирования."""

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, field_validator


RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Thesis(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)

    id: RequiredText
    title: RequiredText
    abstract: RequiredText
    url: HttpUrl
    keywords: list[str] = Field(default_factory=list)
    author: str | None = None
    year: int | None = None

    @field_validator("url", mode="before")
    @classmethod
    def strip_url(cls, value: object) -> object:
        # str_strip_whitespace does not apply to Pydantic URL types.
        return value.strip() if isinstance(value, str) else value

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if "/" in value:
            raise ValueError("id must not contain '/'")
        return value


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float


@dataclass(frozen=True)
class FusedHit(SearchHit):
    bm25_score: float | None
    semantic_score: float | None
    bm25_rank: int | None
    semantic_rank: int | None

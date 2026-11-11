"""Контракты запросов и ответов без зависимости от FastAPI."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from domain.models import Thesis


class HealthResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: Literal["ok"]
    index_loaded: Literal[True]
    documents_count: int = Field(gt=0)
    embedding_model: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]
    top_k: int = Field(default=10, ge=1, le=100)
    debug: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not any(character.isalnum() for character in value):
            raise ValueError("query must contain a letter or digit")
        return value


class SearchResult(Thesis):
    score: float = Field(allow_inf_nan=False)


class DebugSearchResult(SearchResult):
    # Nullable, but required: a missing channel is explicitly represented by null.
    bm25_score: float | None = Field(allow_inf_nan=False)
    semantic_score: float | None = Field(allow_inf_nan=False)
    bm25_rank: int | None = Field(ge=1)
    semantic_rank: int | None = Field(ge=1)


class SearchResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    query: str
    count: int = Field(ge=0)
    results: list[DebugSearchResult | SearchResult]

    @model_validator(mode="after")
    def validate_count(self) -> Self:
        if self.count != len(self.results):
            raise ValueError("count must equal the number of results")
        return self

    @classmethod
    def from_results(
        cls, request: SearchRequest, results: list[DebugSearchResult]
    ) -> Self:
        """Оформить готовую выдачу с учётом debug, сохранив порядок и score."""
        public_results: list[DebugSearchResult | SearchResult]
        if request.debug:
            public_results = list(results)
        else:
            public_results = [
                SearchResult.model_validate(
                    result.model_dump(include=set(SearchResult.model_fields))
                )
                for result in results
            ]
        return cls(query=request.query, count=len(public_results), results=public_results)

"""Контракты запросов и ответов без зависимости от FastAPI."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from domain.models import Thesis


class HealthResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: Literal["ok"]
    index_loaded: Literal[True]
    documents_count: int = Field(gt=0)
    search_mode: Literal["bm25", "hybrid"]
    embedding_model: str | None


class NotReadyHealthResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: Literal["not_ready"] = "not_ready"
    index_loaded: Literal[False] = False
    documents_count: Literal[0] = 0
    search_mode: Literal["bm25", "hybrid"] = "bm25"
    embedding_model: str | None = None


class ReplaceThesesRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", json_schema_extra={
        "examples": [{"cards": [{
            "id": "hse-1162011207",
            "title": "LLM для описания спутниковых данных",
            "abstract": (
                "Предложена система ответов на вопросы о наблюдениях Земли, где языковая модель "
                "выбирает данные и необходимые вычисления. План превращается в граф операций, "
                "исполняемых средствами Rasdaman и инструментами обработки растров. Подход "
                "проверяется на EarthBench: задачи затрагивают спектральные показатели, температуру, "
                "изменения во времени и распознавание сцен. Разделение планирования и вычислений "
                "позволяет локализовать ошибки обработки вопроса."
            ),
            "keywords": ["геопространственная аналитика", "LLM-планирование", "Rasdaman", "EarthBench", "обработка растров"],
            "author": "Шехзад",
            "year": 2026,
            "url": "https://www.hse.ru/edu/vkr/1162011207",
        }]}],
    })

    cards: list[Thesis] = Field(min_length=1, description="Новый полный корпус; ID должны быть уникальными")

    @field_validator("cards")
    @classmethod
    def unique_ids(cls, cards: list[Thesis]) -> list[Thesis]:
        if len({card.id for card in cards}) != len(cards):
            raise ValueError("Card IDs must be unique after normalization")
        return cards


class ReplaceThesesResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: Literal["ok"] = "ok"
    documents_count: int = Field(gt=0)
    search_mode: Literal["bm25", "hybrid"] = "bm25"


class SearchRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    search_mode: Literal["bm25", "hybrid"] = Field(
        default="bm25", description="bm25 работает сразу; hybrid требует подключённой Ollama и векторного индекса",
    )
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
        Field(description="Тема или слова из карточки ВКР", examples=["Rasdaman"]),
    ]
    top_k: int = Field(default=10, ge=1, le=100, description="Максимальное число результатов")
    debug: bool = Field(default=False, description="Добавить оценки и позиции отдельных поисковых каналов")

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

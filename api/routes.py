"""Синхронные HTTP-обработчики выполняются FastAPI в пуле потоков."""

from typing import Annotated
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse, NotReadyHealthResponse, ReplaceThesesRequest,
    ReplaceThesesResponse, SearchRequest, SearchResponse,
)
from domain.models import Thesis
from external.embedder import EncoderError
from services.search import BM25SearchService, SearchService


router = APIRouter()


def get_search_service(request: Request) -> SearchService | BM25SearchService:
    service = request.app.state.search_service
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Корпус не загружен. Загрузите карточки через PUT /theses",
        )
    return service


SearchDependency = Annotated[SearchService | BM25SearchService, Depends(get_search_service)]


@router.post("/theses/search", response_model=SearchResponse)
def search_theses(request: SearchRequest, service: SearchDependency) -> SearchResponse:
    try:
        return service.search(request)
    except EncoderError as error:
        raise HTTPException(status_code=503, detail="Энкодер временно недоступен") from error


@router.get("/theses/{thesis_id}", response_model=Thesis)
def get_thesis(thesis_id: str, service: SearchDependency) -> Thesis:
    card = service.get_thesis(thesis_id)
    if card is None:
        raise HTTPException(status_code=404, detail="Работа не найдена")
    return card


@router.put(
    "/theses", response_model=ReplaceThesesResponse,
    responses={409: {"description": "Импорт недоступен в hybrid"},
               503: {"description": "Хранилище недоступно"}},
)
def replace_theses(body: ReplaceThesesRequest, request: Request) -> ReplaceThesesResponse:
    state = request.app.state
    if state.search_mode == "hybrid":
        raise HTTPException(status_code=409, detail="Импорт доступен только в режиме bm25")
    if state.storage is None:
        raise HTTPException(status_code=503, detail="Хранилище не инициализировано")
    cards = tuple(body.cards)
    with state.import_lock:
        service = BM25SearchService.from_cards(cards)
        try:
            state.storage.replace_cards(cards)
        except (OSError, sqlite3.OperationalError, sqlite3.IntegrityError) as error:
            raise HTTPException(
                status_code=503, detail="Хранилище временно недоступно; корпус не заменён",
            ) from error
        state.search_service = service
    return ReplaceThesesResponse(documents_count=len(cards))


@router.get(
    "/health", response_model=HealthResponse,
    responses={503: {"model": NotReadyHealthResponse, "description": "Корпус не загружен"}},
)
def health(request: Request) -> HealthResponse | JSONResponse:
    service = request.app.state.search_service
    if service is None:
        return JSONResponse(status_code=503, content=NotReadyHealthResponse().model_dump())
    return HealthResponse.model_validate(service.health())

"""Синхронные HTTP-обработчики выполняются FastAPI в пуле потоков."""

from typing import Annotated
from itertools import count
import logging
import sqlite3
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse, NotReadyHealthResponse, ReplaceThesesRequest,
    ReplaceThesesResponse, SearchRequest, SearchResponse,
)
from domain.models import Thesis
from external.embedder import EncoderError
from services.indexing import prepare_index
from services.search import BM25SearchService, SearchService


router = APIRouter()
logger = logging.getLogger(__name__)
_import_ids = count(1)


def get_search_service(request: Request) -> SearchService | BM25SearchService:
    service = request.app.state.search_service
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Корпус не загружен. Загрузите карточки через PUT /theses",
        )
    return service


SearchDependency = Annotated[SearchService | BM25SearchService, Depends(get_search_service)]


@router.post(
    "/theses/search", response_model=SearchResponse,
    responses={503: {"description": "Корпус или эмбеддер недоступен"}},
)
def search_theses(body: SearchRequest, request: Request) -> SearchResponse:
    service = request.app.state.search_service
    if body.search_mode == "hybrid" and request.app.state.encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Гибридный поиск недоступен: эмбеддер не подключён",
        )
    if service is None:
        service = get_search_service(request)
    try:
        return service.search(body)
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
    responses={503: {"description": "Хранилище или энкодер недоступен; корпус не заменён"}},
)
def replace_theses(body: ReplaceThesesRequest, request: Request) -> ReplaceThesesResponse:
    state = request.app.state
    if state.storage is None:
        raise HTTPException(status_code=503, detail="Хранилище не инициализировано")
    cards = tuple(body.cards)
    with state.import_lock:
        import_id = next(_import_ids)
        mode = "bm25" if state.encoder is None else "hybrid"
        started = perf_counter()
        status, failure = "error", "internal_error"
        logger.info(
            "theses_import import_id=%s mode=%s documents=%s status=started",
            import_id, mode, len(cards),
        )
        try:
            if state.encoder is None:
                service = BM25SearchService.from_cards(cards)
                state.storage.replace_cards(cards)
            else:
                with state.encode_lock:
                    snapshot, _ = prepare_index(cards, encoder=state.encoder)
                service = SearchService.from_snapshot(
                    snapshot, encoder=state.encoder, encode_lock=state.encode_lock,
                )
                state.storage.replace_snapshot(snapshot)
            state.search_service = service
            status, failure = "saved", "none"
        except EncoderError as error:
            failure = "encoder_error"
            raise HTTPException(
                status_code=503, detail="Энкодер временно недоступен; корпус не заменён",
            ) from error
        except (OSError, sqlite3.OperationalError, sqlite3.IntegrityError) as error:
            failure = "storage_error"
            raise HTTPException(
                status_code=503, detail="Хранилище временно недоступно; корпус не заменён",
            ) from error
        finally:
            logger.log(
                logging.INFO if status == "saved" else logging.WARNING,
                "theses_import import_id=%s mode=%s documents=%s status=%s duration_ms=%.1f error=%s",
                import_id, mode, len(cards), status, (perf_counter() - started) * 1000, failure,
            )
    return ReplaceThesesResponse(documents_count=len(cards), search_mode=mode)


@router.get(
    "/health", response_model=HealthResponse,
    responses={503: {"model": NotReadyHealthResponse, "description": "Корпус не загружен"}},
)
def health(request: Request) -> HealthResponse | JSONResponse:
    service = request.app.state.search_service
    if service is None:
        encoder = request.app.state.encoder
        body = NotReadyHealthResponse(
            search_mode="hybrid" if encoder is not None else "bm25",
            embedding_model=encoder.model.identifier if encoder is not None else None,
        )
        return JSONResponse(status_code=503, content=body.model_dump())
    return HealthResponse.model_validate(service.health())

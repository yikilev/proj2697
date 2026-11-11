"""Синхронные HTTP-обработчики выполняются FastAPI в пуле потоков."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from api.schemas import HealthResponse, SearchRequest, SearchResponse
from domain.models import Thesis
from external.embedder import EncoderError
from services.search import SearchService


router = APIRouter()


def get_search_service(request: Request) -> SearchService:
    service = request.app.state.search_service
    if service is None:
        raise HTTPException(status_code=503, detail="Поисковый сервис не инициализирован")
    return service


SearchDependency = Annotated[SearchService, Depends(get_search_service)]


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


@router.get("/health", response_model=HealthResponse)
def health(service: SearchDependency) -> dict[str, str | bool | int]:
    return service.health()

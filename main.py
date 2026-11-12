"""Сборка HTTP-приложения; интеграции создаются только внутри lifespan."""

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import sqlite3
from threading import Lock

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from api.routes import router
from config.settings import load_settings
from external.embedder import Encoder
from infrastructure.storage import IndexStorage, SQLiteStorage
from services.search import BM25SearchService, SearchService


@dataclass(frozen=True)
class AppDependencies:
    encoder: Encoder
    storage: IndexStorage


def _initialize_service(
    app: FastAPI,
    dependency_factory: Callable[[], AppDependencies] | None,
) -> SearchService | BM25SearchService | None:
    settings = load_settings()
    logging.basicConfig(level=settings.log_level)
    logging.getLogger().setLevel(settings.log_level)
    if dependency_factory is not None:
        dependencies = dependency_factory()
        app.state.search_mode = "hybrid"
        return SearchService(storage=dependencies.storage, encoder=dependencies.encoder)
    app.state.search_mode = settings.search_mode
    if settings.search_mode == "hybrid":
        raise RuntimeError(
            "SEARCH_MODE=hybrid: адаптер эмбеддера не подключён. "
            "Для hybrid передайте реальные зависимости через "
            "create_app(dependency_factory=...). Автоматического перехода на bm25 нет."
        )
    try:
        storage = SQLiteStorage(settings.database_path)
        cards = storage.load_cards(validate_snapshot=False)
        service = None if cards is None else BM25SearchService.from_cards(cards)
        app.state.storage = storage
        return service
    except (OSError, sqlite3.Error) as error:
        raise RuntimeError(
            f"DATABASE_PATH: не удалось прочитать корпус SQLite из {settings.database_path}"
        ) from error


def create_app(
    *, dependency_factory: Callable[[], AppDependencies] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = await run_in_threadpool(_initialize_service, app, dependency_factory)
        app.state.search_service = service
        try:
            yield
        finally:
            app.state.search_service = None
            app.state.storage = None

    app = FastAPI(
        title="Поиск выпускных квалификационных работ",
        description="Поиск ВКР: BM25 по SQLite или hybrid с явно подключённым эмбеддером.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.search_service = None
    app.state.storage = None
    app.state.search_mode = None
    app.state.import_lock = Lock()
    app.include_router(router)
    return app


app = create_app()

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
from external.embedder import Encoder, OllamaEncoder
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
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().setLevel(settings.log_level)
    if dependency_factory is not None:
        dependencies = dependency_factory()
        service = SearchService(
            storage=dependencies.storage, encoder=dependencies.encoder, encode_lock=app.state.encode_lock,
        )
        app.state.storage = dependencies.storage
        app.state.encoder = dependencies.encoder
        return service
    if settings.ollama_enabled:
        encoder = OllamaEncoder(settings.ollama_base_url, settings.ollama_timeout_seconds)
        storage = SQLiteStorage(settings.database_path)
        try:
            snapshot = storage.load_snapshot()
            service = None if snapshot is None else SearchService.from_snapshot(
                snapshot, encoder=encoder, encode_lock=app.state.encode_lock,
            )
        except (ValueError, OSError, sqlite3.Error) as error:
            raise RuntimeError(
                f"Гибридный индекс несовместим или повреждён: {error}. "
                "Остановите API. Проверьте DATABASE_PATH в .env или окружении. "
                "Для переиндексации нужен исходный UTF-8 JSON-массив карточек; "
                "объект HTTP-импорта сначала преобразуйте в массив cards. "
                "Из корня проекта запустите python -m scripts.build_index --input "
                '"путь к подготовленному массиву.json". Это полностью заменит корпус. '
                "Для повреждённой БД выберите новый DATABASE_PATH, сохранив старый файл. "
                "Подготовка входного файла и восстановление описаны в README.md."
            ) from error
        app.state.storage = storage
        app.state.encoder = encoder
        return service
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
            app.state.encoder = None

    app = FastAPI(
        title="Поиск выпускных квалификационных работ",
        description="Поиск ВКР: BM25 по SQLite или hybrid с Ollama BGE-M3.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.search_service = None
    app.state.storage = None
    app.state.encoder = None
    app.state.import_lock = Lock()
    app.state.encode_lock = Lock()
    app.include_router(router)
    return app


app = create_app()

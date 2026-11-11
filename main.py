"""Сборка HTTP-приложения; интеграции создаются только внутри lifespan."""

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from api.routes import router
from external.embedder import Encoder
from infrastructure.storage import IndexStorage
from services.search import SearchService


@dataclass(frozen=True)
class AppDependencies:
    encoder: Encoder
    storage: IndexStorage


def build_dependencies() -> AppDependencies:
    raise RuntimeError(
        "Адаптеры энкодера и хранилища не подключены. "
        "Передайте сборку зависимостей в create_app(dependency_factory=...)."
    )


def create_app(
    *, dependency_factory: Callable[[], AppDependencies] = build_dependencies,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        dependencies = await run_in_threadpool(dependency_factory)
        service = await run_in_threadpool(
            SearchService, storage=dependencies.storage, encoder=dependencies.encoder,
        )
        app.state.search_service = service
        try:
            yield
        finally:
            app.state.search_service = None

    app = FastAPI(
        title="Поиск выпускных квалификационных работ",
        description="Гибридный поиск ВКР. Для запуска требуется сборка реальных адаптеров.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.search_service = None
    app.include_router(router)
    return app


app = create_app()

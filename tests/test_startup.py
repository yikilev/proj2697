import logging
import sqlite3
from contextlib import closing
import subprocess
import sys
from threading import get_ident
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main
from config.settings import PROJECT_ROOT, load_settings
from infrastructure.storage import SQLiteStorage
from tests.test_api import FakeAssembly
from tests.test_bm25_service import cards as cards, storage as storage
from tests.test_search_service import snapshot as snapshot


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for name in ("SEARCH_MODE", "DATABASE_PATH", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    level = logging.getLogger().level
    yield
    logging.getLogger().setLevel(level)


def test_defaults_ignore_cwd_and_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SEARCH_MODE=hybrid\nDATABASE_PATH=other.sqlite3\n")
    settings = load_settings()
    assert settings.search_mode == "bm25"
    assert settings.database_path == PROJECT_ROOT / "data/theses.sqlite3"
    assert settings.log_level == "INFO"


def test_relative_database_path_is_project_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_PATH", "data/../custom.sqlite3")
    assert load_settings().database_path == PROJECT_ROOT / "custom.sqlite3"


@pytest.mark.parametrize("name,value", [
    ("SEARCH_MODE", "typo"), ("SEARCH_MODE", ""),
    ("LOG_LEVEL", "verbose"), ("LOG_LEVEL", ""),
    ("DATABASE_PATH", ""), ("DATABASE_PATH", "  "),
    ("DATABASE_PATH", "."), ("DATABASE_PATH", "main.py/child.sqlite3"),
])
def test_invalid_settings_abort_before_storage(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    storage = Mock(side_effect=AssertionError("Storage must not be created"))
    monkeypatch.setattr(main, "SQLiteStorage", storage)
    app = main.create_app()
    with pytest.raises(ValueError, match=name):
        with TestClient(app):
            pytest.fail("Startup must fail")
    assert app.state.search_service is None
    storage.assert_not_called()


@pytest.mark.parametrize("mode", [None, "bm25"])
def test_bm25_lifespan_loads_sqlite_once_outside_event_loop(storage, monkeypatch, mode):
    app = main.create_app()
    # Настройки должны читаться при startup, а не при создании приложения.
    monkeypatch.setenv("DATABASE_PATH", str(storage.path))
    monkeypatch.setenv("LOG_LEVEL", "warning")
    if mode is not None:
        monkeypatch.setenv("SEARCH_MODE", mode)
    threads = []
    original_load = SQLiteStorage.load_cards

    def load_cards(self, **kwargs):
        threads.append(get_ident())
        return original_load(self, **kwargs)

    monkeypatch.setattr(SQLiteStorage, "load_cards", load_cards)
    hybrid = Mock(side_effect=AssertionError("Hybrid must not be initialized"))
    monkeypatch.setattr(main, "SearchService", hybrid)
    with TestClient(app) as client:
        assert logging.getLogger().level == logging.WARNING
        assert client.get("/health").json() == {
            "status": "ok", "index_loaded": True, "documents_count": 3,
            "search_mode": "bm25", "embedding_model": None,
        }
        response = client.post("/theses/search", json={"query": "Поиск", "debug": True})
        assert response.status_code == 200
        assert [hit["id"] for hit in response.json()["results"]] == ["10", "2"]
        assert all(hit["semantic_rank"] is None for hit in response.json()["results"])
        assert client.get("/theses/2").status_code == 200
        assert len(threads) == 1
        assert threads[0] != client.portal.call(get_ident)
    assert app.state.search_service is None
    assert TestClient(app).get("/health").status_code == 503
    hybrid.assert_not_called()


def test_missing_corpus_starts_without_creating_files(tmp_path, monkeypatch):
    path = tmp_path / "missing" / "theses.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    app = main.create_app()
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
        assert client.get("/health").status_code == 503
        assert client.get("/health").json() == {
            "status": "not_ready", "index_loaded": False, "documents_count": 0,
            "search_mode": "bm25", "embedding_model": None,
        }
        assert client.get("/theses/unknown").status_code == 503
        response = client.post("/theses/search", json={"query": "Поиск"})
        assert response.status_code == 503
        assert "PUT /theses" in response.json()["detail"]
    assert not path.parent.exists()
    assert app.state.search_service is None
    assert TestClient(app).get("/health").status_code == 503


def test_invalid_database_has_clear_error(tmp_path, monkeypatch):
    path = tmp_path / "broken.sqlite3"
    path.write_text("not a SQLite database")
    monkeypatch.setenv("DATABASE_PATH", str(path))
    with pytest.raises(RuntimeError, match="DATABASE_PATH: не удалось прочитать корпус SQLite"):
        with TestClient(main.create_app()):
            pytest.fail("Startup must fail")


@pytest.mark.parametrize("damage", ["empty_file", "missing_table", "invalid_column", "empty_corpus"])
def test_invalid_existing_corpus_is_not_treated_as_missing(storage, monkeypatch, damage):
    monkeypatch.setenv("DATABASE_PATH", str(storage.path))
    if damage == "empty_file":
        storage.path.write_bytes(b"")
    else:
        with closing(sqlite3.connect(storage.path)) as connection, connection:
            connection.execute({
                "missing_table": "DROP TABLE vectors",
                "invalid_column": "ALTER TABLE cards RENAME COLUMN title TO wrong_title",
                "empty_corpus": "DELETE FROM cards",
            }[damage])
    before = storage.path.read_bytes()
    with pytest.raises(ValueError):
        with TestClient(main.create_app()):
            pytest.fail("Startup must fail")
    assert storage.path.read_bytes() == before


def test_storage_access_error_aborts_startup(storage, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(storage.path))

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(SQLiteStorage, "load_cards", denied)
    with pytest.raises(RuntimeError, match="DATABASE_PATH"):
        with TestClient(main.create_app()):
            pytest.fail("Startup must fail")


def test_hybrid_cannot_fall_back_to_existing_bm25_corpus(storage, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(storage.path))
    monkeypatch.setenv("SEARCH_MODE", "hybrid")
    bm25 = Mock(side_effect=AssertionError("BM25 fallback is forbidden"))
    monkeypatch.setattr(main, "BM25SearchService", bm25)
    app = main.create_app()
    with pytest.raises(RuntimeError, match="SEARCH_MODE=hybrid: адаптер эмбеддера не подключён"):
        with TestClient(app):
            pytest.fail("Startup must fail")
    bm25.assert_not_called()
    assert app.state.search_service is None
    client = TestClient(app)
    assert client.get("/health").status_code == 503
    assert client.post("/theses/search", json={"query": "Поиск"}).status_code == 503
    result = subprocess.run(
        [sys.executable, "-m", "uvicorn", "main:app", "--app-dir", str(PROJECT_ROOT),
         "--workers", "1"],
        cwd=storage.path.parent, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "SEARCH_MODE=hybrid: адаптер эмбеддера не подключён" in result.stderr
    assert "Application startup failed" in result.stderr
    assert "Application startup complete" not in result.stderr


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_explicit_hybrid_factory_is_preserved(snapshot, monkeypatch, mode):
    monkeypatch.setenv("SEARCH_MODE", mode)
    assembly = FakeAssembly(snapshot)
    threads = []

    def factory():
        threads.append(get_ident())
        return assembly()

    with TestClient(main.create_app(dependency_factory=factory)) as client:
        assert client.get("/health").json()["search_mode"] == "hybrid"
        assert client.post("/theses/search", json={"query": "Поиск"}).status_code == 200
        assert assembly.calls == assembly.storage.reads == 1
        assert assembly.encoder.queries == ["Поиск"]
        assert threads[0] != client.portal.call(get_ident)

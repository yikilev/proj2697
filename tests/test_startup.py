import logging
import sqlite3
from contextlib import closing
from threading import get_ident
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main
import config.settings as settings_module
from config.settings import load_settings
from infrastructure.storage import SQLiteStorage
from tests.test_api import FakeAssembly
from tests.test_bm25_service import cards as cards, storage as storage
from tests.test_search_service import snapshot as snapshot


def test_defaults_ignore_cwd_and_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SEARCH_MODE=hybrid\nDATABASE_PATH=other.sqlite3\n", encoding="utf-8")
    settings = load_settings()
    assert settings.database_path == settings_module.PROJECT_ROOT / "data/theses.sqlite3"
    assert settings.log_level == "INFO"


def test_relative_database_path_is_project_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_PATH", "data/../custom.sqlite3")
    assert load_settings().database_path == settings_module.PROJECT_ROOT / "custom.sqlite3"


def test_project_dotenv_and_environment_priority(tmp_path, monkeypatch):
    project = tmp_path / "настроенный проект"
    project.mkdir()
    (project / ".env").write_text(
        'DATABASE_PATH="data/мои работы.sqlite3"\nLOG_LEVEL=warning\nSEARCH_MODE=hybrid\n'
        'OLLAMA_ENABLED=true\nOLLAMA_BASE_URL=http://localhost:11435\nOLLAMA_TIMEOUT_SECONDS=12.5\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_module, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    assert load_settings().database_path == project / "data/мои работы.sqlite3"
    assert load_settings().log_level == "WARNING"
    assert load_settings().ollama_enabled is True
    assert load_settings().ollama_base_url == "http://localhost:11435"
    assert load_settings().ollama_timeout_seconds == 12.5
    monkeypatch.setenv("DATABASE_PATH", "override.sqlite3")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "30")
    assert load_settings().database_path == project / "override.sqlite3"
    assert load_settings().log_level == "ERROR"
    assert load_settings().ollama_enabled is False
    assert load_settings().ollama_base_url == "http://127.0.0.1:11434"
    assert load_settings().ollama_timeout_seconds == 30


def test_dotenv_does_not_interpolate_environment_variables(isolated_settings, monkeypatch):
    monkeypatch.setenv("CORPUS_DIRECTORY", "another-directory")
    (isolated_settings / ".env").write_text(
        "DATABASE_PATH='${CORPUS_DIRECTORY}/index.sqlite3'\n", encoding="utf-8",
    )
    assert load_settings().database_path == isolated_settings / "${CORPUS_DIRECTORY}/index.sqlite3"


@pytest.mark.parametrize("name,value", [
    ("LOG_LEVEL", "verbose"), ("LOG_LEVEL", ""),
    ("DATABASE_PATH", ""), ("DATABASE_PATH", "  "),
    ("DATABASE_PATH", "."), ("DATABASE_PATH", "main.py/child.sqlite3"),
])
def test_invalid_settings_abort_before_storage(name, value, monkeypatch):
    (settings_module.PROJECT_ROOT / "main.py").write_text("", encoding="utf-8")
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


@pytest.mark.parametrize("seed_demo", [False, True])
def test_invalid_database_has_clear_error(tmp_path, monkeypatch, seed_demo):
    path = tmp_path / "broken.sqlite3"
    path.write_text("not a SQLite database", encoding="utf-8")
    monkeypatch.setenv("DATABASE_PATH", str(path))
    with pytest.raises(RuntimeError, match="DATABASE_PATH: не удалось прочитать корпус SQLite"):
        with TestClient(main.create_app(seed_demo=seed_demo)):
            pytest.fail("Startup must fail")
    assert path.read_text(encoding="utf-8") == "not a SQLite database"


@pytest.mark.parametrize("damage", ["empty_file", "missing_table", "invalid_column", "empty_corpus"])
@pytest.mark.parametrize("seed_demo", [False, True])
def test_invalid_existing_corpus_is_not_treated_as_missing(storage, monkeypatch, damage, seed_demo):
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
        with TestClient(main.create_app(seed_demo=seed_demo)):
            pytest.fail("Startup must fail")
    assert storage.path.read_bytes() == before


def test_demo_startup_swagger_examples_and_persistence(tmp_path, monkeypatch):
    path = tmp_path / "volume" / "theses.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(path))
    encoder = Mock(side_effect=AssertionError("Demo must not connect to Ollama"))
    monkeypatch.setattr(main, "OllamaEncoder", encoder)

    with TestClient(main.create_demo_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok", "index_loaded": True, "documents_count": 6,
            "search_mode": "bm25", "embedding_model": None,
        }
        assert client.get("/docs").status_code == 200
        spec = client.get("/openapi.json").json()
        examples = spec["paths"]["/theses/search"]["post"]["requestBody"]["content"]["application/json"]["examples"]
        expected = {
            "satellite": ["hse-1162011207"],
            "museums": ["hse-1154627811"],
            "no_matches": [],
        }
        for name, ids in expected.items():
            response = client.post("/theses/search", json=examples[name]["value"])
            assert response.status_code == 200
            assert [hit["id"] for hit in response.json()["results"]] == ids
        demo_import = spec["components"]["schemas"]["ReplaceThesesRequest"]["examples"][0]
        assert demo_import["cards"] == [client.get("/theses/hse-1162011207").json()]
        parameters = spec["paths"]["/theses/{thesis_id}"]["get"]["parameters"]
        example_id = parameters[0]["schema"]["examples"][0]
        assert client.get(f"/theses/{example_id}").status_code == 200

    before = path.read_bytes()
    with TestClient(main.create_demo_app()) as client:
        assert client.get("/health").json()["documents_count"] == 6
        assert path.read_bytes() == before
        # Пользовательский корпус должен пережить следующий запуск без досева демоданных.
        card = {**demo_import["cards"][0], "id": "my-thesis", "title": "Моя работа"}
        response = client.put("/theses", json={"cards": [card]})
        assert response.status_code == 200
    before = path.read_bytes()
    with TestClient(main.create_demo_app()) as client:
        assert client.get("/health").json()["documents_count"] == 1
        assert client.get("/theses/my-thesis").json() == card
        assert client.get("/theses/hse-1162011207").status_code == 404
        assert path.read_bytes() == before
    encoder.assert_not_called()


def test_storage_access_error_aborts_startup(storage, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(storage.path))

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(SQLiteStorage, "load_cards", denied)
    with pytest.raises(RuntimeError, match="DATABASE_PATH"):
        with TestClient(main.create_app()):
            pytest.fail("Startup must fail")


@pytest.mark.parametrize("mode", ["hybrid", "invalid"])
def test_search_mode_environment_does_not_select_request_mode(storage, monkeypatch, mode):
    monkeypatch.setenv("DATABASE_PATH", str(storage.path))
    monkeypatch.setenv("SEARCH_MODE", mode)
    with TestClient(main.create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/theses/search", json={"query": "Поиск"}).status_code == 200
        response = client.post("/theses/search", json={"query": "Поиск", "search_mode": "hybrid"})
        assert response.status_code == 503
        assert "эмбеддер не подключён" in response.json()["detail"]
        assert client.post("/theses/search", json={"query": "Поиск", "search_mode": "bm25"}).status_code == 200


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
        assert client.post("/theses/search", json={"query": "Поиск", "search_mode": "hybrid"}).status_code == 200
        assert assembly.calls == assembly.storage.reads == 1
        assert assembly.encoder.queries == ["Поиск"]
        assert threads[0] != client.portal.call(get_ident)

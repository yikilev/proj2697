"""Перенос исходников и работа CLI из другого каталога без личных файлов."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_relocated_project_import_search_restart_and_indexing(tmp_path):
    project = tmp_path / "копия проекта"
    project.mkdir()
    shutil.copyfile(ROOT / "main.py", project / "main.py")
    for package in ("api", "config", "domain", "external", "infrastructure", "scripts", "services"):
        (project / package).mkdir()
        for source in (ROOT / package).glob("*.py"):
            shutil.copyfile(source, project / package / source.name)
    (project / "data").mkdir()
    shutil.copyfile(ROOT / "data/sample_theses.json", project / "data/sample_theses.json")
    (project / ".env").write_text(
        "DATABASE_PATH='данные корпуса/index %#.sqlite3'\nOLLAMA_ENABLED=false\n",
        encoding="utf-8",
    )
    caller = tmp_path / "другой каталог"
    caller.mkdir()
    # CWD не должен подменять конфигурацию перенесённого проекта.
    (caller / ".env").write_text("OLLAMA_ENABLED=true\nLOG_LEVEL=invalid\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(project)}
    script = r'''
import json
from pathlib import Path
import socket
import sys
from urllib.request import OpenerDirector

def reject_network(*args, **kwargs):
    raise AssertionError("Network must not be used")
OpenerDirector.open = reject_network
socket.create_connection = reject_network

import numpy as np
from fastapi.testclient import TestClient
from config.settings import PROJECT_ROOT, load_settings
from external.embedder import EncodedDocuments, ModelDescription
from infrastructure.storage import SQLiteStorage
from main import create_app
from scripts.build_index import main as build_cli

project = Path(sys.argv[1])
assert PROJECT_ROOT == project
path = project / "данные корпуса/index %#.sqlite3"
assert load_settings().database_path == path
cards = json.loads((project / "data/sample_theses.json").read_text(encoding="utf-8"))
query = {"query": "Rasdaman", "debug": True}
with TestClient(create_app()) as client:
    assert client.get("/health").status_code == 503
    assert not path.exists()
    assert client.put("/theses", json={"cards": cards}).json() == {
        "status": "ok", "documents_count": 6, "search_mode": "bm25",
    }
    response = client.post("/theses/search", json=query)
    assert response.status_code == 200
    result = response.json()
    assert result["count"] == 1
    assert result["results"][0]["id"] == "hse-1162011207"
    assert result["results"][0]["score"] == 1 / 61
before = path.read_bytes()
with TestClient(create_app()) as client:
    assert client.get("/health").json()["documents_count"] == 6
    assert client.post("/theses/search", json=query).json() == result
assert path.read_bytes() == before

class FakeEncoder:
    model = ModelDescription(identifier="test-only", version="1", dimension=2, normalization="l2")
    def encode_documents(self, texts):
        return EncodedDocuments(np.tile([1.0, 0.0], (len(texts), 1)), 0)

# Реальный CLI, конфигурация и SQLite; подменён только энкодер, не проверяем Ollama.
assert build_cli(["--input", "data/sample_theses.json"], encoder=FakeEncoder()) == 0
assert len(SQLiteStorage(path).load_snapshot().cards) == 6
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(project)], cwd=caller, env=env,
        capture_output=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for module in ("scripts.build_index", "scripts.prepare_miem_import"):
        result = subprocess.run(
            [sys.executable, "-B", "-m", module, "--help"], cwd=caller, env=env,
            capture_output=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "--input" in result.stdout


def test_converter_paths_are_cwd_relative_and_preserve_utf8(tmp_path):
    source = tmp_path / "исходные работы.jsonl"
    record = {
        "title": "Поиск работ", "annotation": "Описание.\nКлючевые слова: поиск.",
        "source_url": "https://www.hse.ru/edu/vkr/123",
        "author_raw": "Автор", "defense_year": 2026,
    }
    original = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    source.write_bytes(original)
    output = tmp_path / "импорт работ.json"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "scripts.prepare_miem_import",
         "--input", source.name, "--output", output.name],
        cwd=tmp_path, env=env, capture_output=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stderr
    card = json.loads(output.read_text(encoding="utf-8"))["cards"][0]
    assert card["title"] == record["title"]
    assert card["keywords"] == ["поиск"]
    assert source.read_bytes() == original

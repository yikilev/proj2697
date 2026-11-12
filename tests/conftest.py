import logging
import socket
from urllib.request import OpenerDirector

import pytest

import config.settings as settings_module


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    # Даже тесты значений по умолчанию читают только собственный корень.
    # Очистка окружения без подмены корня оставила бы доступ к личному .env и БД.
    project = tmp_path / 'проект с пробелами'
    project.mkdir()
    monkeypatch.setattr(settings_module, 'PROJECT_ROOT', project)
    for name in (
        'DATABASE_PATH', 'LOG_LEVEL', 'OLLAMA_ENABLED', 'OLLAMA_BASE_URL',
        'OLLAMA_TIMEOUT_SECONDS', 'SEARCH_MODE', 'PYTHON_DOTENV_DISABLED',
    ):
        monkeypatch.delenv(name, raising=False)
    # Дочерние CLI печатают русский help и пути; кодировка pipe не зависит от ОС.
    monkeypatch.setenv('PYTHONUTF8', '1')
    monkeypatch.setenv('PYTHONIOENCODING', 'utf-8')
    level = logging.getLogger().level
    yield project
    logging.getLogger().setLevel(level)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject_http(*args, **kwargs):
        raise AssertionError('Подмените HTTP Ollama в тесте')

    monkeypatch.setattr(OpenerDirector, 'open', reject_http)
    # socket.connect не подменяем: Windows может использовать его внутри
    # socketpair для служебного канала event loop TestClient.
    monkeypatch.setattr(socket, 'create_connection', reject_http)

import io
import json
from contextlib import closing
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import external.embedder as embedder
import main
from config import settings as settings_module
from external.embedder import EncoderError, OllamaEncoder
from infrastructure.storage import SQLiteStorage
from scripts.build_index import main as build_cli
from services.indexing import build_index
from tests.test_indexing import corpus as corpus, records as records

DIGEST = '7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab'


def vector(position=0):
    row = [0.0] * 1024
    row[position] = 1.0
    return row


def tags(digest=DIGEST):
    return {'models': [{'name': 'bge-m3:latest', 'digest': digest}]}


def embeddings(*rows):
    return {'model': 'bge-m3:latest', 'embeddings': list(rows)}


@pytest.fixture
def http(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_module, 'PROJECT_ROOT', tmp_path)
    for key in ('OLLAMA_ENABLED', 'OLLAMA_BASE_URL', 'OLLAMA_TIMEOUT_SECONDS', 'DATABASE_PATH'):
        monkeypatch.delenv(key, raising=False)
    responses = [tags()]
    calls = []
    streams = []

    def open_request(request, *, timeout):
        calls.append((request, timeout))
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        stream = io.BytesIO(item if isinstance(item, bytes) else json.dumps(item).encode())
        stream.status = 200
        streams.append(stream)
        return stream

    opener = Mock(open=open_request)
    factory = Mock(return_value=opener)
    monkeypatch.setattr(embedder, 'build_opener', factory)
    return responses, calls, streams, factory


def test_query_batch_order_dtype_payload_and_no_proxy(http):
    responses, calls, streams, factory = http
    responses.extend([embeddings(vector(2)), embeddings(vector(3), vector(1))])
    encoder = OllamaEncoder(timeout=13)
    assert encoder.model.version == DIGEST
    assert encoder.model.dimension == 1024
    assert encoder.model.normalization == 'l2'
    query = encoder.encode_query('естественный запрос')
    batch = encoder.encode_documents(['первый документ', 'второй документ'])
    assert query.shape == (1024,) and query.dtype == np.float32
    assert batch.vectors.shape == (2, 1024) and batch.vectors.dtype == np.float32
    assert query.argmax() == 2
    assert batch.vectors.argmax(axis=1).tolist() == [3, 1]
    assert batch.truncated_count == 0
    for (request, timeout), expected in zip(calls[1:], ['естественный запрос', ['первый документ', 'второй документ']]):
        assert request.full_url == 'http://127.0.0.1:11434/api/embed'
        assert request.get_method() == 'POST'
        assert json.loads(request.data) == {
            'model': 'bge-m3:latest', 'input': expected, 'truncate': False, 'keep_alive': '30m',
        }
        assert timeout == 13
    assert factory.call_args.args[0].proxies == {}
    assert all(stream.closed for stream in streams)
    assert encoder.encode_documents([]).vectors.shape == (0, 1024)
    assert len(calls) == 3


@pytest.mark.parametrize('payload', [
    embeddings(), embeddings(vector(), vector()), embeddings([1.0] * 1023),
    embeddings([0.0] * 1024), embeddings([2.0] + [0.0] * 1023),
    embeddings([float('nan')] + [0.0] * 1023),
    embeddings([float('inf')] + [0.0] * 1023),
    embeddings([float('-inf')] + [0.0] * 1023),
    embeddings(['1'] + ['0'] * 1023), embeddings([True] + [False] * 1023),
    embeddings(vector(), [1]), {'embeddings': [vector()]},
    {'model': 'wrong', 'embeddings': [vector()]}, {}, [], None,
    {'error': 'context length exceeded'}, b'not JSON', b'\xff',
])
def test_bad_response_is_encoder_error(http, payload):
    http[0].append(payload)
    encoder = OllamaEncoder()
    with pytest.raises(EncoderError):
        encoder.encode_query('query')
    assert all(stream.closed for stream in http[2])


@pytest.mark.parametrize('error', [
    TimeoutError('timed out'), URLError(ConnectionRefusedError()), ConnectionResetError(),
    IncompleteRead(b'partial'), HTTPError('http://localhost', 400, 'context too long', {}, io.BytesIO()),
])
@pytest.mark.parametrize('stage', ['tags', 'embed'])
def test_expected_http_failures(http, error, stage):
    if stage == 'tags':
        http[0][:] = [error]
        with pytest.raises(EncoderError):
            OllamaEncoder()
    else:
        http[0].append(error)
        encoder = OllamaEncoder()
        with pytest.raises(EncoderError):
            encoder.encode_documents(['document'])


@pytest.mark.parametrize('payload', [
    {}, {'models': []}, {'models': None}, {'models': [None]}, tags('latest'), tags(None),
    {'models': [{'name': 'other', 'digest': DIGEST}]},
])
def test_invalid_model_description(http, payload):
    http[0][:] = [payload]
    with pytest.raises(EncoderError):
        OllamaEncoder()


def test_programming_error_not_masked(http):
    http[0].append(RuntimeError('bug'))
    encoder = OllamaEncoder()
    with pytest.raises(RuntimeError, match='bug'):
        encoder.encode_query('query')


def test_cli_real_adapters_lifespan_restart_and_failure(http, corpus, tmp_path, monkeypatch):
    path = tmp_path / 'hybrid.sqlite3'
    monkeypatch.setenv('DATABASE_PATH', str(path))
    http[0].extend([embeddings(vector(0), vector(1)), embeddings(vector(2))])
    assert build_cli(['--input', str(corpus), '--batch-size', '2']) == 0
    before = path.read_bytes()
    monkeypatch.setenv('OLLAMA_ENABLED', 'true')
    for _ in range(2):
        http[0].extend([tags(), embeddings(vector(1)), TimeoutError()])
        with TestClient(main.create_app()) as client:
            count = len(http[1])
            assert client.get('/health').json()['search_mode'] == 'hybrid'
            assert client.post('/theses/search', json={'query': 'Поиск', 'search_mode': 'bm25'}).status_code == 200
            assert len(http[1]) == count
            result = client.post('/theses/search', json={'query': 'Поиск', 'search_mode': 'hybrid', 'debug': True})
            assert result.status_code == 200
            assert any(hit['semantic_rank'] is not None for hit in result.json()['results'])
            assert client.post('/theses/search', json={'query': 'Поиск', 'search_mode': 'hybrid'}).status_code == 503
            assert client.post('/theses/search', json={'query': 'Поиск'}).status_code == 200
    assert path.read_bytes() == before
    # После индексации единственный вход каждого embed — строка запроса.
    embeds = [json.loads(req.data)['input'] for req, _ in http[1] if req.data]
    assert all(isinstance(item, str) for item in embeds[2:])
    assert http[0] == []


def test_disabled_integration_never_calls_ollama(http, tmp_path, monkeypatch):
    monkeypatch.setenv('DATABASE_PATH', str(tmp_path / 'missing.sqlite3'))
    with TestClient(main.create_app()) as client:
        assert client.get('/health').status_code == 503
    assert http[1] == []


@pytest.mark.parametrize('kind', ['digest', 'dimension', 'normalization'])
def test_startup_rejects_unusable_index_with_command(http, corpus, tmp_path, monkeypatch, kind):
    path = tmp_path / 'index.sqlite3'
    storage = SQLiteStorage(path)
    if kind != 'missing':
        http[0].append(embeddings(vector(), vector(), vector()))
        encoder = OllamaEncoder()
        build_index(corpus, encoder=encoder, storage=storage)
        snapshot = storage.load_snapshot()
        if kind == 'cards_only':
            storage.replace_cards(snapshot.cards)
        else:
            import sqlite3
            metadata = snapshot.metadata.model_dump()
            metadata['model'][{'digest': 'version', 'dimension': 'dimension', 'normalization': 'normalization'}[kind]] = {
                'digest': 'a' * 64, 'dimension': 768, 'normalization': 'none',
            }[kind]
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute('UPDATE index_state SET metadata = ?', (json.dumps(metadata),))
        http[0].append(tags())
    monkeypatch.setenv('DATABASE_PATH', str(path))
    monkeypatch.setenv('OLLAMA_ENABLED', 'true')
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(RuntimeError, match='python -m scripts.build_index --input'):
        with TestClient(main.create_app()):
            pytest.fail('startup must fail')
    assert (path.read_bytes() if path.exists() else None) == before


def test_failed_later_http_batch_preserves_sqlite(http, corpus, tmp_path):
    storage = SQLiteStorage(tmp_path / 'index.sqlite3')
    http[0].extend([embeddings(vector(), vector(), vector()), embeddings(vector(), vector()), TimeoutError()])
    encoder = OllamaEncoder()
    build_index(corpus, encoder=encoder, storage=storage)
    before = storage.path.read_bytes()
    with pytest.raises(EncoderError):
        build_index(corpus, encoder=encoder, storage=storage, batch_size=2)
    assert storage.path.read_bytes() == before


@pytest.mark.parametrize('name,value', [
    ('OLLAMA_ENABLED', 'yes'), ('OLLAMA_TIMEOUT_SECONDS', '0'),
    ('OLLAMA_TIMEOUT_SECONDS', 'nan'), ('OLLAMA_TIMEOUT_SECONDS', 'inf'),
    ('OLLAMA_TIMEOUT_SECONDS', '-1'), ('OLLAMA_TIMEOUT_SECONDS', 'bad'),
])
def test_invalid_settings(http, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        settings_module.load_settings()
    assert http[1] == []


@pytest.mark.parametrize('cards_only', [False, True])
def test_http_bootstrap_with_ollama_then_restart(http, corpus, tmp_path, monkeypatch, cards_only):
    from services.indexing import read_corpus
    path = tmp_path / 'http-index.sqlite3'
    storage = SQLiteStorage(path)
    if cards_only:
        storage.replace_cards(read_corpus(corpus))
    before = path.read_bytes() if path.exists() else None
    monkeypatch.setenv('DATABASE_PATH', str(path))
    monkeypatch.setenv('OLLAMA_ENABLED', 'true')
    with TestClient(main.create_app()) as client:
        assert client.get('/health').status_code == 503
        assert client.get('/health').json() == {
            'status': 'not_ready', 'index_loaded': False, 'documents_count': 0,
            'search_mode': 'hybrid', 'embedding_model': 'bge-m3:latest',
        }
        response = client.post('/theses/search', json={'query': 'Поиск', 'search_mode': 'hybrid'})
        assert response.status_code == 503 and 'PUT /theses' in response.json()['detail']
        assert (path.read_bytes() if path.exists() else None) == before
        assert len(http[1]) == 1  # Только tags; startup не индексирует.
        http[0].append(embeddings(vector(0), vector(1), vector(2)))
        response = client.put('/theses', json={'cards': json.loads(corpus.read_text(encoding='utf-8'))})
        assert response.status_code == 200
        assert response.json() == {'status': 'ok', 'documents_count': 3, 'search_mode': 'hybrid'}
        snapshot = storage.load_snapshot()
        assert snapshot.metadata.model.version == DIGEST
        assert snapshot.vectors.shape == (3, 1024)
        expected = {'z': 0, 'a': 1, 'b': 2}
        assert {key: int(row.argmax()) for key, row in zip(snapshot.vector_ids, snapshot.vectors)} == expected
        assert client.get('/health').status_code == 200
    before = path.read_bytes()
    http[0].extend([tags(), embeddings(vector(1))])
    with TestClient(main.create_app()) as client:
        count = len(http[1])
        assert client.get('/health').status_code == 200
        assert client.post('/theses/search', json={'query': 'Поиск'}).status_code == 200
        assert len(http[1]) == count
        result = client.post('/theses/search', json={'query': 'Поиск', 'search_mode': 'hybrid', 'debug': True})
        assert result.status_code == 200
        assert all(hit['semantic_rank'] for hit in result.json()['results'])
    assert path.read_bytes() == before
    assert http[0] == []


@pytest.mark.parametrize('loaded', [False, True])
@pytest.mark.parametrize('failure', [TimeoutError(), embeddings([0.0] * 1024)])
def test_http_later_batch_error_keeps_previous_state(http, corpus, tmp_path, monkeypatch, loaded, failure):
    path = tmp_path / 'index.sqlite3'
    monkeypatch.setenv('DATABASE_PATH', str(path))
    monkeypatch.setenv('OLLAMA_ENABLED', 'true')
    with TestClient(main.create_app()) as client:
        records = json.loads(corpus.read_text(encoding='utf-8'))
        if loaded:
            http[0].append(embeddings(vector(), vector(), vector()))
            assert client.put('/theses', json={'cards': records}).status_code == 200
        before = path.read_bytes() if loaded else None
        old_service = client.app.state.search_service
        cards = [{**records[0], 'id': f'new-{i}'} for i in range(33)]
        http[0].extend([embeddings(*[vector()] * 32), failure])
        response = client.put('/theses', json={'cards': cards})
        assert response.status_code == 503
        assert response.json() == {'detail': 'Энкодер временно недоступен; корпус не заменён'}
        assert client.app.state.search_service is old_service
        assert (path.read_bytes() if path.exists() else None) == before
        assert client.get('/health').status_code == (200 if loaded else 503)
        if loaded:
            assert client.post('/theses/search', json={'query': 'Поиск'}).status_code == 200
        assert http[0] == []


def test_request_logs_start_and_validated_result_without_content(http, caplog):
    caplog.set_level('INFO', logger='external.embedder')
    http[0].extend([embeddings(vector()), embeddings(vector(), vector())])
    encoder = OllamaEncoder()
    encoder.encode_query('PRIVATE_QUERY_TEXT')
    encoder.encode_documents(['PRIVATE_DOCUMENT_ONE', 'PRIVATE_DOCUMENT_TWO'])
    messages = [record.getMessage() for record in caplog.records if record.name == 'external.embedder']
    assert len(messages) == 6
    for start, finish, operation, count in zip(messages[::2], messages[1::2], ['models', 'query', 'documents'], [0, 1, 2]):
        assert 'status=started' in start
        assert f'operation={operation}' in start and f'texts={count}' in start
        request_id = start.split('request_id=')[1].split()[0]
        assert f'request_id={request_id}' in finish
        assert 'status=ok http_status=200' in finish
        assert 'duration_ms=' in finish and 'error=none' in finish
    assert 'PRIVATE_' not in caplog.text
    assert 'embeddings' not in caplog.text


@pytest.mark.parametrize('response,http_status,category', [
    (TimeoutError('PRIVATE_TIMEOUT_DETAILS'), 'none', 'timeout'),
    (URLError(TimeoutError('PRIVATE_TIMEOUT_DETAILS')), 'none', 'timeout'),
    (URLError('PRIVATE_CONNECTION_DETAILS'), 'none', 'URLError'),
    (HTTPError('http://localhost', 500, 'PRIVATE_HTTP_DETAILS', {}, io.BytesIO()), '500', 'http_error'),
    (b'PRIVATE_BAD_JSON', '200', 'invalid_json'),
    ({'error': 'PRIVATE_OLLAMA_DETAILS'}, '200', 'ollama_error'),
    (embeddings([0.0] * 1024), '200', 'invalid_response'),
    ({'model': 'PRIVATE_WRONG_MODEL', 'embeddings': [vector()]}, '200', 'invalid_response'),
])
def test_error_logs_status_without_sensitive_details(http, caplog, response, http_status, category):
    encoder = OllamaEncoder()
    caplog.clear()
    caplog.set_level('INFO', logger='external.embedder')
    http[0].append(response)
    with pytest.raises(EncoderError):
        encoder.encode_query('PRIVATE_QUERY_TEXT')
    messages = [record for record in caplog.records if record.name == 'external.embedder']
    assert len(messages) == 2
    assert messages[-1].levelname == 'WARNING'
    assert f'status=error http_status={http_status}' in messages[-1].getMessage()
    assert f'error={category}' in messages[-1].getMessage()
    assert 'status=ok' not in caplog.text
    assert 'PRIVATE_' not in caplog.text


def test_invalid_model_metadata_is_not_logged_as_success(http, caplog):
    caplog.set_level('INFO', logger='external.embedder')
    http[0][:] = [tags('PRIVATE_BAD_DIGEST')]
    with pytest.raises(EncoderError):
        OllamaEncoder()
    assert 'operation=models' in caplog.text
    assert 'status=error http_status=200' in caplog.text
    assert 'error=invalid_response' in caplog.text
    assert 'status=ok' not in caplog.text
    assert 'PRIVATE_' not in caplog.text


def test_programming_error_is_logged_and_still_propagates(http, caplog):
    encoder = OllamaEncoder()
    http[0].append(RuntimeError('PRIVATE_BUG_DETAILS'))
    caplog.clear()
    caplog.set_level('INFO', logger='external.embedder')
    with pytest.raises(RuntimeError, match='PRIVATE_BUG_DETAILS'):
        encoder.encode_query('PRIVATE_QUERY_TEXT')
    assert 'status=error' in caplog.text and 'error=internal_error' in caplog.text
    assert 'PRIVATE_' not in caplog.text

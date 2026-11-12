"""Контракт энкодера и проверка его результата; без загрузки модели."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPException
import json
import logging
from itertools import count
import math
import re
from time import perf_counter
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

from domain.models import RequiredText


L2_RTOL = 1e-5
L2_ATOL = 1e-6
logger = logging.getLogger(__name__)
_request_ids = count(1)


class EncoderError(RuntimeError):
    """Ожидаемый сбой модели; адаптер явно отделяет его от ошибок программы."""


class ModelDescription(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    identifier: RequiredText
    version: RequiredText
    dimension: int = Field(gt=0)
    normalization: Literal["l2", "none"]


@dataclass(frozen=True)
class EncodedDocuments:
    vectors: NDArray[np.float32]
    truncated_count: int


class Encoder(Protocol):
    @property
    def model(self) -> ModelDescription:
        """Неизменное описание модели на всё время жизни энкодера."""
        ...

    def encode_query(self, text: str) -> NDArray[np.float32]:
        """Вернуть вектор формы (dimension,) для естественного текста.

        Ожидаемый сбой кодирования сообщается через EncoderError.
        """
        ...

    def encode_documents(self, texts: Sequence[str]) -> EncodedDocuments:
        """Строки (len(texts), dimension) в порядке входа.

        Адаптер отвечает за подготовку входа и число обрезанных текстов
        именно в этом батче. OllamaEncoder передаёт текст без префиксов
        с truncate=false: слишком длинный вход отклоняется, а не обрезается.
        Ожидаемый сбой кодирования сообщается через EncoderError.
        """
        ...


def validate_vectors(vectors: ArrayLike, *, shape: tuple[int, ...]) -> NDArray[np.float32]:
    """Проверить вектор запроса или матрицу документов и вернуть свою копию.

    Допуск L2 учитывает погрешность float32. Значения не нормализуются
    автоматически: неверный результат адаптера должен оставаться ошибкой.
    """
    matrix = np.asarray(vectors)
    if matrix.shape != shape:
        raise ValueError(f"Vectors must have shape {shape}, got {matrix.shape}")
    if matrix.dtype.kind not in "fiu":
        raise ValueError("Vectors must contain real numbers")
    if not np.isfinite(matrix).all():
        raise ValueError("Vectors must contain only finite values")
    # Проверяем именно сохраняемое представление: округление float32
    # не должно превращать принятый вектор в непригодный для загрузки.
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(matrix, dtype=np.float32, copy=True)
        norms = np.linalg.norm(result.astype(np.float64), axis=-1)
    if not np.allclose(norms, 1.0, rtol=L2_RTOL, atol=L2_ATOL):
        raise ValueError("Vectors must be L2-normalized")
    result.flags.writeable = False
    return result


class OllamaEncoder:
    """BGE-M3 через локальный API; каждый HTTP-ответ закрывается сразу.

    Digest читается при создании экземпляра, вне импорта модулей. Модель
    в Ollama не следует заменять во время индексации или работы приложения.
    """

    MODEL_NAME = "bge-m3:latest"
    DIMENSION = 1024

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0):
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("OLLAMA_BASE_URL: требуется локальный HTTP-адрес Ollama")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS: требуется конечное положительное число")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._opener = build_opener(ProxyHandler({}))
        with self._request("/api/tags", operation="models") as payload:
            models = payload.get("models")
            if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
                raise EncoderError("Ollama: некорректный список моделей")
            matches = [item for item in models if item.get("name") == self.MODEL_NAME]
            if len(matches) != 1:
                raise EncoderError(f"Ollama: установленная модель {self.MODEL_NAME} не найдена")
            digest = matches[0].get("digest")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise EncoderError("Ollama: некорректный digest модели")
            self._model = ModelDescription(
                identifier=self.MODEL_NAME, version=digest,
                dimension=self.DIMENSION, normalization="l2",
            )

    @property
    def model(self) -> ModelDescription:
        return self._model

    @contextmanager
    def _request(
        self, path: str, payload: dict | None = None, *, operation: str, texts_count: int = 0,
    ) -> Iterator[dict]:
        """Логируем завершение после проверки ответа вызывающим кодом, без содержимого."""
        request = Request(
            self._base_url + path,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        request_id = next(_request_ids)
        started = perf_counter()
        http_status = "none"
        status, failure = "error", "internal_error"
        logger.info(
            "ollama request_id=%s operation=%s method=%s path=%s texts=%s status=started",
            request_id, operation, request.get_method(), path, texts_count,
        )
        try:
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    http_status = response.status
                    result = json.loads(response.read())
            except HTTPError as error:
                http_status, failure = error.code, "http_error"
                # HTTPError тоже владеет ответом; не оставляем его открытым.
                error.close()
                raise EncoderError(f"Ollama: HTTP {error.code} для {path}; проверьте модель и длину входа") from error
            except (URLError, OSError, HTTPException) as error:
                cause = error.reason if isinstance(error, URLError) else error
                failure = "timeout" if isinstance(cause, TimeoutError) else type(error).__name__
                raise EncoderError(f"Ollama: запрос {path} не выполнен ({type(error).__name__})") from error
            except (ValueError, UnicodeError) as error:
                failure = "invalid_json"
                raise EncoderError("Ollama: некорректный JSON ответа") from error
            if not isinstance(result, dict):
                failure = "invalid_response"
                raise EncoderError("Ollama: ответ должен быть JSON-объектом")
            if "error" in result:
                failure = "ollama_error"
                raise EncoderError("Ollama: API сообщил ошибку кодирования; проверьте модель и длину входа")
            try:
                yield result
            except EncoderError:
                failure = "invalid_response"
                raise
            status, failure = "ok", "none"
        finally:
            # Только безопасные категории: тексты исключений могут содержать входные данные.
            logger.log(
                logging.INFO if status == "ok" else logging.WARNING,
                "ollama request_id=%s operation=%s texts=%s status=%s http_status=%s duration_ms=%.1f error=%s",
                request_id, operation, texts_count, status, http_status,
                (perf_counter() - started) * 1000, failure,
            )

    def _encode(self, texts: str | list[str], count: int) -> NDArray[np.float32]:
        with self._request("/api/embed", {
            "model": self.MODEL_NAME, "input": texts,
            "truncate": False, "keep_alive": "30m",
        }, operation="query" if isinstance(texts, str) else "documents", texts_count=count) as result:
            if result.get("model") != self.MODEL_NAME:
                raise EncoderError("Ollama: ответ получен от другой модели")
            try:
                return validate_vectors(result.get("embeddings"), shape=(count, self.DIMENSION))
            except (ValueError, TypeError, OverflowError) as error:
                raise EncoderError(f"Ollama: некорректные эмбеддинги: {error}") from error

    def encode_query(self, text: str) -> NDArray[np.float32]:
        return self._encode(text, 1)[0]

    def encode_documents(self, texts: Sequence[str]) -> EncodedDocuments:
        if not texts:
            return EncodedDocuments(np.empty((0, self.DIMENSION), dtype=np.float32), 0)
        return EncodedDocuments(self._encode(list(texts), len(texts)), 0)

"""Контракт энкодера и проверка его результата; без загрузки модели."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

from domain.models import RequiredText


L2_RTOL = 1e-5
L2_ATOL = 1e-6


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
        """Вернуть вектор формы (dimension,). Префиксы добавляет адаптер.

        Ожидаемый сбой кодирования сообщается через EncoderError.
        """
        ...

    def encode_documents(self, texts: Sequence[str]) -> EncodedDocuments:
        """Строки (len(texts), dimension) в порядке входа.

        Адаптер отвечает за префиксы, токенизацию и число обрезанных текстов
        именно в этом батче, с учётом служебных токенов модели.
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

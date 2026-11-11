"""Контракт хранения целого индекса; реализация SQLite пока отсутствует."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from domain.models import RequiredText, Thesis
from external.embedder import ModelDescription


class IndexMetadata(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    model: ModelDescription
    text_preparation_version: RequiredText


@dataclass(frozen=True)
class IndexSnapshot:
    cards: tuple[Thesis, ...]
    vector_ids: tuple[str, ...]
    vectors: NDArray[np.float32]
    metadata: IndexMetadata
    # vectors[i] принадлежит vector_ids[i], независимо от порядка cards.
    # Снимок передаётся целиком; его содержимое после передачи не изменяют.


class IndexStorage(Protocol):
    def load_snapshot(self) -> IndexSnapshot | None:
        """Прочитать карточки, векторы и метаданные одной согласованной версии.

        None означает отсутствие индекса. Ошибки чтения не маскируются.
        """
        ...

    def replace_snapshot(self, snapshot: IndexSnapshot) -> None:
        """Атомарно заменить весь индекс, сохранив прежний при ошибке.

        Будущий SQLite-адаптер обязан записать все части одной транзакцией.
        """
        ...

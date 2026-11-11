"""Подготовка корпуса и публикация целого индекса через явные зависимости."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from domain.models import Thesis
from external.embedder import Encoder, ModelDescription, validate_vectors
from infrastructure.storage import IndexMetadata, IndexSnapshot, IndexStorage


TEXT_PREPARATION_VERSION = "1"


def build_search_text(thesis: Thesis) -> str:
    return "\n".join((thesis.title, " ".join(thesis.keywords), thesis.abstract))


def read_corpus(path: str | Path) -> tuple[Thesis, ...]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    cards = tuple(TypeAdapter(list[Thesis]).validate_python(records, strict=True))
    _validate_card_ids(cards)
    return cards


def _validate_card_ids(cards: tuple[Thesis, ...]) -> None:
    if not cards:
        raise ValueError("Corpus must not be empty")
    if len({card.id for card in cards}) != len(cards):
        raise ValueError("Card IDs must be unique")


def _metadata(model: ModelDescription) -> IndexMetadata:
    if model.normalization != "l2":
        raise ValueError("Encoder must produce L2-normalized vectors")
    return IndexMetadata(model=model, text_preparation_version=TEXT_PREPARATION_VERSION)


@dataclass(frozen=True)
class IndexingResult:
    cards_count: int
    truncated_count: int


def build_index(
    input_path: str | Path,
    *,
    encoder: Encoder,
    storage: IndexStorage,
    batch_size: int = 32,
) -> IndexingResult:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    cards = read_corpus(input_path)
    metadata = _metadata(encoder.model)
    vectors = np.empty((len(cards), metadata.model.dimension), dtype=np.float32)
    truncated_count = 0
    for start in range(0, len(cards), batch_size):
        batch = cards[start:start + batch_size]
        encoded = encoder.encode_documents([build_search_text(card) for card in batch])
        vectors[start:start + len(batch)] = validate_vectors(
            encoded.vectors, shape=(len(batch), metadata.model.dimension),
        )
        if type(encoded.truncated_count) is not int or not 0 <= encoded.truncated_count <= len(batch):
            raise ValueError("truncated_count must be an integer between zero and batch size")
        truncated_count += encoded.truncated_count

    vectors.flags.writeable = False
    snapshot = IndexSnapshot(cards, tuple(card.id for card in cards), vectors, metadata)
    storage.replace_snapshot(snapshot)
    return IndexingResult(len(cards), truncated_count)


def load_index(*, storage: IndexStorage, model: ModelDescription) -> IndexSnapshot:
    """Проверить границу хранения перед использованием снимка в поиске."""
    expected = _metadata(model)
    snapshot = storage.load_snapshot()
    if snapshot is None:
        raise ValueError("Index is missing; build it before loading")
    if snapshot.metadata != expected:
        raise ValueError("Index metadata is incompatible with the model or text preparation version")

    cards = tuple(Thesis.model_validate(card.model_dump()) for card in snapshot.cards)
    _validate_card_ids(cards)
    vector_ids = tuple(snapshot.vector_ids)
    if len(set(vector_ids)) != len(vector_ids):
        raise ValueError("Vector IDs must be unique")
    if set(vector_ids) != {card.id for card in cards}:
        raise ValueError("Card IDs and vector IDs must match")
    vectors = validate_vectors(snapshot.vectors, shape=(len(vector_ids), model.dimension))
    return IndexSnapshot(cards, vector_ids, vectors, snapshot.metadata)

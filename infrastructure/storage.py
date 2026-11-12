"""Контракт полного индекса и транзакционное SQLite-хранилище."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from domain.models import RequiredText, Thesis
from external.embedder import ModelDescription, validate_vectors


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

        Все части записываются одной транзакцией.
        """
        ...


class SQLiteStorage:
    """Файловая БД; каждое обращение открывает и закрывает своё соединение.

    Векторы хранятся как IEEE 754 float32 little-endian (NumPy ``<f4``).
    Отсутствующий файл даёт None при чтении, существующая некорректная БД —
    исключение. Порядок карточек при загрузке — по ID, порядок входа не хранится.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).absolute()

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        uri = self.path.as_uri() + ("?mode=rwc" if write else "?mode=ro")
        with closing(sqlite3.connect(uri, uri=True, isolation_level=None)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            # Контекст соединения завершает транзакцию, но сам его не закрывает.
            with connection:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield connection

    @staticmethod
    def _check_schema(connection: sqlite3.Connection) -> None:
        # Проверяем структуру, не читая векторы в режиме BM25.
        expected = {
            "cards": {"id", "title", "abstract", "url", "keywords", "author", "year"},
            "vectors": {"card_id", "vector"},
            "index_state": {"id", "kind", "metadata"},
        }
        for table, columns in expected.items():
            actual = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            if actual != columns:
                raise ValueError(f"Invalid SQLite schema: {table}")

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        # executescript делает неявный COMMIT: DDL выполняется по одному запросу.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS cards (
                id TEXT PRIMARY KEY NOT NULL,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                url TEXT NOT NULL,
                keywords TEXT NOT NULL,
                author TEXT,
                year INTEGER
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS vectors (
                card_id TEXT PRIMARY KEY NOT NULL REFERENCES cards(id),
                vector BLOB NOT NULL CHECK(typeof(vector) = 'blob')
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS index_state (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                kind TEXT NOT NULL CHECK(kind IN ('cards', 'snapshot')),
                metadata TEXT,
                CHECK((kind = 'cards' AND metadata IS NULL)
                   OR (kind = 'snapshot' AND metadata IS NOT NULL))
            )
        """)

    @staticmethod
    def _validate_cards(cards: tuple[Thesis, ...]) -> tuple[Thesis, ...]:
        cards = tuple(Thesis.model_validate(card.model_dump()) for card in cards)
        if not cards:
            raise ValueError("Corpus must not be empty")
        if len({card.id for card in cards}) != len(cards):
            raise ValueError("Card IDs must be unique")
        return cards

    def replace_cards(self, cards: tuple[Thesis, ...]) -> None:
        """Заменить непустой корпус, атомарно удалив векторы и метаданные."""
        self._replace(self._validate_cards(cards), None)

    def replace_snapshot(self, snapshot: IndexSnapshot) -> None:
        cards = self._validate_cards(snapshot.cards)
        metadata = IndexMetadata.model_validate(snapshot.metadata.model_dump())
        if metadata.model.normalization != "l2":
            raise ValueError("Index vectors must be L2-normalized")
        if len(set(snapshot.vector_ids)) != len(snapshot.vector_ids):
            raise ValueError("Vector IDs must be unique")
        if set(snapshot.vector_ids) != {card.id for card in cards}:
            raise ValueError("Card IDs and vector IDs must match")
        vectors = validate_vectors(
            snapshot.vectors, shape=(len(cards), metadata.model.dimension),
        )
        self._replace(cards, IndexSnapshot(cards, snapshot.vector_ids, vectors, metadata))

    def _replace(self, cards: tuple[Thesis, ...], snapshot: IndexSnapshot | None) -> None:
        try:
            self.path.stat()
        except FileNotFoundError:
            existing = False
        else:
            existing = True
        with self._transaction(write=True) as connection:
            if existing:
                self._check_schema(connection)
            else:
                self._create_schema(connection)
            connection.execute("DELETE FROM vectors")
            connection.execute("DELETE FROM index_state")
            connection.execute("DELETE FROM cards")
            connection.executemany(
                "INSERT INTO cards (id, title, abstract, url, keywords, author, year) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (card.id, card.title, card.abstract, str(card.url),
                     json.dumps(card.keywords, ensure_ascii=False), card.author, card.year)
                    for card in cards
                ],
            )
            if snapshot is not None:
                connection.executemany(
                    "INSERT INTO vectors (card_id, vector) VALUES (?, ?)",
                    [
                        (card_id, vector.astype("<f4").tobytes())
                        for card_id, vector in zip(snapshot.vector_ids, snapshot.vectors)
                    ],
                )
            connection.execute(
                "INSERT INTO index_state (id, kind, metadata) VALUES (1, ?, ?)",
                ("cards", None) if snapshot is None else
                ("snapshot", snapshot.metadata.model_dump_json()),
            )

    def load_cards(self, *, validate_snapshot: bool = True) -> tuple[Thesis, ...] | None:
        """Прочитать корпус любого режима; None только для отсутствующего файла.

        По умолчанию проверяется и полный индекс. Для лексического поиска
        validate_snapshot=False пропускает чтение векторов и проверку модели.
        """
        loaded = self._load(validate_snapshot=validate_snapshot)
        return None if loaded is None else loaded[0]

    def load_snapshot(self) -> IndexSnapshot | None:
        """None — отсутствующий файл или корректный корпус в режиме cards."""
        loaded = self._load()
        return None if loaded is None else loaded[1]

    def _load(
        self, *, validate_snapshot: bool = True,
    ) -> tuple[tuple[Thesis, ...], IndexSnapshot | None] | None:
        try:
            self.path.stat()
        except FileNotFoundError:
            return None
        with self._transaction(write=False) as connection:
            self._check_schema(connection)
            states = connection.execute("SELECT id, kind, metadata FROM index_state").fetchall()
            if len(states) != 1 or states[0]["id"] != 1:
                raise ValueError("Invalid index state: expected one state record")
            state = states[0]
            card_rows = connection.execute(
                "SELECT id, title, abstract, url, keywords, author, year FROM cards ORDER BY id"
            ).fetchall()
            cards = tuple(
                Thesis.model_validate({**dict(row), "keywords": json.loads(row["keywords"])})
                for row in card_rows
            )
            if not cards or len({card.id for card in cards}) != len(cards):
                raise ValueError("Stored corpus must be nonempty with unique IDs")
            if any(card.id != row["id"] for card, row in zip(cards, card_rows)):
                raise ValueError("Stored card IDs must be canonical")
            if state["kind"] == "cards":
                if state["metadata"] is not None:
                    raise ValueError("Cards-only state contains model metadata")
            elif state["kind"] != "snapshot" or state["metadata"] is None:
                raise ValueError("Invalid full index state or missing metadata")
            if not validate_snapshot:
                return cards, None
            rows = connection.execute(
                "SELECT card_id, vector FROM vectors ORDER BY card_id"
            ).fetchall()
            if state["kind"] == "cards":
                if rows:
                    raise ValueError("Cards-only state contains vectors or model metadata")
                return cards, None
            metadata = IndexMetadata.model_validate_json(state["metadata"])
            if metadata.model.normalization != "l2":
                raise ValueError("Stored vectors must be L2-normalized")
            vector_ids = tuple(row["card_id"] for row in rows)
            if len(set(vector_ids)) != len(vector_ids) or set(vector_ids) != {card.id for card in cards}:
                raise ValueError("Stored card IDs and vector IDs must match uniquely")
            dimension = metadata.model.dimension
            # Сначала сверяем длины BLOB: повреждённая размерность не должна
            # приводить к выделению произвольно большой матрицы.
            for row in rows:
                blob = row["vector"]
                if not isinstance(blob, bytes) or len(blob) != dimension * 4:
                    raise ValueError(f"Invalid float32 BLOB for card {row['card_id']!r}")
            vectors = np.empty((len(rows), dimension), dtype=np.float32)
            for i, row in enumerate(rows):
                vectors[i] = np.frombuffer(row["vector"], dtype="<f4")
            vectors = validate_vectors(vectors, shape=(len(rows), dimension))
            return cards, IndexSnapshot(cards, vector_ids, vectors, metadata)

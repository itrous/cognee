"""Opt-in halfvec columns, HNSW index and scan tuning of the pgvector adapter, and the
client-side Matryoshka cut in the OpenAI-compatible embedding engine."""

import asyncio
import math
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest

pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql

import cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter as pg_module
from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
    OpenAICompatibleEmbeddingEngine,
)
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
    PGVectorAdapter,
)


def _adapter():
    from pgvector.sqlalchemy import Vector

    adapter = PGVectorAdapter.__new__(PGVectorAdapter)
    adapter.Vector = Vector
    adapter.engine = SimpleNamespace(dialect=postgresql.dialect())
    adapter.schema = None
    return adapter


class _Session:
    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(str(statement))


def test_column_type_follows_the_halfvec_flag(monkeypatch):
    from pgvector.sqlalchemy import HALFVEC, Vector

    monkeypatch.delenv("PGVECTOR_HALFVEC", raising=False)
    off = _adapter()._vector_column_type(2048)
    monkeypatch.setenv("PGVECTOR_HALFVEC", "true")
    on = _adapter()._vector_column_type(2048)

    assert type(off) is Vector and off.dim == 2048
    assert type(on) is HALFVEC and on.dim == 2048


def test_index_ddl_uses_the_operator_class_of_the_column_type(monkeypatch):
    monkeypatch.delenv("PGVECTOR_HALFVEC", raising=False)
    plain = _adapter()._hnsw_index_ddl("Entity_name")
    monkeypatch.setenv("PGVECTOR_HALFVEC", "1")
    half = _adapter()._hnsw_index_ddl("Entity_name")

    assert "vector_cosine_ops" in plain and "halfvec" not in plain
    assert "halfvec_cosine_ops" in half
    assert 'ON "Entity_name" USING hnsw' in half
    assert half.startswith("CREATE INDEX IF NOT EXISTS ")


def test_index_name_stays_within_the_identifier_limit():
    ddl = _adapter()._hnsw_index_ddl("c" * 80)
    index_name = ddl.split("EXISTS ")[1].split(" ON")[0].strip('"')
    assert len(index_name) <= 63


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (15, ["SET LOCAL hnsw.ef_search = 100", "SET LOCAL hnsw.iterative_scan = strict_order"]),
        (500, ["SET LOCAL hnsw.ef_search = 500", "SET LOCAL hnsw.iterative_scan = strict_order"]),
        # The edge of the bound: 1000 still uses the index, 1001 is already a wide request.
        (
            1000,
            ["SET LOCAL hnsw.ef_search = 1000", "SET LOCAL hnsw.iterative_scan = strict_order"],
        ),
        (1001, ["SET LOCAL enable_indexscan = off"]),
        (508_000, ["SET LOCAL enable_indexscan = off"]),
    ],
)
async def test_scan_tuning_keeps_a_wide_request_off_the_index(limit, expected):
    session = _Session()
    await _adapter()._tune_index_scan(session, limit)
    assert session.statements == expected


@pytest.mark.asyncio
async def test_stored_width_is_read_from_halfvec_columns():
    adapter = _adapter()
    session = SimpleNamespace(
        execute=_return_scalars(["halfvec(2048)", "halfvec(2048)", "text"]),
    )

    @asynccontextmanager
    async def get_async_session():
        yield session

    adapter.get_async_session = get_async_session
    assert await adapter.get_stored_vector_size() == 2048


def _return_scalars(values):
    async def execute(*_args, **_kwargs):
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: values))

    return execute


def _engine(dimensions):
    engine = OpenAICompatibleEmbeddingEngine.__new__(OpenAICompatibleEmbeddingEngine)
    engine.dimensions = dimensions
    return engine


def test_wide_vector_is_left_alone_without_the_flag(monkeypatch):
    monkeypatch.delenv("EMBEDDING_TRUNCATE_TO_DIMENSIONS", raising=False)
    vector = list(np.linspace(0.1, 1.0, 4096))
    assert _engine(2048)._fit_to_dimensions(vector) == vector


def test_wide_vector_is_cut_to_its_normalised_prefix(monkeypatch):
    monkeypatch.setenv("EMBEDDING_TRUNCATE_TO_DIMENSIONS", "true")
    vector = list(np.linspace(0.1, 1.0, 4096))
    cut = _engine(2048)._fit_to_dimensions(vector)

    prefix = np.asarray(vector[:2048])
    assert len(cut) == 2048
    assert math.isclose(sum(x * x for x in cut), 1.0, rel_tol=1e-9)
    assert np.allclose(cut, prefix / np.linalg.norm(prefix))


def test_vector_of_the_configured_width_is_untouched(monkeypatch):
    monkeypatch.setenv("EMBEDDING_TRUNCATE_TO_DIMENSIONS", "true")
    vector = [3.0, 4.0]
    assert _engine(2)._fit_to_dimensions(vector) == vector


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["true", None])
async def test_embed_text_fits_a_wide_provider_vector(monkeypatch, flag):
    """The cut is wired into ``embed_text`` on the provider reply, after v1.6.5's decoding.

    Through the engine, not ``_fit_to_dimensions`` alone: a provider reply of 4096 floats
    must reach the caller as 2048 normalised ones with the flag on, and unchanged without it.
    """
    monkeypatch.delenv("MOCK_EMBEDDING", raising=False)
    # The constructor resolves a tokenizer for the model id from the HuggingFace hub; this
    # test is about the vector, so keep it off the network.
    monkeypatch.setattr(
        OpenAICompatibleEmbeddingEngine,
        "get_tokenizer",
        lambda self: SimpleNamespace(count_tokens=len),
    )
    if flag is None:
        monkeypatch.delenv("EMBEDDING_TRUNCATE_TO_DIMENSIONS", raising=False)
    else:
        monkeypatch.setenv("EMBEDDING_TRUNCATE_TO_DIMENSIONS", flag)
    engine = OpenAICompatibleEmbeddingEngine(
        model="test-model",
        dimensions=2048,
        max_completion_tokens=8191,
        endpoint="http://localhost:8099",
        api_key="test-key",
    )
    wide = list(np.linspace(0.1, 1.0, 4096))
    item = MagicMock()
    item.embedding = wide
    response = MagicMock()
    response.data = [item]
    engine._client = MagicMock()
    engine._client.embeddings.create = AsyncMock(return_value=response)

    result = await engine.embed_text(["text"])

    engine._client.embeddings.create.assert_awaited_once()
    if flag is None:
        assert result[0] == wide and len(result[0]) == 4096
        return
    prefix = np.asarray(wide[:2048])
    assert len(result[0]) == 2048
    assert abs(np.linalg.norm(result[0]) - 1) <= 1e-6
    assert np.allclose(result[0], prefix / np.linalg.norm(prefix))


def _index_name(ddl):
    return ddl.split("EXISTS ")[1].split(" ON")[0].strip('"')


@pytest.mark.parametrize("prefix", ["c" * 60, "é" * 40])
def test_long_collection_names_get_distinct_index_names(prefix):
    # Two collections sharing their first 60 characters must not collapse to one index name:
    # CREATE INDEX IF NOT EXISTS would then silently leave the second table without its index.
    first = _index_name(_adapter()._hnsw_index_ddl(prefix + "a"))
    second = _index_name(_adapter()._hnsw_index_ddl(prefix + "b"))
    assert first != second
    assert len(first.encode()) <= 63 and len(second.encode()) <= 63


def test_short_collection_name_keeps_the_readable_index_name():
    assert _index_name(_adapter()._hnsw_index_ddl("Entity_name")) == "ix_Entity_name_vector_hnsw"


# --- filtered search: the index first, the exact scan only for a short answer ----------------


class _FakeQuery:
    """Stands in for the SQLAlchemy select: every builder step returns itself."""

    def where(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def select_from(self, *_args, **_kwargs):
        return _CountQuery()


class _CountQuery(_FakeQuery):
    """The row count a search with ``limit=None`` takes first."""


class _SearchSession:
    """A SELECT answers with the next queued row set; any other statement is only recorded."""

    def __init__(self, *row_sets, count=None):
        self.row_sets = list(row_sets)
        self.count = count
        self.log = []

    async def execute(self, statement):
        if isinstance(statement, _CountQuery):
            self.log.append("COUNT")
            return SimpleNamespace(scalar_one=lambda: self.count)
        if isinstance(statement, _FakeQuery):
            self.log.append("SELECT")
            rows = self.row_sets.pop(0)
            return SimpleNamespace(all=lambda: rows)
        self.log.append(str(statement))
        return SimpleNamespace()


def _distances(*values):
    return [SimpleNamespace(id=str(uuid4()), payload=None, similarity=v) for v in values]


def _search_adapter(monkeypatch, session):
    monkeypatch.setattr(pg_module, "select", lambda *_args, **_kwargs: _FakeQuery())
    distance = SimpleNamespace(label=lambda _name: "distance_col")
    table = SimpleNamespace(
        c=SimpleNamespace(
            id="id_col",
            payload="payload_col",
            vector=SimpleNamespace(cosine_distance=lambda _vector: distance),
        )
    )
    adapter = _adapter()
    adapter.get_table = AsyncMock(return_value=table)

    @asynccontextmanager
    async def get_async_session():
        yield session

    adapter.get_async_session = get_async_session
    return adapter


_EXACT = "SET LOCAL enable_indexscan = off"


def _tuned(limit):
    return [
        f"SET LOCAL hnsw.ef_search = {max(100, limit)}",
        "SET LOCAL hnsw.iterative_scan = strict_order",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hnsw", "node_name", "limit", "count", "row_sets", "log", "scores"),
    [
        pytest.param(
            True,
            ["A"],
            3,
            None,
            [_distances(0.1, 0.2, 0.3)],
            [*_tuned(3), "SELECT"],
            [0.1, 0.2, 0.3],
            id="filtered-full-answer-stays-on-the-index",
        ),
        pytest.param(
            True,
            ["A"],
            3,
            None,
            [_distances(0.4), _distances(0.1, 0.2, 0.3)],
            [*_tuned(3), "SELECT", _EXACT, "SELECT"],
            [0.1, 0.2, 0.3],
            id="filtered-short-answer-is-rechecked-exactly",
        ),
        pytest.param(
            True,
            ["A"],
            3,
            None,
            [[], []],
            [*_tuned(3), "SELECT", _EXACT, "SELECT"],
            [],
            id="filtered-empty-answer-is-rechecked-and-may-stay-empty",
        ),
        pytest.param(
            True,
            ["A"],
            1000,
            None,
            [_distances(0.4), _distances(0.1, 0.4)],
            [*_tuned(1000), "SELECT", _EXACT, "SELECT"],
            [0.1, 0.4],
            id="filtered-short-answer-at-the-bound-1000-is-rechecked",
        ),
        pytest.param(
            True,
            ["A"],
            1001,
            None,
            [_distances(0.4)],
            [_EXACT, "SELECT"],
            [0.4],
            id="filtered-above-the-bound-is-exact-once",
        ),
        pytest.param(
            True,
            None,
            3,
            None,
            [_distances(0.4)],
            [*_tuned(3), "SELECT"],
            [0.4],
            id="unfiltered-short-answer-is-final",
        ),
        pytest.param(
            False,
            ["A"],
            3,
            None,
            [_distances(0.4)],
            ["SELECT"],
            [0.4],
            id="index-flag-off-leaves-the-scan-alone",
        ),
        pytest.param(
            True,
            ["A"],
            None,
            3,
            [_distances(0.4), _distances(0.1, 0.2)],
            ["COUNT", *_tuned(3), "SELECT", _EXACT, "SELECT"],
            [0.1, 0.2],
            id="whole-small-collection-filtered-is-rechecked",
        ),
        pytest.param(
            True,
            ["A"],
            None,
            5000,
            [_distances(0.4)],
            ["COUNT", _EXACT, "SELECT"],
            [0.4],
            id="whole-large-collection-filtered-is-exact-once",
        ),
    ],
)
async def test_filtered_search_uses_the_index_and_rechecks_only_a_short_answer(
    monkeypatch, hnsw, node_name, limit, count, row_sets, log, scores
):
    if hnsw:
        monkeypatch.setenv("PGVECTOR_HNSW_INDEX", "true")
    else:
        monkeypatch.delenv("PGVECTOR_HNSW_INDEX", raising=False)
    session = _SearchSession(*row_sets, count=count)
    adapter = _search_adapter(monkeypatch, session)

    results = await adapter.search(
        "Entity_name", query_vector=[0.1, 0.2, 0.3], limit=limit, node_name=node_name
    )

    assert session.log == log
    assert [result.score for result in results] == scores
    # Every queued row set was needed: nothing is fetched twice by accident or left unread.
    assert session.row_sets == []


# --- a collection wider than pgvector can index is created without the index ------------------


class _Begin:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *_exc):
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("halfvec", "size", "indexed"),
    [
        pytest.param(True, 2048, True, id="halfvec-2048"),
        pytest.param(True, 4000, True, id="halfvec-at-its-limit"),
        pytest.param(True, 4001, False, id="halfvec-over-its-limit"),
        pytest.param(False, 2000, True, id="vector-at-its-limit"),
        pytest.param(False, 2001, False, id="vector-over-its-limit"),
        pytest.param(False, 2048, False, id="vector-2048-with-the-halfvec-flag-off"),
    ],
)
async def test_wide_collection_is_created_without_an_index_pgvector_would_refuse(
    monkeypatch, halfvec, size, indexed
):
    from pgvector.sqlalchemy import HALFVEC, Vector

    monkeypatch.setenv("PGVECTOR_HNSW_INDEX", "true")
    if halfvec:
        monkeypatch.setenv("PGVECTOR_HALFVEC", "true")
    else:
        monkeypatch.delenv("PGVECTOR_HALFVEC", raising=False)
    warning = MagicMock()
    monkeypatch.setattr(pg_module, "logger", MagicMock(warning=warning))
    connection = SimpleNamespace(run_sync=AsyncMock(), exec_driver_sql=AsyncMock())
    adapter = _adapter()
    adapter.embedding_engine = SimpleNamespace(get_vector_size=lambda: size)
    adapter.has_collection = AsyncMock(return_value=False)
    adapter.VECTOR_DB_LOCK = asyncio.Lock()
    adapter._metadata = MetaData()
    adapter.engine = SimpleNamespace(dialect=postgresql.dialect(), begin=lambda: _Begin(connection))
    collection = f"wide_{'halfvec' if halfvec else 'vector'}_{size}"

    try:
        await adapter.create_collection(collection)

        # The table is created either way: a refused CREATE INDEX would roll it back with it.
        created = [call for call in connection.run_sync.await_args_list if "tables" in call.kwargs]
        assert len(created) == 1
        (table,) = created[0].kwargs["tables"]
        assert table.name == collection
        assert type(table.c.vector.type) is (HALFVEC if halfvec else Vector)
        assert table.c.vector.type.dim == size
        opclass = "halfvec_cosine_ops" if halfvec else "vector_cosine_ops"
        if indexed:
            assert connection.exec_driver_sql.await_count == 1
            assert opclass in connection.exec_driver_sql.await_args.args[0]
            warning.assert_not_called()
        else:
            connection.exec_driver_sql.assert_not_awaited()
            warning.assert_called_once()
            limit = 4000 if halfvec else 2000
            assert warning.call_args.args[1:] == (collection, size, limit)
        # The collection is still reflected afterwards, so searches can use it.
        assert connection.run_sync.await_args.kwargs == {"only": [collection]}
    finally:
        registered = pg_module.Base.metadata.tables.get(collection)
        if registered is not None:
            pg_module.Base.metadata.remove(registered)

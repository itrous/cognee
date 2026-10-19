"""Opt-in halfvec columns, HNSW index and scan tuning of the pgvector adapter, and the
client-side Matryoshka cut in the OpenAI-compatible embedding engine."""

import math
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

pytest.importorskip("asyncpg")
pytest.importorskip("pgvector")

from sqlalchemy.dialects import postgresql

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

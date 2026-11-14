from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.tasks.storage.index_graph_edges import create_edge_type_datapoints, index_graph_edges


@pytest.mark.asyncio
async def test_index_graph_edges_success():
    """Test that index_graph_edges retrieves edges and delegates to index_data_points."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (
        None,
        [
            [{"relationship_name": "rel1"}, {"relationship_name": "rel1"}],
            [{"relationship_name": "rel2"}],
        ],
    )
    mock_index_data_points = AsyncMock()

    with patch.dict(
        index_graph_edges.__globals__,
        {
            "get_graph_engine": AsyncMock(return_value=mock_graph_engine),
            "index_data_points": mock_index_data_points,
        },
    ):
        await index_graph_edges()

    mock_graph_engine.get_graph_data.assert_awaited_once()
    mock_index_data_points.assert_awaited_once()

    call_args = mock_index_data_points.call_args[0][0]
    assert len(call_args) == 2
    assert all(hasattr(item, "relationship_name") for item in call_args)


@pytest.mark.asyncio
async def test_index_graph_edges_no_relationships():
    """Test that index_graph_edges handles empty relationships correctly."""
    mock_graph_engine = AsyncMock()
    mock_graph_engine.get_graph_data.return_value = (None, [])
    mock_index_data_points = AsyncMock()

    with patch.dict(
        index_graph_edges.__globals__,
        {
            "get_graph_engine": AsyncMock(return_value=mock_graph_engine),
            "index_data_points": mock_index_data_points,
        },
    ):
        await index_graph_edges()

    mock_graph_engine.get_graph_data.assert_awaited_once()
    mock_index_data_points.assert_awaited_once()

    call_args = mock_index_data_points.call_args[0][0]
    assert len(call_args) == 0


def test_create_edge_type_datapoints_uses_nonblank_edge_text():
    edges = [
        (
            "source",
            "target",
            "related_to",
            {"relationship_name": "related_to", "edge_text": " custom text "},
        )
    ]

    datapoints = create_edge_type_datapoints(edges)

    assert len(datapoints) == 1
    assert datapoints[0].relationship_name == "custom text"


def test_create_edge_type_datapoints_uses_edge_text_without_relationship_property():
    edges = [("source", "target", "related_to", {"edge_text": " custom text "})]

    datapoints = create_edge_type_datapoints(edges)

    assert len(datapoints) == 1
    assert datapoints[0].relationship_name == "custom text"


def test_create_edge_type_datapoints_falls_back_from_blank_edge_text_to_relationship_name():
    edges = [
        ("source", "target", "related_to", {"relationship_name": "related_to", "edge_text": ""}),
        ("source", "target", "mentions", {"relationship_name": "mentions", "edge_text": "   "}),
        ("source", "target", "works_at", {"relationship_name": "works_at", "edge_text": None}),
    ]

    datapoints = create_edge_type_datapoints(edges)

    relationship_names = {datapoint.relationship_name for datapoint in datapoints}
    assert relationship_names == {"related_to", "mentions", "works_at"}


def test_create_edge_type_datapoints_skips_empty_retrieval_text():
    edges = [("source", "target", "", {"relationship_name": "", "edge_text": ""})]

    assert create_edge_type_datapoints(edges) == []


@pytest.mark.asyncio
async def test_index_graph_edges_initialization_error():
    """Test that index_graph_edges raises a RuntimeError if initialization fails."""
    with (
        patch.dict(
            index_graph_edges.__globals__,
            {
                "get_graph_engine": AsyncMock(side_effect=Exception("Graph engine failed")),
                "get_vector_engine_async": AsyncMock(return_value=AsyncMock()),
            },
        ),
        pytest.raises(RuntimeError, match="Graph edge indexing error"),
    ):
        await index_graph_edges()


def test_normalized_edge_type_batch():
    """Spellings with one EdgeType id are one datapoint: first text kept, counts summed.

    A raw-text Counter would emit two datapoints with the same id, and the second
    write would replace the first's payload text.
    """
    from cognee.modules.graph.models.EdgeType import EdgeType

    edges = [
        ("a", "b", "works", {"edge_text": "Alice Works"}),
        ("c", "d", "works", {"edge_text": "alice_works"}),
        ("e", "f", "works", {"edge_text": "Alice Works"}),
        ("g", "h", "lives_in", {"edge_text": "Bob lives in Oslo"}),
    ]

    datapoints = create_edge_type_datapoints(edges)

    assert [(dp.relationship_name, dp.number_of_edges) for dp in datapoints] == [
        ("Alice Works", 3),
        ("Bob lives in Oslo", 1),
    ]
    assert len({dp.id for dp in datapoints}) == 2
    assert datapoints[0].id == EdgeType.id_for("alice_works")


@pytest.mark.asyncio
async def test_full_graph_index_reads_edges_inside_the_lifecycle(monkeypatch):
    import sys

    pytest.importorskip("neo4j")
    pytest.importorskip("asyncpg")
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    calls = []
    lifecycle_module = sys.modules["cognee.tasks.storage.edge_type_lifecycle"]

    class _RecordedLock:
        def __init__(self, vector_adapter):
            pass

        async def acquire(self):
            calls.append("lock")

        async def ensure_held(self):
            pass

        async def release(self):
            calls.append("unlock")

    monkeypatch.setattr(lifecycle_module, "_AdvisoryLock", _RecordedLock)
    graph = AsyncMock(spec=Neo4jAdapter)

    async def graph_data():
        calls.append("read_graph")
        return [], [("a", "b", "r", {"edge_text": "a r b"})]

    graph.get_graph_data = AsyncMock(side_effect=graph_data)
    vector = MagicMock(spec=PGVectorAdapter)
    index = AsyncMock(side_effect=lambda *a, **k: calls.append("index"))

    with patch.dict(
        index_graph_edges.__globals__,
        {
            "get_graph_engine": AsyncMock(return_value=graph),
            "get_vector_engine_async": AsyncMock(return_value=vector),
            "index_data_points": index,
        },
    ):
        await index_graph_edges()

    assert calls == ["lock", "read_graph", "index", "unlock"]
    assert index.await_args.kwargs["vector_engine"] is vector


@pytest.mark.asyncio
async def test_empty_batch_takes_no_lifecycle_lock(monkeypatch):
    import sys

    pytest.importorskip("neo4j")
    pytest.importorskip("asyncpg")
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    lifecycle_module = sys.modules["cognee.tasks.storage.edge_type_lifecycle"]

    class _UnreachablePostgres:
        def __init__(self, vector_adapter):
            pass

        async def acquire(self):
            raise ConnectionError("PostgreSQL is down")

    monkeypatch.setattr(lifecycle_module, "_AdvisoryLock", _UnreachablePostgres)
    index = AsyncMock()

    with patch.dict(index_graph_edges.__globals__, {"index_data_points": index}):
        await index_graph_edges(
            [],
            vector_engine=MagicMock(spec=PGVectorAdapter),
            graph_engine=AsyncMock(spec=Neo4jAdapter),
        )

    index.assert_awaited_once()
    assert index.await_args.args[0] == []

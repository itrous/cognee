from unittest.mock import AsyncMock

import pytest

pytest.importorskip("neo4j")

from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter


def _make_adapter() -> Neo4jAdapter:
    return Neo4jAdapter(
        "bolt://unused",
        graph_database_allow_anonymous=True,
        driver=object(),
    )


@pytest.mark.asyncio
async def test_has_edges_returns_existing_edge_tuples():
    """has_edges must return the subset of input edges that exist, as
    (from_node, to_node, relationship_name) tuples — the graph_db_interface
    contract shared with the ladybug/postgres/neptune adapters — not a list
    of booleans."""
    adapter = _make_adapter()
    adapter.query = AsyncMock(
        return_value=[
            {
                "from_node": "node-a",
                "to_node": "node-b",
                "relationship_name": "knows",
                "edge_exists": True,
            },
            {
                "from_node": "node-a",
                "to_node": "node-c",
                "relationship_name": "likes",
                "edge_exists": False,
            },
        ]
    )

    result = await adapter.has_edges([("node-a", "node-b", "knows"), ("node-a", "node-c", "likes")])

    assert result == [("node-a", "node-b", "knows")]


@pytest.mark.asyncio
async def test_has_edges_matches_on_id_property_not_internal_id():
    """Nodes are stored with an ``id`` property (a UUID string); Neo4j's
    internal ``id()`` is an integer and never equals it, which made the
    previous query match nothing and existing-edge deduplication silently
    return empty."""
    adapter = _make_adapter()
    adapter.query = AsyncMock(return_value=[])

    await adapter.has_edges([("node-a", "node-b", "knows")])

    query = adapter.query.await_args.args[0]
    assert "a.id = edge.from_node" in query
    assert "b.id = edge.to_node" in query
    assert "id(a)" not in query
    assert "id(b)" not in query


@pytest.mark.asyncio
async def test_get_connections_preserves_edge_properties():
    """get_connections must merge the relationship's properties into the edge
    dict (as the postgres/ladybug/neptune adapters do), not reduce it to just
    the relationship name — consumers such as legacy_delete read keys like
    ``edge_text`` from it."""
    adapter = _make_adapter()
    relation_tuple = ({"id": "node-a"}, "knows", {"id": "node-b"})
    adapter.query = AsyncMock(
        side_effect=[
            [
                {
                    "relation": relation_tuple,
                    "relation_properties": {"edge_text": "a knows b", "weight": 0.7},
                }
            ],
            [],
        ]
    )

    connections = await adapter.get_connections("node-b")

    assert connections == [
        (
            {"id": "node-a"},
            {"relationship_name": "knows", "edge_text": "a knows b", "weight": 0.7},
            {"id": "node-b"},
        )
    ]


@pytest.mark.asyncio
async def test_get_connections_handles_missing_edge_properties():
    """Edges without extra properties still yield a relationship_name-only dict."""
    adapter = _make_adapter()
    relation_tuple = ({"id": "node-a"}, "knows", {"id": "node-b"})
    adapter.query = AsyncMock(
        side_effect=[
            [],
            [{"relation": relation_tuple, "relation_properties": {}}],
        ]
    )

    connections = await adapter.get_connections("node-a")

    assert connections == [({"id": "node-a"}, {"relationship_name": "knows"}, {"id": "node-b"})]


# --- Connection retry at the adapter boundary (N1 / N2 / N3) ------------------
# A scripted driver: each opened session takes the next fault plan, so a test
# controls exactly which attempt fails and where (run, data, stream, consume,
# transaction close, session close). Mocking ``adapter.query`` alone would never
# exercise the session boundary the retry is about.

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from neo4j.exceptions import (  # noqa: E402
    ClientError,
    IncompleteCommit,
    ServiceUnavailable,
    SessionExpired,
)

import cognee.infrastructure.databases.graph.neo4j_driver.adapter as adapter_module  # noqa: E402
import cognee.infrastructure.databases.graph.neo4j_driver.deadlock_retry as deadlock_retry_module  # noqa: E402
from cognee.infrastructure.databases.provenance import EdgeIdentity  # noqa: E402
from cognee.infrastructure.engine import DataPoint  # noqa: E402
from cognee.modules.graph.models.EdgeType import EdgeType  # noqa: E402


class _ScriptedResult:
    def __init__(self, rows, faults):
        self._rows = rows
        self._faults = faults

    async def data(self):
        if "data" in self._faults:
            raise self._faults["data"]
        return [dict(row) for row in self._rows]

    def __aiter__(self):
        return self._stream()

    async def _stream(self):
        fail_at = self._faults.get("stream_at", 0)
        for index, row in enumerate(self._rows):
            if "stream" in self._faults and index == fail_at:
                raise self._faults["stream"]
            yield row
        if "stream" in self._faults and fail_at >= len(self._rows):
            raise self._faults["stream"]

    async def consume(self):
        if "consume" in self._faults:
            raise self._faults["consume"]


class _ScriptedTransaction:
    def __init__(self, session):
        self._session = session

    async def run(self, query, parameters=None):
        return self._session._result(query, parameters)

    async def close(self):
        self._session.driver.events.append("tx.close")
        if "tx_close" in self._session.faults:
            raise self._session.faults["tx_close"]


class _ScriptedSession:
    def __init__(self, driver, faults):
        self.driver = driver
        self.faults = faults
        self._runs = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.driver.events.append("session.close")
        if "close" in self.faults:
            raise self.faults["close"]
        return False

    def _result(self, query, parameters):
        self._runs += 1
        self.driver.queries.append(query)
        if "run" in self.faults and self._runs == self.faults.get("run_at", 1):
            raise self.faults["run"]
        return _ScriptedResult(self.driver.responder(query, parameters), self.faults)

    async def run(self, query, parameters=None):
        return self._result(query, parameters)

    def begin_transaction(self):
        return _ScriptedTransaction(self)


class _ScriptedDriver:
    """Session N gets ``plans[N]`` (no fault once the plans run out)."""

    def __init__(self, plans=(), responder=None):
        self.plans = list(plans)
        self.responder = responder or (lambda query, parameters: [])
        self.sessions_opened = 0
        self.queries: list[str] = []
        self.events: list[str] = []
        self.close = AsyncMock()

    def session(self, database=None):
        faults = self.plans[self.sessions_opened] if self.sessions_opened < len(self.plans) else {}
        self.sessions_opened += 1
        return _ScriptedSession(self, faults)


def _adapter_with(driver) -> Neo4jAdapter:
    return Neo4jAdapter("bolt://unused", graph_database_allow_anonymous=True, driver=driver)


def _default_rows(query, parameters):
    if "node_exists" in query:
        return [{"node_exists": True}]
    return []


@pytest.fixture
def recorded_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(deadlock_retry_module, "asyncio", SimpleNamespace(sleep=sleep))
    return sleep


class _Point(DataPoint):
    name: str = "x"


_RETRYABLE_CALLS = {
    # B.3 reads
    "is_empty": ((), 1),
    "get_node_delete_plan_data": ((["n1"],), 1),
    "get_edge_delete_plan_data": (([EdgeIdentity("a", "b", "r")],), 1),
    "get_existing_edge_texts": ((["text"],), 1),
    "find_node_source_refs_by_dataset": (("ds",), 1),
    "find_edge_source_refs_by_dataset": (("ds",), 1),
    "get_graph_metadata": ((), 1),
    "get_graph_data": ((), 2),
    "get_neighborhood": ((["n1"],), 3),
    "get_filtered_graph_data": (([{"type": ["Entity"]}],), 2),
    "get_incident_edge_retrieval_texts": ((["n1"],), 1),
    # B.4 writes
    "add_nodes": (([_Point()],), 1),
    "add_edges": (([("a", "b", "r", {"edge_text": "a r b"})],), 1),
    "delete_nodes": ((["n1"],), 1),
    "delete_edge_triples": (([EdgeIdentity("a", "b", "r")],), 1),
    "set_graph_metadata": (({"k": "v"},), 1),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", sorted(_RETRYABLE_CALLS))
@pytest.mark.parametrize("error_class", [ServiceUnavailable, SessionExpired])
async def test_retryable_query_consumes_new_session(recorded_sleep, method_name, error_class):
    args, queries = _RETRYABLE_CALLS[method_name]
    driver = _ScriptedDriver([{"run": error_class("dropped")}], responder=_default_rows)
    adapter = _adapter_with(driver)

    await getattr(adapter, method_name)(*args)

    # The failed attempt took one session, every statement of the method one more.
    assert driver.sessions_opened == queries + 1
    assert recorded_sleep.await_count == 1
    driver.close.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["data", "close"])
async def test_retryable_query_restarts_after_data_or_close_failure(recorded_sleep, fault):
    driver = _ScriptedDriver(
        [{fault: ServiceUnavailable("dropped")}],
        responder=lambda query, parameters: [{"key": "k", "value": "v"}],
    )
    adapter = _adapter_with(driver)

    assert await adapter.get_graph_metadata() == {"k": "v"}
    assert driver.sessions_opened == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name, args",
    [
        ("get_edges", ("n1",)),
        ("remove_belongs_to_set_tags", (["tag"],)),
        ("get_entity_type_names", (["n1"],)),
    ],
)
async def test_unlisted_calls_keep_the_old_decorator(recorded_sleep, method_name, args):
    driver = _ScriptedDriver([{"run": ServiceUnavailable("dropped")}])
    adapter = _adapter_with(driver)

    with pytest.raises(ServiceUnavailable):
        await getattr(adapter, method_name)(*args)
    assert driver.sessions_opened == 1


@pytest.mark.asyncio
async def test_raw_query_does_not_replay(recorded_sleep):
    driver = _ScriptedDriver([{"run": ServiceUnavailable("dropped")}])
    adapter = _adapter_with(driver)

    with pytest.raises(ServiceUnavailable):
        await adapter.query("CREATE (n:Thing) RETURN n")
    assert driver.sessions_opened == 1
    recorded_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_incomplete_commit_on_retryable_write_is_not_replayed(recorded_sleep):
    driver = _ScriptedDriver([{"data": IncompleteCommit("commit outcome unknown")}])
    adapter = _adapter_with(driver)

    with pytest.raises(IncompleteCommit):
        await adapter.add_edges([("a", "b", "r", {"edge_text": "a r b"})])
    assert driver.sessions_opened == 1


@pytest.mark.asyncio
async def test_connection_budget_exhaustion_on_adapter(recorded_sleep):
    errors = [SessionExpired(f"attempt {index}") for index in range(11)]
    driver = _ScriptedDriver([{"run": error} for error in errors])
    adapter = _adapter_with(driver)

    with pytest.raises(SessionExpired) as raised:
        await adapter.get_graph_metadata()
    assert raised.value is errors[-1]
    assert driver.sessions_opened == 11


_ID_FILTERED_EDGES = [
    {
        "aid": "a",
        "bid": "b",
        "type": "r",
        "properties": {"edge_text": "a r b"},
        "aeid": 1,
        "beid": 2,
    },
    {
        "aid": "b",
        "bid": "c",
        "type": "r",
        "properties": {"edge_text": "b r c"},
        "aeid": 2,
        "beid": 3,
    },
]
_ID_FILTERED_NODES = [{"id": node_id, "properties": {"id": node_id}} for node_id in "abc"]


def _id_filtered_rows(query, parameters):
    return _ID_FILTERED_NODES if "element_ids" in (parameters or {}) else _ID_FILTERED_EDGES


@pytest.mark.asyncio
async def test_stream_read_restarts_whole_attempt(recorded_sleep):
    # The first attempt streams one edge row of the first query, then loses the
    # connection mid-stream; the retry must not keep that row.
    driver = _ScriptedDriver(
        [{"stream": ServiceUnavailable("dropped"), "stream_at": 1}],
        responder=_id_filtered_rows,
    )
    adapter = _adapter_with(driver)

    nodes, edges = await adapter._read_id_filtered_rows("EDGES", "NODES", ["a"])

    assert sorted(nodes) == ["a", "b", "c"]
    assert len(edges) == 2
    assert driver.sessions_opened == 2
    assert driver.queries == ["EDGES", "EDGES", "NODES"]


@pytest.mark.asyncio
async def test_existing_edge_type_ids_stream_restarts_whole_attempt(recorded_sleep):
    rows = [
        {"edge_text": "Alice works at Acme", "rel": "works_at"},
        {"edge_text": None, "rel": "lives_in"},
        {"edge_text": "  ", "rel": "knows"},
    ]
    driver = _ScriptedDriver(
        [{"stream": SessionExpired("dropped"), "stream_at": 2}],
        responder=lambda query, parameters: rows,
    )
    adapter = _adapter_with(driver)
    live = {str(EdgeType.id_for(text)) for text in ("Alice works at Acme", "lives_in", "knows")}
    orphan = str(EdgeType.id_for("Bob lives in Oslo"))

    found = await adapter.get_existing_edge_type_ids(live | {orphan})

    assert found == live
    assert driver.sessions_opened == 2


@pytest.mark.asyncio
async def test_existing_edge_type_ids_compares_normalized_ids_not_raw_text(recorded_sleep):
    # "Alice Works" and "alice_works" are one EdgeType id; a raw-string
    # comparison would call the vector of the other spelling an orphan.
    driver = _ScriptedDriver(
        responder=lambda query, parameters: [{"edge_text": "alice_works", "rel": "r"}]
    )
    adapter = _adapter_with(driver)
    candidate = str(EdgeType.id_for("Alice Works"))

    assert await adapter.get_existing_edge_type_ids({candidate}) == {candidate}


@pytest.mark.asyncio
async def test_existing_edge_type_ids_normalizes_the_live_text_too(recorded_sleep):
    # The live edge is spelled "Alice Works", the candidate came from "alice_works":
    # the live side must be normalized by the same id function, not compared raw.
    driver = _ScriptedDriver(
        responder=lambda query, parameters: [{"edge_text": "Alice Works", "rel": "r"}]
    )
    adapter = _adapter_with(driver)
    candidate = str(EdgeType.id_for("alice_works"))

    assert await adapter.get_existing_edge_type_ids({candidate}) == {candidate}


@pytest.mark.asyncio
async def test_existing_edge_type_ids_without_candidates_reads_nothing():
    driver = _ScriptedDriver()
    adapter = _adapter_with(driver)

    assert await adapter.get_existing_edge_type_ids(set()) == set()
    assert driver.sessions_opened == 0


@pytest.mark.asyncio
async def test_primary_error_survives_close_failure(recorded_sleep):
    primary = ClientError("Neo.ClientError.Statement.SyntaxError")
    driver = _ScriptedDriver(
        [{"stream": primary, "stream_at": 0, "tx_close": ServiceUnavailable("close failed")}],
        responder=_id_filtered_rows,
    )
    adapter = _adapter_with(driver)

    with pytest.raises(ClientError) as raised:
        await adapter._read_id_filtered_rows("EDGES", "NODES", ["a"])
    assert raised.value is primary
    assert driver.sessions_opened == 1
    assert "tx.close" in driver.events


@pytest.mark.asyncio
async def test_lone_close_failure_is_retried(recorded_sleep):
    driver = _ScriptedDriver(
        [{"tx_close": SessionExpired("close failed")}], responder=_id_filtered_rows
    )
    adapter = _adapter_with(driver)

    _nodes, edges = await adapter._read_id_filtered_rows("EDGES", "NODES", ["a"])

    assert len(edges) == 2
    assert driver.sessions_opened == 2


@pytest.mark.asyncio
async def test_retry_preserves_other_session_and_driver_config(recorded_sleep, monkeypatch):
    # The failing call and a healthy one interleave: the healthy statement is
    # running when the other attempt fails and retries in a new session.
    healthy_running = asyncio.Event()
    release_healthy = asyncio.Event()

    class _InterleavedDriver(_ScriptedDriver):
        def session(self, database=None):
            session = super().session(database)
            if self.sessions_opened == 1:  # the healthy call's session
                original = session.run

                async def run(query, parameters=None):
                    healthy_running.set()
                    await release_healthy.wait()
                    return await original(query, parameters)

                session.run = run
            return session

    driver = _InterleavedDriver(
        [{}, {"run": ServiceUnavailable("dropped")}],
        responder=lambda query, parameters: [{"key": "k", "value": "v"}],
    )
    adapter = _adapter_with(driver)

    healthy = asyncio.create_task(adapter.get_graph_metadata())
    await healthy_running.wait()
    assert await adapter.get_graph_metadata() == {"k": "v"}
    release_healthy.set()
    assert await healthy == {"k": "v"}

    assert driver.sessions_opened == 3
    driver.close.assert_not_awaited()
    assert adapter.driver is driver

    # Constructor settings stay: lifetime 120, keep_alive, no new pool options.
    created = {}

    def fake_driver(url, **kwargs):
        created.update(kwargs)
        return driver

    monkeypatch.setattr(adapter_module.AsyncGraphDatabase, "driver", fake_driver)
    Neo4jAdapter("bolt://unused", graph_database_allow_anonymous=True)
    assert created["max_connection_lifetime"] == 120
    assert created["keep_alive"] is True
    assert set(created) == {
        "auth",
        "max_connection_lifetime",
        "notifications_min_severity",
        "keep_alive",
    }

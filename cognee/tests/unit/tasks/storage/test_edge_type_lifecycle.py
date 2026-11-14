"""EdgeType lifecycle on the Neo4j + PGVector pair (E1, E2, E4).

The graph and vector doubles are subclasses of the real adapters (so the
lifecycle takes its Neo4j + PGVector branch) holding plain dicts; the advisory
lock is replaced by an asyncio lock per (database, schema) that records every
attempt. Every scenario runs a real entrypoint, not the helper alone.
"""

import asyncio
from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("neo4j")
pytest.importorskip("asyncpg")

import cognee.tasks.storage.edge_type_lifecycle as lifecycle_module
from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
from cognee.infrastructure.databases.provenance import EdgeIdentity, NodeDeletePlanData
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.tasks.storage.edge_type_lifecycle import (
    EDGE_TYPE_COLLECTION,
    EdgeTypeLifecycleError,
    edge_type_lifecycle,
)


def K(text: str) -> str:
    return str(EdgeType.id_for(text))


class FakeNeo4j(Neo4jAdapter):
    """Neo4j adapter whose graph is two dicts; the lifecycle reads go through them."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple[str, str, str], dict] = {}
        self.fail: dict[str, Exception] = {}
        self.calls: list[str] = []

    def _maybe_fail(self, name):
        self.calls.append(name)
        if name in self.fail:
            raise self.fail[name]

    def _text(self, key):
        return get_edge_retrieval_text(self.edges[key].get("edge_text"), key[2])

    async def get_edge_delete_plan_data(self, edges):
        self._maybe_fail("get_edge_delete_plan_data")
        from cognee.infrastructure.databases.provenance import EdgeDeletePlanData

        result = {}
        for edge in edges:
            key = (edge.source_id, edge.target_id, edge.relationship_name)
            if key in self.edges:
                result[edge] = EdgeDeletePlanData(edge, self._text(key), [])
        return result

    async def get_incident_edge_retrieval_texts(self, node_ids):
        self._maybe_fail("get_incident_edge_retrieval_texts")
        ids = set(node_ids)
        return {self._text(key) for key in self.edges if key[0] in ids or key[1] in ids} - {""}

    async def get_existing_edge_type_ids(self, candidate_ids):
        self._maybe_fail("get_existing_edge_type_ids")
        return {K(self._text(key)) for key in self.edges if self._text(key)} & set(candidate_ids)

    async def add_nodes(self, nodes, source_ref_key=None, pipeline_run_id=None):
        self._maybe_fail("add_nodes")
        for node in nodes:
            self.nodes[str(node.id)] = {"type": type(node).__name__}

    async def add_edges(self, edges, source_ref_key=None, pipeline_run_id=None):
        self._maybe_fail("add_edges")
        for source, target, relationship, properties in edges:
            self.edges[(str(source), str(target), relationship)] = dict(properties or {})

    async def delete_nodes(self, node_ids):
        self._maybe_fail("delete_nodes")
        ids = {str(node_id) for node_id in node_ids}
        for node_id in ids:
            self.nodes.pop(node_id, None)
        self.edges = {k: v for k, v in self.edges.items() if k[0] not in ids and k[1] not in ids}

    async def delete_node(self, node_id):
        await self.delete_nodes([node_id])

    async def delete_edge_triples(self, edges):
        self._maybe_fail("delete_edge_triples")
        for edge in edges:
            self.edges.pop((edge.source_id, edge.target_id, edge.relationship_name), None)

    async def get_graph_data(self):
        self._maybe_fail("get_graph_data")
        return list(self.nodes.items()), [(*key, props) for key, props in self.edges.items()]

    async def remove_belongs_to_set_tags(self, tags, node_ids=None):
        return None

    def live_ids(self) -> set[str]:
        return {K(self._text(key)) for key in self.edges if self._text(key)}


class _EmbeddingEngine:
    def __init__(self):
        self.embedded: list[str] = []

    def get_batch_size(self):
        return 10


class FakePGVector(PGVectorAdapter):
    def __init__(self, db_uri="postgresql+asyncpg://test/db", schema=""):
        self.db_uri = db_uri
        self.schema = schema
        self.collections: dict[str, dict[str, str]] = {}
        self.embedding_engine = _EmbeddingEngine()
        self.fail: dict[str, Exception] = {}
        self.deleted: list[tuple[str, list[str]]] = []

    async def has_collection(self, collection_name):
        return collection_name in self.collections

    async def create_vector_index(self, index_name, index_property_name):
        self.collections.setdefault(f"{index_name}_{index_property_name}", {})

    async def index_data_points(self, index_name, index_property_name, data_points):
        if "index_data_points" in self.fail:
            raise self.fail["index_data_points"]
        collection = self.collections.setdefault(f"{index_name}_{index_property_name}", {})
        for point in data_points:
            text = getattr(point, index_property_name)
            self.embedding_engine.embedded.append(text)
            collection[str(point.id)] = text

    async def delete_data_points(self, collection_name, data_point_ids):
        if "delete_data_points" in self.fail:
            raise self.fail["delete_data_points"]
        ids = [str(point_id) for point_id in data_point_ids]
        self.deleted.append((collection_name, ids))
        for point_id in ids:
            self.collections.get(collection_name, {}).pop(point_id, None)

    async def remove_belongs_to_set_tags(self, tags):
        return None

    def edge_type_ids(self) -> set[str]:
        return set(self.collections.get(EDGE_TYPE_COLLECTION, {}))


class FakeLocks:
    """Per (database, schema) asyncio locks standing in for pg advisory locks."""

    def __init__(self):
        self.locks: dict[tuple, asyncio.Lock] = {}
        self.events: list[tuple] = []
        self.attempted = asyncio.Event()
        self.enabled = True
        self.lost = False
        self.release_error: Exception | None = None

    def factory(self, vector_adapter):
        registry = self
        key = (vector_adapter.db_uri, vector_adapter.schema or "public")

        class _Lock:
            async def acquire(self):
                registry.events.append(("attempt", key, asyncio.current_task()))
                registry.attempted.set()
                if registry.enabled:
                    await registry.locks.setdefault(key, asyncio.Lock()).acquire()
                registry.events.append(("acquired", key))

            async def ensure_held(self):
                if registry.lost:
                    raise EdgeTypeLifecycleError("lock connection lost")

            async def release(self):
                if registry.enabled:
                    registry.locks[key].release()
                registry.events.append(("released", key))
                if registry.release_error is not None:
                    raise registry.release_error

        return _Lock()

    def acquired(self) -> int:
        return sum(1 for event in self.events if event[0] == "acquired")


@pytest.fixture
def locks(monkeypatch):
    registry = FakeLocks()
    monkeypatch.setattr(lifecycle_module, "_AdvisoryLock", registry.factory)
    return registry


async def _event_or_done(event: asyncio.Event, task: asyncio.Task) -> None:
    """Wait for ``event``, or for ``task`` to end: a regression fails, never hangs."""
    waiter = asyncio.ensure_future(event.wait())
    await asyncio.wait({waiter, task}, return_when=asyncio.FIRST_COMPLETED)
    waiter.cancel()
    assert event.is_set(), f"{task!r} ended before the event"


def _ids():
    return {name: str(uuid4()) for name in ("alice", "acme", "carol", "other", "xavier", "doc")}


@pytest.fixture
def store():
    """Xavier is deleted by every scenario: one unique text, one shared spelling."""
    graph, vector = FakeNeo4j(), FakePGVector()
    ids = _ids()
    for node_id in ids.values():
        graph.nodes[node_id] = {"type": "Entity"}
    graph.edges = {
        (ids["xavier"], ids["acme"], "works_at"): {"edge_text": "Xavier works at Acme"},
        (ids["xavier"], ids["alice"], "knows"): {"edge_text": "Shared Fact"},
        (ids["carol"], ids["other"], "knows"): {"edge_text": "shared_fact"},
        (ids["alice"], ids["acme"], "works_at"): {"edge_text": "Alice works at Acme"},
    }
    texts = ["Xavier works at Acme", "Shared Fact", "Alice works at Acme", "historic orphan"]
    vector.collections[EDGE_TYPE_COLLECTION] = {K(text): text for text in texts}
    vector.collections["Entity_name"] = {node_id: "name" for node_id in ids.values()}
    return SimpleNamespace(graph=graph, vector=vector, ids=ids)


# --- entrypoints that delete Xavier -----------------------------------------


async def _provenance(store, monkeypatch):
    from cognee.infrastructure.databases.unified.provenance_delete_planner import (
        execute_source_ref_removal,
    )

    xavier = store.ids["xavier"]
    await execute_source_ref_removal(
        store.graph,
        store.vector,
        node_data={xavier: NodeDeletePlanData(xavier, "Entity", ["name"], {}, ["ref"])},
        edge_data={},
        refs_by_node={xavier: ["ref"]},
        refs_by_edge={},
    )


async def _ledger(store, monkeypatch):
    from unittest.mock import AsyncMock
    from uuid import UUID

    module = import_module("cognee.modules.graph.methods.delete_from_graph_and_vector")
    from cognee.modules.graph.models import Node

    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=store.graph))
    monkeypatch.setattr(module, "get_vector_engine_async", AsyncMock(return_value=store.vector))
    monkeypatch.setattr(module, "mark_ledger_nodes_as_deleted", AsyncMock())
    monkeypatch.setattr(module, "mark_ledger_edges_as_deleted", AsyncMock())
    node = Node(slug=UUID(store.ids["xavier"]), type="Entity", indexed_fields=["name"])
    await module.delete_from_graph_and_vector([node], [], [False], [])


async def _legacy_hard(store, monkeypatch):
    from unittest.mock import AsyncMock

    module = import_module("cognee.modules.graph.methods.legacy_delete")

    doc = store.ids["doc"]
    store.graph.edges[(doc, store.ids["alice"], "contains")] = {"edge_text": "doc mentions alice"}
    store.vector.collections[EDGE_TYPE_COLLECTION][K("doc mentions alice")] = "doc mentions alice"

    async def subgraph(document_id):
        return {
            "orphan_entities": [],
            "orphan_types": [],
            "made_from_nodes": [],
            "chunks": [],
            "document": [{"id": doc}],
        }

    async def degree_one(node_type):
        return [{"id": store.ids["xavier"]}] if node_type == "Entity" else []

    store.graph.get_document_subgraph = subgraph
    store.graph.get_degree_one_nodes = degree_one
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=store.graph))
    monkeypatch.setattr(module, "get_vector_engine_async", AsyncMock(return_value=store.vector))
    deleted = await module.legacy_delete(SimpleNamespace(id=doc), "hard")
    # Node ids only: no EdgeType id travels through the node-id vector loop.
    assert deleted.node_ids == {doc, store.ids["xavier"]}
    assert K("doc mentions alice") not in store.vector.edge_type_ids()


async def _conflict_finalize(store, monkeypatch):
    from cognee.tasks.memify.review_conflicts.models import ConflictsToFinalize
    from cognee.tasks.memify.review_conflicts.write_conflicts import _finalize_run

    await _finalize_run(
        store.graph,
        store.vector,
        ConflictsToFinalize(accepted_conflicts=[], dropped_conflict_ids=[store.ids["xavier"]]),
    )


async def _code_doc_sweep(store, monkeypatch):
    from unittest.mock import AsyncMock

    import cognee.infrastructure.databases.vector as vector_module
    from cognee.tasks.code_docs.add_code_doc_edges import _sweep_stale_code_docs

    monkeypatch.setattr(
        vector_module, "get_vector_engine_async", AsyncMock(return_value=store.vector)
    )
    await _sweep_stale_code_docs(
        store.graph,
        {"repo"},
        set(),
        set(),
        {store.ids["xavier"]: {"type": "CodeDocument", "repo": "repo"}},
        [],
    )


async def _context_delete(store, monkeypatch):
    from cognee.tasks.memify.global_context_index.persist import delete_context_index_nodes

    await delete_context_index_nodes(
        SimpleNamespace(graph=store.graph, vector=store.vector), [store.ids["xavier"]]
    )


DELETE_ENTRYPOINTS = {
    "provenance": _provenance,
    "ledger": _ledger,
    "legacy_hard": _legacy_hard,
    "conflict_finalize": _conflict_finalize,
    "code_doc_sweep": _code_doc_sweep,
    "global_context_delete": _context_delete,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", sorted(DELETE_ENTRYPOINTS))
async def test_overwrite_delete_lifecycle_paths(entrypoint, store, locks, monkeypatch):
    await DELETE_ENTRYPOINTS[entrypoint](store, monkeypatch)

    ids = store.vector.edge_type_ids()
    assert store.ids["xavier"] not in store.graph.nodes
    assert K("Xavier works at Acme") not in ids  # unique text of a deleted edge: gone
    assert K("Shared Fact") in ids  # same id as Carol's live "shared_fact": kept
    assert K("Alice works at Acme") in ids  # untouched live fact: kept
    # Not a candidate of this operation: historic orphans are the runbook's job.
    assert K("historic orphan") in ids
    assert (ids - {K("historic orphan")}) <= store.graph.live_ids()  # V ⊆ L(G) for this work
    assert locks.acquired() == 1


class _Person(DataPoint):
    name: str


async def _add_data_points(store, monkeypatch, edges, *, graph_only=False, index_edges=True):
    from unittest.mock import AsyncMock

    module = import_module("cognee.tasks.storage.add_data_points")

    unified = SimpleNamespace(
        graph=store.graph, vector=store.vector, has_capability=lambda capability: False
    )
    monkeypatch.setattr(module, "get_unified_engine", AsyncMock(return_value=unified))
    graph_module = import_module("cognee.infrastructure.databases.graph.get_graph_engine")

    monkeypatch.setattr(graph_module, "get_graph_engine", AsyncMock(return_value=store.graph))
    return await module.add_data_points(
        [_Person(name="p")], custom_edges=edges, graph_only=graph_only, index_edges=index_edges
    )


@pytest.mark.asyncio
async def test_overwrite_replaces_first_text_id(store, locks, monkeypatch):
    alice, acme = store.ids["alice"], store.ids["acme"]
    for text in ("Alice is employed by Acme", "Alice is on Acme's payroll"):
        await _add_data_points(store, monkeypatch, [(alice, acme, "works_at", {"edge_text": text})])
        ids = store.vector.edge_type_ids() - {K("historic orphan")}
        assert ids <= store.graph.live_ids()
        assert K(text) in ids
    assert K("Alice works at Acme") not in store.vector.edge_type_ids()
    assert K("Alice is employed by Acme") not in store.vector.edge_type_ids()

    # A genuinely new fact grows V.
    before = len(store.vector.edge_type_ids())
    carol, other = store.ids["carol"], store.ids["other"]
    await _add_data_points(
        store, monkeypatch, [(carol, other, "works_at", {"edge_text": "Carol works at Other"})]
    )
    assert len(store.vector.edge_type_ids()) == before + 1


@pytest.mark.asyncio
async def test_overwrite_control_without_lifecycle_keeps_first_id(store, locks, monkeypatch):
    """Positive control: the pre-lifecycle path leaves the overwritten text's id behind."""
    monkeypatch.setattr(lifecycle_module, "_is_supported_pair", lambda graph, vector: False)
    alice, acme = store.ids["alice"], store.ids["acme"]
    await _add_data_points(
        store, monkeypatch, [(alice, acme, "works_at", {"edge_text": "Alice is employed by Acme"})]
    )
    assert K("Alice works at Acme") in store.vector.edge_type_ids()
    assert locks.acquired() == 0


@pytest.mark.asyncio
async def test_graph_only_and_index_edges_false(store, locks, monkeypatch):
    from unittest.mock import AsyncMock

    module = import_module("cognee.tasks.storage.add_data_points")

    alice, acme = store.ids["alice"], store.ids["acme"]

    # index_edges=False: no embedding, but the overwritten text is still cleaned up.
    await _add_data_points(
        store,
        monkeypatch,
        [(alice, acme, "works_at", {"edge_text": "Alice is employed by Acme"})],
        index_edges=False,
    )
    assert "Alice is employed by Acme" not in store.vector.embedding_engine.embedded
    assert K("Alice works at Acme") not in store.vector.edge_type_ids()
    assert locks.acquired() == 1

    # graph_only: no unified engine, no vector engine, no lock.
    unified = AsyncMock(side_effect=AssertionError("graph_only initialized a vector engine"))
    monkeypatch.setattr(module, "get_unified_engine", unified)
    vector_before = dict(store.vector.collections[EDGE_TYPE_COLLECTION])
    await module.add_data_points(
        [_Person(name="p")],
        custom_edges=[(alice, acme, "works_at", {"edge_text": "graph only text"})],
        graph_only=True,
    )
    assert store.vector.collections[EDGE_TYPE_COLLECTION] == vector_before
    assert locks.acquired() == 1


@pytest.mark.asyncio
async def test_consolidation_keeps_moved_texts(store, locks, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    module = import_module("cognee.tasks.memify.consolidate_entities")

    xavier, acme, alice = store.ids["xavier"], store.ids["acme"], store.ids["alice"]
    canonical = str(uuid4())
    store.graph.nodes[canonical] = {"type": "Entity"}
    # The canonical already holds the edge the repoint overwrites.
    store.graph.edges[(canonical, acme, "works_at")] = {"edge_text": "Yuri works at Acme"}
    store.vector.collections[EDGE_TYPE_COLLECTION][K("Yuri works at Acme")] = "Yuri"
    # Higher degree than Xavier, so the canonical is picked as the survivor.
    for name in ("carol", "other"):
        store.graph.edges[(canonical, store.ids[name], "likes")] = {"edge_text": "Yuri likes"}
    edges = [(*key, props) for key, props in store.graph.edges.items()]
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=store.graph))
    monkeypatch.setattr(module, "get_vector_engine", MagicMock(return_value=store.vector))

    await module.merge_entity_duplicates(
        {
            "clusters": [
                [
                    {"id": canonical, "name": "Xavier", "props": {}},
                    {"id": xavier, "name": "Xavier", "props": {}},
                ]
            ],
            "edges": edges,
        }
    )

    ids = store.vector.edge_type_ids()
    assert xavier not in store.graph.nodes
    assert K("Xavier works at Acme") in ids  # moved onto the canonical: live
    assert K("Shared Fact") in ids
    assert K("Yuri works at Acme") not in ids  # overwritten by the repoint
    assert (canonical, alice, "knows") in store.graph.edges
    assert locks.acquired() == 1


# --- E2: shared ids, nesting and concurrency ---------------------------------


@pytest.mark.asyncio
async def test_normalized_spellings_share_one_id_until_last_user(store, locks):
    graph, vector = store.graph, store.vector
    carol, other, xavier = store.ids["carol"], store.ids["other"], store.ids["xavier"]

    async with edge_type_lifecycle(graph, vector, node_ids=[xavier]):
        await graph.delete_nodes([xavier])
    assert K("Shared Fact") in vector.edge_type_ids()

    edge = EdgeIdentity(carol, other, "knows")
    async with edge_type_lifecycle(graph, vector, edge_identities=[edge]):
        await graph.delete_edge_triples([edge])
    assert K("Shared Fact") not in vector.edge_type_ids()


@pytest.mark.asyncio
async def test_shared_normalized_text(store, locks, monkeypatch):
    """Two clients interleave: A deletes Xavier's unique edge while B writes a new
    edge with the same text. With the lock the result equals the serial one."""
    final_ids = await _interleave(store, locks, monkeypatch, lock_enabled=True)
    assert K("Xavier works at Acme") in final_ids  # B's live edge keeps its hit


@pytest.mark.asyncio
async def test_shared_normalized_text_control_without_lock(store, locks, monkeypatch):
    """Positive control: without the lock B's write lands between A's live scan and
    A's vector delete, and the hit of a live edge is lost."""
    final_ids = await _interleave(store, locks, monkeypatch, lock_enabled=False)
    assert K("Xavier works at Acme") not in final_ids
    assert K("Xavier works at Acme") in store.graph.live_ids()


async def _interleave(store, locks, monkeypatch, *, lock_enabled):
    locks.enabled = lock_enabled
    graph, vector = store.graph, store.vector
    xavier, alice, acme = store.ids["xavier"], store.ids["alice"], store.ids["acme"]
    a_scanned, a_go = asyncio.Event(), asyncio.Event()
    original_scan = graph.get_existing_edge_type_ids

    async def scan_then_pause(candidate_ids):
        live = await original_scan(candidate_ids)
        if asyncio.current_task() is client_a:
            a_scanned.set()
            await a_go.wait()
        return live

    monkeypatch.setattr(graph, "get_existing_edge_type_ids", scan_then_pause)

    async def a():
        edge = EdgeIdentity(xavier, acme, "works_at")
        async with edge_type_lifecycle(graph, vector, edge_identities=[edge]):
            await graph.delete_edge_triples([edge])

    async def b():
        await _add_data_points(
            store, monkeypatch, [(alice, acme, "mentions", {"edge_text": "Xavier works at Acme"})]
        )

    client_a = asyncio.create_task(a())
    await _event_or_done(a_scanned, client_a)
    locks.attempted.clear()
    client_b = asyncio.create_task(b())
    if lock_enabled:
        await _event_or_done(locks.attempted, client_b)  # B announced its acquire
        assert not client_b.done()
    else:
        await client_b
    a_go.set()
    await asyncio.gather(client_a, client_b)
    return vector.edge_type_ids()


@pytest.mark.asyncio
async def test_nested_task_ownership(store, locks):
    graph, vector = store.graph, store.vector
    outer_entered = asyncio.Event()

    async with edge_type_lifecycle(graph, vector) as outer:
        # Same task, same pair: no second acquire (a second would self-deadlock).
        async with edge_type_lifecycle(graph, vector, node_ids=[store.ids["xavier"]]) as inner:
            assert inner is outer
        assert K("Xavier works at Acme") in outer.candidate_ids

        # A child task inherits the context but not the ownership: it must acquire.
        async def child():
            async with edge_type_lifecycle(graph, vector):
                outer_entered.set()

        locks.attempted.clear()
        child_task = asyncio.create_task(child())
        await _event_or_done(locks.attempted, child_task)  # it asked for the lock
        assert not outer_entered.is_set()
        assert locks.acquired() == 1

    await child_task
    assert outer_entered.is_set()
    assert locks.acquired() == 2


@pytest.mark.asyncio
async def test_other_pair_in_same_task_acquires_its_own_lock(store, locks):
    other_vector = FakePGVector(schema="ds_other")
    async with (
        edge_type_lifecycle(store.graph, store.vector),
        edge_type_lifecycle(store.graph, other_vector),
    ):
        pass
    assert locks.acquired() == 2
    assert {event[1] for event in locks.events} == {
        ("postgresql+asyncpg://test/db", "public"),
        ("postgresql+asyncpg://test/db", "ds_other"),
    }


@pytest.mark.asyncio
async def test_other_backends_take_no_lock(store, locks):
    async with edge_type_lifecycle(SimpleNamespace(), store.vector) as lifecycle:
        assert lifecycle is None
    async with edge_type_lifecycle(store.graph, None) as lifecycle:
        assert lifecycle is None
    async with edge_type_lifecycle(store.graph, SimpleNamespace()) as lifecycle:
        assert lifecycle is None
    assert locks.events == []


# --- E4: failures surface and never delete a live id --------------------------


def _delete_xavier(store):
    async def run():
        async with edge_type_lifecycle(store.graph, store.vector, node_ids=[store.ids["xavier"]]):
            await store.graph.delete_nodes([store.ids["xavier"]])

    return run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "where",
    ["snapshot", "graph_write", "cleanup_scan", "vector_delete", "lock_lost"],
)
async def test_lifecycle_errors_propagate(store, locks, where):
    error = RuntimeError(where)
    if where == "snapshot":
        store.graph.fail["get_incident_edge_retrieval_texts"] = error
    elif where == "graph_write":
        store.graph.fail["delete_nodes"] = error
    elif where == "cleanup_scan":
        store.graph.fail["get_existing_edge_type_ids"] = error
    elif where == "vector_delete":
        store.vector.fail["delete_data_points"] = error
    else:
        locks.lost = True
        error = EdgeTypeLifecycleError

    with pytest.raises(RuntimeError):
        await _delete_xavier(store)()

    assert K("Shared Fact") in store.vector.edge_type_ids()
    assert K("Alice works at Acme") in store.vector.edge_type_ids()
    if where in ("snapshot", "graph_write", "lock_lost", "cleanup_scan"):
        assert store.vector.deleted == []  # nothing deleted without a confirmed scan
    assert [event[0] for event in locks.events][-1] == "released"


@pytest.mark.asyncio
async def test_embedding_error_propagates_without_cleanup(store, locks, monkeypatch):
    store.vector.fail["index_data_points"] = RuntimeError("embedding down")
    alice, acme = store.ids["alice"], store.ids["acme"]

    with pytest.raises(RuntimeError):
        await _add_data_points(
            store, monkeypatch, [(alice, acme, "works_at", {"edge_text": "Alice is employed"})]
        )
    assert store.vector.deleted == []
    assert locks.events[-1][0] == "released"


@pytest.mark.asyncio
async def test_body_error_survives_release_failure(store, locks):
    store.graph.fail["delete_nodes"] = ValueError("graph write failed")
    locks.release_error = OSError("unlock failed")

    with pytest.raises(ValueError, match="graph write failed"):
        await _delete_xavier(store)()


@pytest.mark.asyncio
async def test_release_failure_after_success_is_an_error(store, locks):
    locks.release_error = EdgeTypeLifecycleError("lock was not held at release")

    with pytest.raises(EdgeTypeLifecycleError):
        await _delete_xavier(store)()


@pytest.mark.asyncio
async def test_missing_collection_is_a_noop(store, locks):
    del store.vector.collections[EDGE_TYPE_COLLECTION]
    await _delete_xavier(store)()
    assert store.vector.deleted == []
    assert "get_existing_edge_type_ids" not in store.graph.calls


@pytest.mark.asyncio
async def test_engine_handles_resolve_to_one_owner(store, locks):
    """Handles are new per get_*_engine call; ownership compares the adapters."""

    class Handle:
        def __init__(self, adapter):
            object.__setattr__(self, "_adapter", adapter)

        def _engine(self):
            return self._adapter

        @property
        def __class__(self):
            return self._adapter.__class__

        def __getattr__(self, name):
            return getattr(self._adapter, name)

    async with (
        edge_type_lifecycle(Handle(store.graph), Handle(store.vector)) as outer,
        edge_type_lifecycle(Handle(store.graph), Handle(store.vector)) as inner,
    ):
        assert inner is outer
    assert locks.acquired() == 1


# --- A.3 writers: each runs its edge write and its index inside one lock ------


def _record_write_events(store, locks):
    """Put the graph write and the vector index into the lock's event log, in order."""
    graph_add_edges = store.graph.add_edges
    vector_index = store.vector.index_data_points

    async def add_edges(edges, *args, **kwargs):
        locks.events.append(("add_edges",))
        return await graph_add_edges(edges, *args, **kwargs)

    async def index_data_points(index_name, index_property_name, data_points):
        locks.events.append(("index", index_name))
        return await vector_index(index_name, index_property_name, data_points)

    store.graph.add_edges = add_edges
    store.vector.index_data_points = index_data_points


def _assert_one_locked_write(store, locks, expected_text: str):
    kinds = [event[0] for event in locks.events]
    assert kinds.count("acquired") == 1, "the nested index call must reuse the owner's lock"
    assert kinds.index("acquired") < kinds.index("add_edges") < kinds.index("index")
    assert kinds.index("index") < kinds.index("released"), "the index is written under the lock"
    assert K(expected_text) in store.vector.edge_type_ids()
    assert (store.vector.edge_type_ids() - {K("historic orphan")}) <= store.graph.live_ids()


@pytest.mark.asyncio
async def test_chunk_associations_writer_holds_the_lock_for_write_and_index(
    store, locks, monkeypatch
):
    from unittest.mock import AsyncMock

    module = import_module("cognee.tasks.chunks.create_chunk_associations")
    ids = {"first chunk text": str(uuid4()), "second chunk text": str(uuid4())}
    for node_id in ids.values():
        store.graph.nodes[node_id] = {"type": "DocumentChunk"}

    async def search(collection_name, query_text=None, limit=None, **kwargs):
        if limit == 1:
            return [SimpleNamespace(id=ids[query_text])]
        return [SimpleNamespace(id=node_id) for node_id in ids.values()]

    store.vector.search = search
    similar = module.ChunkSimilarity(
        are_similar=True, similarity_score=0.9, reasoning="same topic", association_type="topical"
    )
    monkeypatch.setattr(module, "get_vector_engine_async", AsyncMock(return_value=store.vector))
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=store.graph))
    monkeypatch.setattr(module, "graph_provenance_write_kwargs", AsyncMock(return_value={}))
    monkeypatch.setattr(module, "_compare_chunks", AsyncMock(return_value=similar))
    _record_write_events(store, locks)

    seen = [chunk async for chunk in module.create_chunk_associations(list(ids))]

    assert seen == list(ids)
    _assert_one_locked_write(store, locks, "associated_with")


@pytest.mark.asyncio
async def test_rule_associations_writer_holds_the_lock_for_write_and_index(
    store, locks, monkeypatch
):
    from unittest.mock import AsyncMock

    module = import_module("cognee.tasks.codingagents.coding_rule_associations")
    origin = str(uuid4())
    store.graph.nodes[origin] = {"type": "DocumentChunk"}

    async def search(collection_name, query_text=None, limit=None, **kwargs):
        return [SimpleNamespace(id=origin)]

    store.vector.search = search
    rules = module.RuleSet(rules=[module.Rule(text="always add tests")])
    store.graph.nodes[str(rules.rules[0].id)] = {"type": "Rule"}
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=store.graph))
    monkeypatch.setattr(module, "get_vector_engine_async", AsyncMock(return_value=store.vector))
    monkeypatch.setattr(module, "get_existing_rules", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        module.LLMGateway, "acreate_structured_output", AsyncMock(return_value=rules)
    )
    monkeypatch.setattr(module, "add_data_points", AsyncMock())
    monkeypatch.setattr(
        module, "graph_provenance_write_kwargs", AsyncMock(return_value={"source_ref_key": None})
    )
    _record_write_events(store, locks)

    await module.add_rule_associations("a chat about tests", "rules_set")

    _assert_one_locked_write(store, locks, "rule_associated_from")


@pytest.mark.asyncio
async def test_cross_connect_writer_holds_the_lock_for_write_and_index(store, locks, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    module = import_module("cognee.tasks.memify.cross_connect_entities")
    alice, carol = store.ids["alice"], store.ids["carol"]

    async def get_neighbors(entity_id):
        return []

    store.graph.get_neighbors = get_neighbors
    relation = module.InferredRelation(
        related=True, relationship_name="colleague_of", confidence=0.9
    )
    monkeypatch.setattr(module, "get_graph_engine", AsyncMock(return_value=store.graph))
    monkeypatch.setattr(module, "get_vector_engine", MagicMock(return_value=store.vector))
    monkeypatch.setattr(
        module, "_vector_candidate_pairs", AsyncMock(return_value={tuple(sorted((alice, carol)))})
    )
    monkeypatch.setattr(module, "_infer_relation", AsyncMock(return_value=relation))
    monkeypatch.setattr(module, "_drop_existing_edges", AsyncMock(side_effect=lambda g, e: e))
    _record_write_events(store, locks)

    result = await module.cross_connect_entities(
        [(alice, {"name": "Alice"}), (carol, {"name": "Carol"})]
    )

    assert result["written"] == 1
    _assert_one_locked_write(store, locks, "colleague_of")

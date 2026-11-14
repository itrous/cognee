"""EdgeType lifecycle on a live Neo4j + PGVector pair (E1-E4, N1, O1).

Real adapters, real SDK write/delete entrypoints, a fixed embedding function
(2048-dimension unit vectors derived from the text, no API) and a local corpus.
The advisory lock, the Neo4j streaming read and the PostgreSQL maintenance
procedure of the runbook run for real; nothing on the lifecycle path is mocked.

Skipped unless ``TEST_NEO4J_URL`` and ``TEST_PGVECTOR_URL`` are set. Run them
against throw-away local containers (never a production store), with
``COGNEE_ENV_FILE`` pointing at an env file that selects Neo4j + PGVector with
``PGVECTOR_HALFVEC=true``, ``PGVECTOR_HNSW_INDEX=true`` and 2048 embedding
dimensions, e.g.::

    COGNEE_ENV_FILE=/abs/path/edgetype-integration.env \\
        uv run pytest cognee/tests/integration/tasks/test_edge_type_lifecycle.py -q

The module drops every table of the test database and clears the test graph
before each test, so it refuses to run against anything but ``127.0.0.1`` or
``localhost``.

No test waits on the wall clock: concurrent steps are ordered by events and by
the peer process's stdout lines.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from edge_type_lifecycle_peer import (  # the second client lives beside this file
    DIMENSIONS,
    EmbedLog,
    install_fixed_embeddings,
)

import cognee

NEO4J_URL = os.getenv("TEST_NEO4J_URL")
NEO4J_USER = os.getenv("TEST_NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("TEST_NEO4J_PASSWORD", "")
PG_URL = os.getenv("TEST_PGVECTOR_URL")  # postgresql+asyncpg://user:pw@host:port/db

pytestmark = [
    pytest.mark.skipif(
        not (NEO4J_URL and PG_URL),
        reason="TEST_NEO4J_URL / TEST_PGVECTOR_URL not set; skipping live lifecycle tests",
    ),
    pytest.mark.asyncio,
]

COLLECTION = "EdgeType_relationship_name"
HNSW_INDEX = f"ix_{COLLECTION}_vector_hnsw"


# --- independent observers of the two stores ---------------------------------


def _pg_dsn() -> str:
    return PG_URL.replace("postgresql+asyncpg://", "postgresql://", 1)


def _require_local_test_stores() -> None:
    from urllib.parse import urlparse

    for url in (NEO4J_URL, PG_URL):
        host = urlparse(url).hostname
        assert host in ("127.0.0.1", "localhost"), f"refusing to clean a non-local store: {host}"


@asynccontextmanager
async def pg():
    import asyncpg

    connection = await asyncpg.connect(_pg_dsn())
    try:
        yield connection
    finally:
        await connection.close()


@asynccontextmanager
async def neo4j_session():
    from neo4j import AsyncGraphDatabase

    async with (
        AsyncGraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver,
        driver.session() as session,
    ):
        yield session


def K(text: str) -> str:
    from cognee.modules.graph.models.EdgeType import EdgeType

    return str(EdgeType.id_for(text))


async def live_ids() -> set[str]:
    """L(G): EdgeType ids of every edge of the physical graph, by the runbook's exporter."""
    from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text

    ids: set[str] = set()
    async with neo4j_session() as session:
        result = await session.run(
            "MATCH (a:__Node__)-[r]->(b:__Node__) RETURN r.edge_text AS text, type(r) AS rel"
        )
        async for row in result:
            text = get_edge_retrieval_text(row["text"], row["rel"])
            if text:
                ids.add(K(text))
        await result.consume()
    return ids


async def vector_ids(collection: str = COLLECTION) -> set[str]:
    """V: ids stored in the collection; empty when it does not exist yet."""
    async with pg() as connection:
        if not await connection.fetchval("SELECT to_regclass($1)", f'public."{collection}"'):
            return set()
        return {row["id"] for row in await connection.fetch(f'SELECT id::text FROM "{collection}"')}


async def vector_rows() -> dict[str, tuple[str, str]]:
    """id -> (payload text, vector text): what must be byte-equivalent across maintenance."""
    async with pg() as connection:
        rows = await connection.fetch(
            f"SELECT id::text AS id, payload::text AS payload, vector::text AS vector "
            f'FROM "{COLLECTION}"'
        )
    return {row["id"]: (row["payload"], row["vector"]) for row in rows}


async def graph_snapshot() -> tuple[set[str], set[tuple[str, str, str, str]]]:
    async with neo4j_session() as session:
        nodes = await (await session.run("MATCH (n:__Node__) RETURN n.id AS id")).data()
        edges = await (
            await session.run(
                "MATCH (a:__Node__)-[r]->(b:__Node__) "
                "RETURN a.id AS s, b.id AS t, type(r) AS rel, r.edge_text AS text"
            )
        ).data()
    return (
        {row["id"] for row in nodes},
        {(row["s"], row["t"], row["rel"], row["text"] or "") for row in edges},
    )


async def other_tables_state() -> dict[str, list]:
    """Content of every table except the EdgeType collection, for 'untouched' checks."""
    state = {}
    async with pg() as connection:
        tables = await connection.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename <> $1 "
            "ORDER BY tablename",
            COLLECTION,
        )
        for table in tables:
            name = table["tablename"]
            state[name] = [
                tuple(row.values())
                for row in await connection.fetch(f'SELECT id::text, payload::text FROM "{name}"')
            ]
    return state


# --- the stack ---------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    """One loop for the module: the Neo4j driver and pools are bound to the loop that made them."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def _wipe() -> None:
    _require_local_test_stores()
    async with neo4j_session() as session:
        await (await session.run("MATCH (n) DETACH DELETE n")).consume()
    async with pg() as connection:
        for row in await connection.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ):
            await connection.execute(f'DROP TABLE IF EXISTS "{row["tablename"]}" CASCADE')


@pytest_asyncio.fixture
async def stack(monkeypatch):
    """Real engines of the configured pair, an empty store, the fixed embedding function."""
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
    from cognee.infrastructure.databases.unified import get_unified_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    log = EmbedLog()
    install_fixed_embeddings(monkeypatch.setattr, log)
    await _wipe()
    graph = await get_graph_engine()
    vector = await get_vector_engine_async()
    # The cached adapter remembers the tables it reflected; the wipe dropped them.
    vector._metadata.clear()
    assert isinstance(graph, Neo4jAdapter), "configure GRAPH_DATABASE_PROVIDER=neo4j"
    assert isinstance(vector, PGVectorAdapter), "configure VECTOR_DB_PROVIDER=pgvector"
    assert vector.embedding_engine.get_vector_size() == DIMENSIONS
    unified = await get_unified_engine()
    yield SimpleNamespace(graph=graph, vector=vector, unified=unified, embedded=log)
    async with pg() as connection:
        leaked = await connection.fetchval(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND pid <> pg_backend_pid()"
        )
    await _wipe()
    assert leaked == 0, "a lifecycle lock outlived its block: no writer could go on"


# --- corpus helpers ----------------------------------------------------------


def entity(name: str, description: str | None = None):
    from cognee.modules.engine.models.Entity import Entity

    return Entity(name=name, description=description or f"{name} is an entity")


def thing(name: str):
    """A node no vector collection indexes: facts between Things have no entity hit."""
    from cognee.infrastructure.engine import DataPoint

    class Thing(DataPoint):
        name: str
        metadata: dict = {"index_fields": [], "identity_fields": ["name"]}

    return Thing(name=name)


def fact(source, target, relationship: str, text: str):
    """A custom edge whose retrieval text is ``text`` (it overwrites on the same triple)."""
    return (str(source.id), str(target.id), relationship, {"edge_text": text})


async def write(nodes, edges, **kwargs):
    from cognee.tasks.storage import add_data_points

    return await add_data_points(nodes, custom_edges=edges, **kwargs)


async def assert_vectors_within_graph(allow_missing: bool = False) -> tuple[set[str], set[str]]:
    """E1's V ⊆ L(G); with ``allow_missing`` False also every live text is indexed."""
    live, stored = await live_ids(), await vector_ids()
    assert stored <= live, f"orphans in the collection: {sorted(stored - live)}"
    if not allow_missing:
        assert live <= stored, f"live texts without a vector: {sorted(live - stored)}"
    return live, stored


# --- E1: growth bound --------------------------------------------------------


async def test_lifecycle_growth_and_context_delete(stack, monkeypatch):
    """E1: paraphrases and deletes do not accumulate ids; new facts do grow V."""
    alice, acme, bob, oslo = entity("Alice"), entity("Acme"), thing("Bob"), thing("Oslo")
    nodes = [alice, acme, bob, oslo]

    await write(nodes, [fact(alice, acme, "works_at", "Alice works at Acme")])
    live, stored = await assert_vectors_within_graph()
    assert stored == {K("Alice works at Acme")}

    # The same triple is rewritten three times: the older formulations must not stay.
    for paraphrase in (
        "Alice is employed by Acme",
        "Acme employs Alice",
        "Alice works at Acme Inc",
    ):
        await write(nodes, [fact(alice, acme, "works_at", paraphrase)])
        _, stored = await assert_vectors_within_graph()
        assert stored == {K(paraphrase)}

    # A genuinely new fact grows the collection; the first one is still the only old one.
    await write(nodes, [fact(bob, oslo, "lives_in", "Bob lives in Oslo")])
    _, stored = await assert_vectors_within_graph()
    assert stored == {K("Alice works at Acme Inc"), K("Bob lives in Oslo")}

    # graph_only: the edge reaches the graph, nothing is embedded or removed.
    embedded_before = len(stack.embedded.texts)
    await write(nodes, [fact(alice, oslo, "visited", "Alice visited Oslo")], graph_only=True)
    assert len(stack.embedded.texts) == embedded_before
    live, stored = await assert_vectors_within_graph(allow_missing=True)
    assert K("Alice visited Oslo") in live - stored
    assert stored == {K("Alice works at Acme Inc"), K("Bob lives in Oslo")}

    # index_edges=False: the overwritten text is cleaned up, the new one is not embedded.
    await write(nodes, [fact(bob, oslo, "lives_in", "Bob resides in Oslo")], index_edges=False)
    assert "Bob resides in Oslo" not in stack.embedded.texts
    live, stored = await assert_vectors_within_graph(allow_missing=True)
    assert K("Bob lives in Oslo") not in stored
    assert stored == {K("Alice works at Acme Inc")}


async def _context_index_corpus(stack):
    """A root and a bucket whose edges carry one unique and one shared text."""
    from cognee.tasks.summarization.models import GlobalContextSummary

    dataset_id = str(uuid4())
    root = GlobalContextSummary(text="root summary", dataset_id=dataset_id, level=1, is_root=True)
    bucket = GlobalContextSummary(text="bucket summary", dataset_id=dataset_id, level=0)
    keeper_a, keeper_b = entity("Keeper A"), entity("Keeper B")
    document = thing("Document")
    nodes = [root, bucket, keeper_a, keeper_b, document]
    await write(
        nodes,
        [
            fact(bucket, document, "summarizes", "bucket covers the quarterly report"),
            fact(bucket, keeper_a, "mentions", "Shared Retrieval Text"),
            fact(root, bucket, "contains", "root groups the bucket"),
            # Outside the bucket and the root, spelled differently, same EdgeType id.
            fact(keeper_a, keeper_b, "mentions", "shared_retrieval_text"),
        ],
    )
    return root, bucket


async def test_lifecycle_growth_and_context_delete_unique_and_shared_hits(stack):
    """E1: deleting a bucket and the root drops the unique hits, keeps the shared one."""
    from cognee.tasks.memify.global_context_index.persist import delete_context_index_nodes

    root, bucket = await _context_index_corpus(stack)
    _, stored = await assert_vectors_within_graph()
    assert {
        K("bucket covers the quarterly report"),
        K("root groups the bucket"),
        K("Shared Retrieval Text"),
    } <= stored

    await delete_context_index_nodes(stack.unified, [str(bucket.id), str(root.id)])

    nodes, edges = await graph_snapshot()
    assert str(bucket.id) not in nodes and str(root.id) not in nodes
    assert {text for *_, text in edges} == {"shared_retrieval_text"}
    _, stored = await assert_vectors_within_graph()
    assert stored == {K("shared_retrieval_text")}  # the same id as "Shared Retrieval Text"


async def test_context_delete_control_without_lifecycle_keeps_unique_hit(stack, monkeypatch):
    """Control: with the lifecycle switched off in this entrypoint alone, the unique hit stays."""
    from cognee.tasks.memify.global_context_index import persist

    @asynccontextmanager
    async def no_lifecycle(*args, **kwargs):
        yield None

    monkeypatch.setattr(persist, "edge_type_lifecycle", no_lifecycle)
    root, bucket = await _context_index_corpus(stack)
    await persist.delete_context_index_nodes(stack.unified, [str(bucket.id), str(root.id)])

    live, stored = await live_ids(), await vector_ids()
    orphans = stored - live
    assert K("bucket covers the quarterly report") in orphans, "the control must leave an orphan"


# --- E2: shared text, serial and concurrent ----------------------------------


@pytest.mark.parametrize(
    ("first_spelling", "second_spelling"),
    [("Alice Works", "alice_works"), ("alice_works", "Alice Works")],
)
async def test_shared_normalized_text_until_last_user(stack, first_spelling, second_spelling):
    """E2: two spellings are one id, whichever of them is live; it goes with the last user."""
    alice, acme, carol, initech = (
        entity("Alice"),
        entity("Acme"),
        entity("Carol"),
        entity("Initech"),
    )
    nodes = [alice, acme, carol, initech]

    await write(
        nodes,
        [
            fact(alice, acme, "works_at", first_spelling),
            fact(carol, initech, "works_at", second_spelling),
        ],
    )
    _, stored = await assert_vectors_within_graph()
    assert stored == {K("Alice Works")} and K("Alice Works") == K("alice_works")
    payload = json.loads((await vector_rows())[K("Alice Works")][0])
    assert payload["text"] == first_spelling, "one batch, one point: the first text passed is kept"

    await write(nodes, [fact(alice, acme, "works_at", "Alice joined Acme")])
    _, stored = await assert_vectors_within_graph()
    assert K("alice_works") in stored, f"{second_spelling!r} still produces the shared id"

    await write(nodes, [fact(carol, initech, "works_at", "Carol joined Initech")])
    _, stored = await assert_vectors_within_graph()
    assert stored == {K("Alice joined Acme"), K("Carol joined Initech")}


async def test_last_user_deletion_through_provenance_removal(stack):
    """E2: deleting edges through the source-ref planner keeps the id until the last one goes."""
    from cognee.infrastructure.databases.provenance import EdgeDeletePlanData, EdgeIdentity
    from cognee.infrastructure.databases.unified.provenance_delete_planner import (
        execute_source_ref_removal,
    )

    alice, acme, carol, initech = (
        entity("Alice"),
        entity("Acme"),
        entity("Carol"),
        entity("Initech"),
    )
    first = fact(alice, acme, "works_at", "Shared Fact")
    second = fact(carol, initech, "works_at", "shared_fact")
    await write([alice, acme, carol, initech], [first, second])
    assert (await vector_ids()) == {K("Shared Fact")}

    async def delete(edge):
        identity = EdgeIdentity(edge[0], edge[1], edge[2])
        await execute_source_ref_removal(
            stack.graph,
            stack.vector,
            node_data={},
            edge_data={identity: EdgeDeletePlanData(identity, edge[3]["edge_text"], ["ref"])},
            refs_by_node={},
            refs_by_edge={identity: ["ref"]},
        )

    await delete(first)
    _, stored = await assert_vectors_within_graph()
    assert stored == {K("Shared Fact")}, "Carol's spelling of the text is still live"

    await delete(second)
    _, stored = await assert_vectors_within_graph()
    assert stored == set()


async def test_lock_key_follows_schema_and_defaults_to_public(stack):
    """E2: the advisory key is (namespace, schema); an empty schema is 'public'."""
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter
    from cognee.tasks.storage.edge_type_lifecycle import LOCK_NAMESPACE, edge_type_lifecycle

    def adapter(schema: str):
        return PGVectorAdapter(
            connection_string=stack.vector.db_uri,
            api_key=None,
            embedding_engine=stack.vector.embedding_engine,
            schema=schema,
        )

    async def try_lock(connection, schema: str) -> bool:
        taken = await connection.fetchval(
            "SELECT pg_try_advisory_lock(hashtext($1), hashtext($2))", LOCK_NAMESPACE, schema
        )
        if taken:
            await connection.fetchval(
                "SELECT pg_advisory_unlock(hashtext($1), hashtext($2))", LOCK_NAMESPACE, schema
            )
        return taken

    async with pg() as probe:
        for held_schema, key in (("", "public"), ("ds_alpha", "ds_alpha")):
            async with edge_type_lifecycle(stack.graph, adapter(held_schema)):
                assert not await try_lock(probe, key), f"{key!r} must be held inside the block"
                other = "ds_beta" if key != "ds_beta" else "ds_alpha"
                assert await try_lock(probe, other), "another schema is a different lock"
                if key == "public":
                    assert await try_lock(probe, "ds_alpha"), "a schema is not the default one"
            assert await try_lock(probe, key), f"{key!r} must be free after the block"


class Peer:
    """A second client in its own process, followed through its ``PEER:`` stdout lines."""

    PATH = Path(__file__).with_name("edge_type_lifecycle_peer.py")

    def __init__(self, process, lines: asyncio.Queue, drain: asyncio.Task, stderr_path: Path):
        self.process, self.lines, self._drain, self.stderr_path = process, lines, drain, stderr_path

    @classmethod
    async def start(cls, job: dict, stderr_path: Path) -> Peer:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(cls.PATH),
            json.dumps(job),
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_path.open("wb"),
        )
        lines: asyncio.Queue = asyncio.Queue()

        async def drain():
            async for raw in process.stdout:
                line = raw.decode(errors="replace").rstrip()
                if line.startswith("PEER:"):
                    await lines.put(line[len("PEER:") :])
            await lines.put(None)  # end of output: the process is gone

        return cls(process, lines, asyncio.create_task(drain()), stderr_path)

    async def next_line(self) -> str:
        line = await self.lines.get()
        assert line is not None, (
            f"the peer ended without the expected line:\n{self.stderr_path.read_text()[-2000:]}"
        )
        assert not line.startswith("ERROR"), line
        return line

    async def finish(self) -> int:
        code = await self.process.wait()
        await self._drain
        return code


_PEER_EDGE = ("Acme", "Alice", "employs", "shared fact")  # the same id as "Shared Fact"


@asynccontextmanager
async def _scan_pause(monkeypatch):
    """Hold the first cleanup between its live scan and its DELETE, until released."""
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    scanned, release = asyncio.Event(), asyncio.Event()
    original = Neo4jAdapter.get_existing_edge_type_ids

    async def paused(self, candidate_ids):
        result = await original(self, candidate_ids)
        scanned.set()
        await release.wait()
        return result

    monkeypatch.setattr(Neo4jAdapter, "get_existing_edge_type_ids", paused)
    yield scanned, release
    release.set()


async def _until(event: asyncio.Event, task: asyncio.Task) -> None:
    """Wait for ``event`` or for ``task`` to end, so a regression fails instead of hanging."""
    waiter = asyncio.ensure_future(event.wait())
    await asyncio.wait({waiter, task}, return_when=asyncio.FIRST_COMPLETED)
    waiter.cancel()
    assert event.is_set(), f"{task!r} ended before the event"


async def _concurrent_shared_text(stack, monkeypatch, tmp_path, *, locked: bool) -> set[str]:
    """Client A rewrites the only edge holding 'Shared Fact' and is held after its live scan;
    client B, another process, then writes an edge whose text has the same id."""
    import cognee.tasks.storage.edge_type_lifecycle as lifecycle

    alice, acme = entity("Alice"), entity("Acme")
    await write([alice, acme], [fact(alice, acme, "works_at", "Shared Fact")])
    assert await vector_ids() == {K("Shared Fact")}

    if not locked:  # the control: no mutual exclusion in this process either

        class NoLock:
            def __init__(self, vector_adapter):
                pass

            async def acquire(self):
                pass

            async def ensure_held(self):
                pass

            async def release(self):
                pass

        monkeypatch.setattr(lifecycle, "_AdvisoryLock", NoLock)

    async with _scan_pause(monkeypatch) as (scanned, release):
        client_a = asyncio.create_task(
            write([alice, acme], [fact(alice, acme, "works_at", "Alice works at Acme")])
        )
        await _until(scanned, client_a)  # A has found "Shared Fact" unused and holds the DELETE

        peer = await Peer.start(
            {"entities": ["Alice", "Acme"], "edge": list(_PEER_EDGE), "no_lock": not locked},
            tmp_path / "peer.stderr",
        )
        assert await peer.next_line() == "ATTEMPT"  # B is about to take (or skip) the lock
        if not locked:
            assert await peer.next_line() == "DONE"  # B finished entirely inside A's gap
        release.set()
        await client_a
        if locked:
            assert await peer.next_line() == "DONE"  # B could only go on after A was done
        assert await peer.finish() == 0
    return await vector_ids()


async def test_two_process_shared_collection_serializes(stack, monkeypatch, tmp_path):
    """E2: with the lock, the interleaved result equals the serial one; the id stays live."""
    stored = await _concurrent_shared_text(stack, monkeypatch, tmp_path, locked=True)
    live = await live_ids()
    assert K("Shared Fact") in live, "B's edge produces the shared id"
    assert stored == live == {K("Alice works at Acme"), K("Shared Fact")}


async def test_two_process_control_without_lock_loses_the_live_hit(stack, monkeypatch, tmp_path):
    """Control: the same interleaving without the lock deletes the id B just made live."""
    stored = await _concurrent_shared_text(stack, monkeypatch, tmp_path, locked=False)
    live = await live_ids()
    assert K("Shared Fact") in live
    assert K("Shared Fact") not in stored, "the control must show the lost hit"


async def test_concurrent_writes_in_one_process_equal_the_serial_result(stack):
    """E2: tasks writing at once, with texts sharing ids, end where a serial run ends."""
    people = [entity(f"Person {n}") for n in range(6)]
    companies = [entity(f"Company {n}") for n in range(6)]
    nodes = people + companies
    # Start: six edges, pairs of them spelled differently but one id each.
    start = [
        fact(people[n], companies[n], "works_at", "Shared Fact" if n % 2 == 0 else "shared_fact")
        for n in range(6)
    ]
    await write(nodes, start)
    assert await vector_ids() == {K("Shared Fact")}

    # Every edge is rewritten at the same time: the odd ones to a text of their own, the
    # even ones all to one text again (three edges, one id).
    final = [
        fact(
            people[n],
            companies[n],
            "works_at",
            f"Person {n} joined Company {n}" if n % 2 else "Even Fact",
        )
        for n in range(6)
    ]
    await asyncio.gather(*[write(nodes, [edge]) for edge in final])

    live, stored = await assert_vectors_within_graph()
    expected = {K("Even Fact")} | {K(f"Person {n} joined Company {n}") for n in (1, 3, 5)}
    assert live == stored == expected


async def test_cancelled_block_releases_the_lock_and_skips_the_cleanup(stack):
    """E4: a cancelled writer frees the lock connection and deletes nothing."""
    from cognee.tasks.storage.edge_type_lifecycle import LOCK_NAMESPACE, edge_type_lifecycle

    alice, acme = entity("Alice"), entity("Acme")
    await write([alice, acme], [fact(alice, acme, "works_at", "Alice works at Acme")])
    entered, hold = asyncio.Event(), asyncio.Event()

    async def writer():
        async with edge_type_lifecycle(
            stack.graph, stack.vector, candidate_texts=["Alice works at Acme"]
        ):
            await stack.graph.add_edges(
                [(str(alice.id), str(acme.id), "works_at", {"edge_text": "Alice joined Acme"})]
            )
            entered.set()
            await hold.wait()

    task = asyncio.create_task(writer())
    await _until(entered, task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async with pg() as probe:
        assert await probe.fetchval(
            "SELECT pg_try_advisory_lock(hashtext($1), hashtext('public'))", LOCK_NAMESPACE
        ), "the cancelled writer must not keep the lock"
        await probe.fetchval(
            "SELECT pg_advisory_unlock(hashtext($1), hashtext('public'))", LOCK_NAMESPACE
        )
    assert K("Alice works at Acme") in await vector_ids(), "a cancelled block cleans nothing up"


# --- the runbook's maintenance procedure (O.2-O.4), run in-process ------------


class MaintenanceFailure(RuntimeError):
    """An error injected at one boundary of the procedure."""


async def export_manifest(fail_after_rows: int | None = None) -> tuple[set[str], int]:
    """O.2: every EdgeType id of the whole graph; the manifest exists only after a full consume."""
    from neo4j import AsyncGraphDatabase

    from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text

    ids: set[str] = set()
    edges = 0
    async with (
        AsyncGraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver,
        driver.session(default_access_mode="READ") as session,
    ):
        result = await session.run(
            "MATCH (a:__Node__)-[r]->(b:__Node__) RETURN r.edge_text AS text, type(r) AS rel"
        )
        async for row in result:
            edges += 1
            if fail_after_rows is not None and edges > fail_after_rows:
                raise MaintenanceFailure("export interrupted before the manifest was complete")
            text = get_edge_retrieval_text(row["text"], row["rel"])
            if text:
                ids.add(K(text))
        await result.consume()
    return ids, edges


async def _load_manifest(connection, manifest: set[str]) -> None:

    await connection.execute(
        "CREATE TEMP TABLE live_edge_types (id uuid PRIMARY KEY) ON COMMIT DROP"
    )
    await connection.copy_records_to_table(
        "live_edge_types", records=[(UUID(i),) for i in sorted(manifest)], columns=["id"]
    )


async def maintenance_preview(manifest: set[str]) -> dict[str, int]:
    """O.3 first block: counts only, always rolled back."""
    async with pg() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await _load_manifest(connection, manifest)
            counts = dict(
                await connection.fetchrow(
                    "SELECT (SELECT count(*) FROM live_edge_types) AS manifest, "
                    "count(*) FILTER (WHERE l.id IS NOT NULL) AS live_present, "
                    "count(*) FILTER (WHERE l.id IS NULL) AS orphan "
                    f'FROM public."{COLLECTION}" v LEFT JOIN live_edge_types l USING(id)'
                )
            )
            counts["live_missing"] = await connection.fetchval(
                "SELECT count(*) FROM live_edge_types l WHERE NOT EXISTS "
                f'(SELECT 1 FROM public."{COLLECTION}" v WHERE v.id = l.id)'
            )
        finally:
            await transaction.rollback()
    return counts


async def hnsw_state() -> dict:
    async with pg() as connection:
        row = await connection.fetchrow(
            "SELECT indisvalid, indisready, pg_get_indexdef(indexrelid) AS definition, "
            "pg_relation_filenode(indexrelid) AS filenode "
            f"FROM pg_index WHERE indexrelid = 'public.\"{HNSW_INDEX}\"'::regclass"
        )
    assert row is not None, f"{HNSW_INDEX} is missing: run with PGVECTOR_HNSW_INDEX=true"
    return dict(row)


async def run_maintenance(
    *,
    skip_delete: bool = False,
    skip_reindex: bool = False,
    fail_before_commit: bool = False,
    fail_after_commit: bool = False,
    fail_export_after: int | None = None,
) -> dict:
    """O.2 -> O.3 (preview, DELETE + COMMIT) -> O.4 (VACUUM, REINDEX, checks).

    Returns the report; ``completed`` is set only when every step succeeded, which is the
    only moment a runbook owner may restart the services. The ``skip_*`` switches are the
    controls (a procedure missing that step); the ``fail_*`` ones inject an error at the
    named boundary.
    """
    manifest, edges = await export_manifest(fail_export_after)
    report = {"graph_edges": edges, "live_ids": len(manifest), "completed": False}
    report["preview"] = await maintenance_preview(manifest)

    if not skip_delete:
        async with pg() as connection:
            transaction = connection.transaction()
            await transaction.start()
            try:
                await _load_manifest(connection, manifest)
                await connection.execute(
                    f'DELETE FROM public."{COLLECTION}" v '
                    "WHERE NOT EXISTS (SELECT 1 FROM live_edge_types l WHERE l.id = v.id)"
                )
                report["remaining_orphans"] = await connection.fetchval(
                    f'SELECT count(*) FROM public."{COLLECTION}" v '
                    "WHERE NOT EXISTS (SELECT 1 FROM live_edge_types l WHERE l.id = v.id)"
                )
                if fail_before_commit:
                    raise MaintenanceFailure("error before COMMIT")
            except BaseException:
                await transaction.rollback()
                raise
            await transaction.commit()
        if fail_after_commit:
            raise MaintenanceFailure("error after COMMIT")
        assert report["remaining_orphans"] == 0

    async with pg() as connection:  # VACUUM and REINDEX cannot run in a transaction block
        await connection.execute(f'VACUUM (ANALYZE) public."{COLLECTION}"')
        report["hnsw_before"] = await hnsw_state()
        if not skip_reindex:
            await connection.execute(f'REINDEX INDEX public."{HNSW_INDEX}"')
        report["hnsw_after"] = await hnsw_state()
        report["rows"] = await connection.fetchval(f'SELECT count(*) FROM public."{COLLECTION}"')
    report["completed"] = True
    return report


async def plant_orphan(stack, text: str) -> str:
    """A vector row no edge produces, as historic garbage: indexed straight into the store."""
    from cognee.modules.graph.models.EdgeType import EdgeType
    from cognee.tasks.storage.index_data_points import index_data_points

    await index_data_points([EdgeType(relationship_name=text, number_of_edges=1)], stack.vector)
    return K(text)


async def _two_live_facts_and_an_orphan(stack):
    alice, acme = entity("Alice"), entity("Acme")
    bob, oslo = thing("Bob"), thing("Oslo")
    await write(
        [alice, acme, bob, oslo],
        [
            fact(alice, acme, "works_at", "Alice works at Acme"),
            fact(bob, oslo, "lives_in", "Bob lives in Oslo"),
        ],
    )
    orphan = await plant_orphan(stack, "Alice worked at Initech long ago")
    assert await vector_ids() == {
        K("Alice works at Acme"),
        K("Bob lives in Oslo"),
        orphan,
    }
    return orphan


# --- O1 ----------------------------------------------------------------------


async def test_orphan_maintenance_preserves_live_rows_and_rebuilds_hnsw(stack):
    """O1: exactly the complement of the manifest goes; live rows are byte-equivalent."""
    orphan = await _two_live_facts_and_an_orphan(stack)
    live_before = {i: row for i, row in (await vector_rows()).items() if i != orphan}
    others_before, graph_before = await other_tables_state(), await graph_snapshot()
    assert others_before, "the corpus must have another vector table to compare"
    embedded_before = len(stack.embedded.texts)

    report = await run_maintenance()

    assert report["completed"] and report["remaining_orphans"] == 0
    assert report["preview"] == {"manifest": 2, "live_present": 2, "orphan": 1, "live_missing": 0}
    assert await vector_ids() == {K("Alice works at Acme"), K("Bob lives in Oslo")}
    assert await vector_rows() == live_before  # payload::text and vector::text unchanged
    assert await other_tables_state() == others_before
    assert await graph_snapshot() == graph_before
    assert len(stack.embedded.texts) == embedded_before, "no live row is embedded again"

    before, after = report["hnsw_before"], report["hnsw_after"]
    assert after["indisvalid"] and after["indisready"]
    assert "halfvec_cosine_ops" in after["definition"]
    assert after["definition"] == before["definition"]
    assert after["filenode"] != before["filenode"], "REINDEX must build a new index file"
    assert report["rows"] == 2


async def test_maintenance_controls_a_procedure_missing_a_step_is_caught(stack):
    """Controls: without the DELETE the orphan stays; without the REINDEX the index file stays."""
    orphan = await _two_live_facts_and_an_orphan(stack)

    report = await run_maintenance(skip_delete=True)
    assert orphan in await vector_ids(), "no DELETE: the orphan must remain"
    assert report["hnsw_after"]["filenode"] != report["hnsw_before"]["filenode"]

    report = await run_maintenance(skip_reindex=True)
    assert orphan not in await vector_ids()
    assert report["hnsw_after"]["filenode"] == report["hnsw_before"]["filenode"], (
        "no REINDEX: the relfilenode must not change"
    )


async def test_maintenance_failure_boundaries(stack):
    """O1: before COMMIT nothing changes; after COMMIT the rows are gone but nothing is 'done'."""
    orphan = await _two_live_facts_and_an_orphan(stack)
    rows_before = await vector_rows()

    # An export that stops midway yields no manifest, so no DELETE is ever issued.
    with pytest.raises(MaintenanceFailure, match="export interrupted"):
        await run_maintenance(fail_export_after=1)
    assert await vector_rows() == rows_before

    with pytest.raises(MaintenanceFailure, match="before COMMIT"):
        await run_maintenance(fail_before_commit=True)
    assert await vector_rows() == rows_before, "the DELETE was rolled back"
    assert orphan in await vector_ids()

    completed = None
    with pytest.raises(MaintenanceFailure, match="after COMMIT"):
        completed = await run_maintenance(fail_after_commit=True)
    assert completed is None, "a failed run must not return a completion report"
    assert await vector_ids() == {K("Alice works at Acme"), K("Bob lives in Oslo")}, (
        "the DELETE is committed: the rows do not come back"
    )


async def test_maintenance_reports_missing_live_vectors_without_repairing_them(stack):
    """O1: a live text without a vector is printed as live_missing; the procedure embeds nothing."""
    orphan = await _two_live_facts_and_an_orphan(stack)
    carol, initech = entity("Carol"), entity("Initech")
    await write(
        [carol, initech],
        [fact(carol, initech, "works_at", "Carol works at Initech")],
        graph_only=True,
    )
    embedded_before = len(stack.embedded.texts)

    report = await run_maintenance()

    assert report["preview"]["live_missing"] == 1
    assert K("Carol works at Initech") not in await vector_ids()
    assert orphan not in await vector_ids()
    assert len(stack.embedded.texts) == embedded_before


# --- E3: retrieval before and after the cleanup ------------------------------


async def _retrieval_corpus(stack):
    """Entities, a standalone fact, same-type facts, a scoped project, a user resolution, orphans."""
    from cognee.modules.engine.models.FactConflict import FactConflict

    alice, acme = entity("Alice"), entity("Acme")
    carol, initech = entity("Carol"), entity("Initech")
    alice.belongs_to_set = ["project_a"]
    acme.belongs_to_set = ["project_a"]
    carol.belongs_to_set = ["project_b"]
    initech.belongs_to_set = ["project_b"]
    bob, oslo, globex = thing("Bob"), thing("Oslo"), thing("Globex")
    old_record = thing("Alice old record")  # the rejected claim sits on no indexed entity
    resolution = FactConflict(
        dataset_id=str(uuid4()),
        about_id=str(alice.id),
        attribute="employer",
        kind="time_varying",
        status="resolved",
        text="Alice's employer was disputed",
        resolution_source="user",
        selected_fact_id=str(uuid4()),
        selected_fact="Alice works at Acme",
        rejected_fact_ids=[str(uuid4())],
        rejected_fact_texts=["Alice works at Globex"],
        resolved_by="user",
    )
    await write(
        [alice, acme, carol, initech, bob, oslo, globex, old_record, resolution],
        [
            fact(alice, acme, "works_at", "Alice works at Acme"),
            fact(carol, initech, "works_at", "Carol works at Initech"),
            fact(bob, oslo, "lives_in", "Bob lives in Oslo"),
            fact(old_record, globex, "works_at", "Alice works at Globex"),
            fact(acme, initech, "competes_with", "Acme Competes With Initech"),
            fact(initech, acme, "competes_with", "acme_competes_with_initech"),
            fact(resolution, alice, "explains", "Alice's employer was disputed"),
        ],
    )
    orphans = [
        await plant_orphan(stack, "Alice worked at Initech long ago"),
        await plant_orphan(stack, "Bob lived in Bergen"),
    ]
    return SimpleNamespace(alice=alice, orphans=orphans)


async def _retrieve(query: str, **options):
    from cognee.modules.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(**options)
    return await retriever.get_retrieved_objects(query=query)


def _projection(objects) -> dict:
    """What a reader sees of a result: ids, names, edge and fact texts."""
    entities = [
        {
            "id": str(item.get("id")),
            "name": item.get("name"),
            "edges": sorted(
                (edge.get("text") or edge.get("relationship_name") or "")
                for edge in item.get("edges", [])
            ),
            "conflicts": sorted(item.get("conflicts", [])),
        }
        for item in objects.get("entities", [])
    ]
    facts = [fact_["text"] for fact_ in objects.get("facts", [])]
    return {"entities": entities, "facts": facts}


async def _graph_completion_edges(query: str) -> set[tuple[str, str, str]]:
    from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever

    edges = await GraphCompletionRetriever(top_k=5).get_retrieved_objects(query=query)
    return {
        (
            str(edge.node1.id),
            str(edge.node2.id),
            str(edge.attributes.get("edge_text") or edge.attributes.get("relationship_name")),
        )
        for edge in edges
    }


async def _live_edge_texts() -> set[str]:
    _, edges = await graph_snapshot()
    return {text for *_, text in edges}


async def test_live_retrieval_before_after_orphan_cleanup(stack):
    """E3: removing only the orphan ids leaves every live answer as it was."""
    corpus = await _retrieval_corpus(stack)
    orphan_texts = {"Alice worked at Initech long ago", "Bob lived in Bergen"}
    queries = {
        "entity": ("Alice", {}),
        "standalone": ("Bob lives in Oslo", {}),
        "scoped": ("Alice works at Acme", {"node_name": ["project_a"]}),
        "other_scope": ("Carol works at Initech", {"node_name": ["project_b"]}),
        "resolution": ("Alice works at Globex", {}),
        "shared_spelling": ("Acme Competes With Initech", {}),
        "orphan": ("Alice worked at Initech long ago", {}),
    }

    async def ask_all():
        retrieved = {
            name: _projection(await _retrieve(query, **options))
            for name, (query, options) in queries.items()
        }
        retrieved["graph_completion"] = await _graph_completion_edges("Alice works at Acme")
        return retrieved

    stored = await vector_ids()
    assert K("Alice works at Acme") in stored and K("Carol works at Initech") in stored
    assert K("works_at") not in stored, "same-type facts must keep their own texts"
    live_texts = await _live_edge_texts()
    before = await ask_all()

    # The orphans are reachable before the cleanup: that is what the cleanup removes.
    assert orphan_texts <= set(before["orphan"]["facts"])
    assert "Bob lives in Oslo" in before["standalone"]["facts"]
    assert not before["standalone"]["entities"] or all(
        "Bob lives in Oslo" not in item["edges"] for item in before["standalone"]["entities"]
    ), "the fact must be reachable only as a standalone fact"
    assert any(edge[2] == "Alice works at Acme" for edge in before["graph_completion"]), (
        "graph completion needs the live EdgeType distance"
    )

    report = await run_maintenance()
    assert report["preview"]["orphan"] == len(orphan_texts), report
    assert report["preview"]["live_missing"] == 0

    after = await ask_all()
    stored_after = await vector_ids()
    assert stored_after == stored - set(corpus.orphans)
    for name in queries:
        entities_before = sorted(before[name]["entities"], key=lambda item: item["id"])
        entities_after = sorted(after[name]["entities"], key=lambda item: item["id"])
        assert entities_after == entities_before, f"entity lane of {name!r} changed"
        live_before = {text for text in before[name]["facts"] if text in live_texts}
        assert live_before <= set(after[name]["facts"]), f"a live fact of {name!r} was lost"
        assert not orphan_texts & set(after[name]["facts"]), f"an orphan still answers {name!r}"
    assert after["graph_completion"] == before["graph_completion"]
    # The scoped searches still show only their own project.
    assert {item["name"] for item in after["scoped"]["entities"]} == {"Alice", "Acme"}
    assert {item["name"] for item in after["other_scope"]["entities"]} == {"Carol", "Initech"}
    # The user's decision still replaces the rejected standalone fact, and still labels Alice.
    for answer in (before, after):
        assert "Alice works at Acme" in answer["resolution"]["facts"]
        assert "Alice works at Globex" not in answer["resolution"]["facts"]
        alice = next(item for item in answer["entity"]["entities"] if item["name"] == "Alice")
        assert any("Alice works at Acme" in text for text in alice["conflicts"])


async def test_retrieval_control_a_lost_live_fact_is_detected(stack):
    """Control: the same projection notices when a live fact's vector row is gone."""
    await _retrieval_corpus(stack)
    before = _projection(await _retrieve("Bob lives in Oslo"))
    assert "Bob lives in Oslo" in before["facts"]

    async with pg() as connection:  # the damage a wrong cleanup would do
        await connection.execute(
            f'DELETE FROM "{COLLECTION}" WHERE id = $1::uuid', K("Bob lives in Oslo")
        )

    after = _projection(await _retrieve("Bob lives in Oslo"))
    assert "Bob lives in Oslo" not in after["facts"], "the control must show the lost fact"


# --- E4: failures are errors, never silent cleanup ---------------------------


async def test_failed_write_reconciles_without_live_reembedding(stack, monkeypatch):
    """E4: the embedding fails after the graph write; the error surfaces, nothing live is deleted,
    and the runbook's reconciliation removes the leftover without embedding anything."""
    alice, acme, bob, oslo = entity("Alice"), entity("Acme"), thing("Bob"), thing("Oslo")
    nodes = [alice, acme, bob, oslo]
    await write(
        nodes,
        [
            fact(alice, acme, "works_at", "Alice works at Acme"),
            fact(bob, oslo, "lives_in", "Bob lives in Oslo"),
        ],
    )
    live_rows = await vector_rows()
    install_fixed_embeddings(
        monkeypatch.setattr, stack.embedded, fail_on={"Alice is employed by Acme"}
    )

    with pytest.raises(RuntimeError, match="Graph edge indexing error"):
        await write(nodes, [fact(alice, acme, "works_at", "Alice is employed by Acme")])

    # Graph before vectors, as always: the edge has its new text, the vector side is behind.
    live, stored = await live_ids(), await vector_ids()
    assert K("Alice is employed by Acme") in live - stored, "live_missing: the new text"
    assert K("Alice works at Acme") in stored - live, "orphan: the old text, not deleted yet"
    assert live_rows[K("Bob lives in Oslo")] == (await vector_rows())[K("Bob lives in Oslo")]

    install_fixed_embeddings(monkeypatch.setattr, stack.embedded)  # the embedder is back
    embedded_before = len(stack.embedded.texts)
    report = await run_maintenance()

    assert report["preview"]["orphan"] == 1 and report["preview"]["live_missing"] == 1
    assert await vector_ids() == {K("Bob lives in Oslo")}
    assert (await vector_rows()) == {K("Bob lives in Oslo"): live_rows[K("Bob lives in Oslo")]}
    assert len(stack.embedded.texts) == embedded_before, "the reconciliation embeds nothing"


async def test_graph_write_failure_deletes_nothing(stack, monkeypatch):
    """E4: when the graph write itself fails, the error surfaces and no vector is touched."""
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    alice, acme = entity("Alice"), entity("Acme")
    await write([alice, acme], [fact(alice, acme, "works_at", "Alice works at Acme")])
    rows = await vector_rows()

    async def broken(self, *args, **kwargs):
        raise RuntimeError("injected graph write failure")

    monkeypatch.setattr(Neo4jAdapter, "add_edges", broken)
    with pytest.raises(RuntimeError, match="injected graph write failure"):
        await write([alice, acme], [fact(alice, acme, "works_at", "Alice is employed by Acme")])

    assert await vector_rows() == rows
    assert await live_ids() == {K("Alice works at Acme")}


async def test_cleanup_failure_is_an_error_and_keeps_the_graph_change(stack, monkeypatch):
    """E4: a failing orphan DELETE is raised, not downgraded to a warning."""
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    alice, acme = entity("Alice"), entity("Acme")
    await write([alice, acme], [fact(alice, acme, "works_at", "Alice works at Acme")])

    async def broken(self, collection_name, data_point_ids):
        raise RuntimeError("injected vector delete failure")

    monkeypatch.setattr(PGVectorAdapter, "delete_data_points", broken)
    with pytest.raises(RuntimeError, match="injected vector delete failure"):
        await write([alice, acme], [fact(alice, acme, "works_at", "Alice is employed by Acme")])

    assert K("Alice works at Acme") in await vector_ids(), "the id stays until the delete is proven"
    assert await live_ids() == {K("Alice is employed by Acme")}


async def test_lost_lock_connection_is_an_error_and_cleanup_does_not_run(stack):
    """E4: the lock connection dies inside the block; leaving the block raises, nothing is deleted."""
    from cognee.tasks.storage.edge_type_lifecycle import edge_type_lifecycle

    alice, acme = entity("Alice"), entity("Acme")
    await write([alice, acme], [fact(alice, acme, "works_at", "Alice works at Acme")])

    with pytest.raises(Exception) as raised:
        async with edge_type_lifecycle(
            stack.graph, stack.vector, candidate_texts=["Alice works at Acme"]
        ):
            await stack.graph.add_edges(
                [(str(alice.id), str(acme.id), "works_at", {"edge_text": "Alice joined Acme"})]
            )
            async with pg() as admin:
                holders = await admin.fetch(
                    "SELECT pid FROM pg_locks WHERE locktype = 'advisory' AND granted "
                    "AND pid <> pg_backend_pid()"
                )
                assert len(holders) == 1, "exactly the lifecycle's own lock connection"
                await admin.execute("SELECT pg_terminate_backend($1)", holders[0]["pid"])

    assert "injected" not in str(raised.value)
    assert K("Alice works at Acme") in await vector_ids(), "cleanup must not run without the lock"


# --- N1 / N2 / N3: retry of a lost connection against the live graph ----------


class _FlakyResult:
    def __init__(self, result, fail):
        self._result, self._fail = result, fail

    async def data(self):
        data = await self._result.data()  # the statement has run by now
        if self._fail:
            raise self._fail()
        return data

    def __getattr__(self, name):
        return getattr(self._result, name)


class _FlakySession:
    def __init__(self, session, point, error_factory):
        self._session, self._point, self._error = session, point, error_factory

    async def run(self, *args, **kwargs):
        if self._point == "run":
            raise self._error()  # the statement never reached the server
        result = await self._session.run(*args, **kwargs)
        return _FlakyResult(result, self._error if self._point == "data" else None)

    def __getattr__(self, name):
        return getattr(self._session, name)


def inject_connection_failure(monkeypatch, point, error_factory, attempts_to_fail=(1,)):
    """Make the given attempts of ``_query_once``-style sessions fail at ``point``.

    ``run`` fails before the statement is sent; ``data`` after the server executed it (the
    write landed, the reply was lost); ``close`` when the session is closed.
    """
    from cognee.infrastructure.databases.graph.neo4j_driver import deadlock_retry
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    original = Neo4jAdapter.get_session
    sleep = AsyncMock()  # the retry's backoff pause: recorded, never waited for
    counter = SimpleNamespace(sessions=0, sleep=sleep)

    @asynccontextmanager
    async def get_session(self):
        counter.sessions += 1
        failing = counter.sessions in attempts_to_fail
        async with original(self) as session:
            yield _FlakySession(session, point if failing else None, error_factory)
            if failing and point == "close":
                raise error_factory()

    monkeypatch.setattr(Neo4jAdapter, "get_session", get_session)
    monkeypatch.setattr(deadlock_retry, "asyncio", SimpleNamespace(sleep=sleep))
    return counter


def record_driver_close(monkeypatch, driver) -> list:
    """Every close() of the shared driver, still performed."""
    closed: list = []
    original = driver.close

    async def close():
        closed.append(True)
        await original()

    monkeypatch.setattr(driver, "close", close)
    return closed


async def _count_edges(source, target, relationship: str) -> int:
    async with neo4j_session() as session:
        rows = await (
            await session.run(
                "MATCH (a:__Node__ {id: $s})-[r]->(b:__Node__ {id: $t}) "
                "WHERE type(r) = $rel RETURN count(r) AS n",
                s=str(source.id),
                t=str(target.id),
                rel=relationship,
            )
        ).data()
    return rows[0]["n"]


@pytest.mark.parametrize("point", ["run", "data", "close"])
@pytest.mark.parametrize("error_name", ["ServiceUnavailable", "SessionExpired"])
async def test_retryable_merge_has_one_edge(stack, monkeypatch, point, error_name):
    """N1: a lost connection in the MERGE is retried in a new session; one edge, one driver."""
    from neo4j import exceptions

    alice, acme = entity("Alice"), entity("Acme")
    await write([alice, acme], [], graph_only=True)
    driver = stack.graph.driver
    closed = record_driver_close(monkeypatch, driver)
    error = getattr(exceptions, error_name)
    counter = inject_connection_failure(monkeypatch, point, lambda: error("injected"))

    await stack.graph.add_edges(
        [(str(alice.id), str(acme.id), "works_at", {"edge_text": "Alice works at Acme"})]
    )

    assert counter.sessions == 2, "the first attempt failed, the second one succeeded"
    assert await _count_edges(alice, acme, "works_at") == 1, "a replayed MERGE adds no second edge"
    assert stack.graph.driver is driver, "the retry must not replace the shared driver"
    assert closed == [], "the retry must not close the shared driver"


@pytest.mark.parametrize("point", ["run", "data", "close"])
async def test_retryable_read_returns_the_complete_result_of_the_new_attempt(
    stack, monkeypatch, point
):
    """N1: a read that fails at any point is repeated whole and returns every row once."""
    from neo4j.exceptions import ServiceUnavailable

    alice, acme, bob = entity("Alice"), entity("Acme"), entity("Bob")
    await write(
        [alice, acme, bob],
        [fact(alice, acme, "works_at", "Alice works at Acme")],
        graph_only=True,
    )
    expected_nodes, expected_edges = await stack.graph.get_graph_data()
    counter = inject_connection_failure(monkeypatch, point, lambda: ServiceUnavailable("injected"))

    nodes, edges = await stack.graph.get_graph_data()

    assert counter.sessions > 2, "two reads of get_graph_data, the first attempt failed"
    assert sorted(str(node[0]) for node in nodes) == sorted(str(node[0]) for node in expected_nodes)
    assert len(edges) == len(expected_edges) == 1


async def test_incomplete_commit_is_not_replayed(stack, monkeypatch):
    """N2: the commit outcome is unknown; the error is raised and the statement runs once."""
    from neo4j.exceptions import IncompleteCommit

    alice, acme = entity("Alice"), entity("Acme")
    await write([alice, acme], [], graph_only=True)
    counter = inject_connection_failure(monkeypatch, "data", lambda: IncompleteCommit("injected"))

    with pytest.raises(IncompleteCommit):
        await stack.graph.add_edges(
            [(str(alice.id), str(acme.id), "works_at", {"edge_text": "Alice works at Acme"})]
        )

    assert counter.sessions == 1
    assert await _count_edges(alice, acme, "works_at") == 1, "the write had landed once"


async def test_retry_budget_exhaustion_keeps_the_original_error(stack, monkeypatch):
    """N2: 11 failed attempts (the first plus 10 retries), then the original error."""
    from neo4j.exceptions import ServiceUnavailable

    alice, acme = entity("Alice"), entity("Acme")
    await write([alice, acme], [], graph_only=True)
    counter = inject_connection_failure(
        monkeypatch,
        "run",
        lambda: ServiceUnavailable("injected forever"),
        attempts_to_fail=range(1, 1000),
    )

    with pytest.raises(ServiceUnavailable, match="injected forever"):
        await stack.graph.add_edges(
            [(str(alice.id), str(acme.id), "works_at", {"edge_text": "Alice works at Acme"})]
        )

    assert counter.sessions == 11
    assert counter.sleep.await_count == 10, "one recorded backoff between consecutive attempts"
    assert await _count_edges(alice, acme, "works_at") == 0


async def test_raw_query_is_not_replayed_after_a_lost_connection(stack, monkeypatch):
    """N2: caller-supplied Cypher is never replayed on a lost connection."""
    from neo4j.exceptions import ServiceUnavailable

    counter = inject_connection_failure(monkeypatch, "run", lambda: ServiceUnavailable("injected"))

    with pytest.raises(ServiceUnavailable):
        await stack.graph.query("MATCH (n) RETURN count(n) AS n")

    assert counter.sessions == 1


# --- cost of the cleanup scan ------------------------------------------------


class _CountingResult:
    def __init__(self, result, counter):
        self._result, self._counter = result, counter

    def __aiter__(self):
        return self._rows()

    async def _rows(self):
        async for record in self._result:
            self._counter.rows += 1
            yield record

    def __getattr__(self, name):
        return getattr(self._result, name)


class _CountingSession:
    def __init__(self, session, counter):
        self._session, self._counter = session, counter

    async def run(self, *args, **kwargs):
        return _CountingResult(await self._session.run(*args, **kwargs), self._counter)

    def __getattr__(self, name):
        return getattr(self._session, name)


def count_streamed_rows(monkeypatch):
    """Count every row a streamed Neo4j read hands to the adapter."""
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    original = Neo4jAdapter.get_session
    counter = SimpleNamespace(rows=0)

    @asynccontextmanager
    async def get_session(self):
        async with original(self) as session:
            yield _CountingSession(session, counter)

    monkeypatch.setattr(Neo4jAdapter, "get_session", get_session)
    return counter


async def test_cleanup_scan_reads_every_live_edge_and_stays_correct(stack, monkeypatch):
    """The scan is one narrow row per live edge: it grows with the graph, the result does not change."""
    alice, acme, bob, oslo = entity("Alice"), entity("Acme"), thing("Bob"), thing("Oslo")
    nodes = [alice, acme, bob, oslo]
    await write(
        nodes,
        [
            fact(alice, acme, "works_at", "Alice works at Acme"),
            fact(bob, oslo, "lives_in", "Bob lives in Oslo"),
        ],
    )
    counter = count_streamed_rows(monkeypatch)

    await write(nodes, [fact(alice, acme, "works_at", "Alice is employed by Acme")])
    small_scan = counter.rows
    assert small_scan >= 2, "the live edges of this corpus"
    await assert_vectors_within_graph()

    # The same corpus plus 7 more live edges, which the scan has to read as well.
    extra = [
        fact(alice, thing(f"Place {n}"), "visited", f"Alice visited Place {n}") for n in range(7)
    ]
    await write(nodes + [thing(f"Place {n}") for n in range(7)], extra, graph_only=True)
    counter.rows = 0

    await write(nodes, [fact(alice, acme, "works_at", "Acme employs Alice")])

    assert counter.rows == small_scan + 7, "one narrow row per additional live edge"
    _, stored = await assert_vectors_within_graph(allow_missing=True)
    assert K("Alice is employed by Acme") not in stored, "the overwritten text is still removed"
    assert {K("Acme employs Alice"), K("Bob lives in Oslo")} <= stored

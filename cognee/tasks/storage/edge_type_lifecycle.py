"""Keep ``EdgeType_relationship_name`` within the retrieval texts of the live graph.

EdgeType vectors are keyed by normalized edge retrieval text (``EdgeType.id_for``),
so one vector serves every edge with that text, in every dataset sharing the
store. A write that overwrites an edge's text, or a delete that removes edges,
can leave a vector no live edge produces any more. ``edge_type_lifecycle`` wraps
such a mutating block on the Neo4j + PGVector pair:

1. take a session-level PostgreSQL advisory lock on the physical collection
   (database + schema) on a dedicated short-lived connection;
2. snapshot the retrieval texts the block may orphan (given edges, every edge
   incident to given nodes, explicit texts) as candidate ids;
3. run the caller's block unchanged (graph before vectors, as before);
4. after the block succeeded, delete the candidate ids no live edge produces.

Every other backend pair yields ``None`` and keeps its previous path. The block
is reentrant for the same task and the same adapter pair only: a nested call adds
its candidates to the owner; a child task or another pair acquires on its own.
"""

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from cognee.infrastructure.databases.provenance import EdgeIdentity
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.shared.logging_utils import get_logger

logger = get_logger("edge_type_lifecycle")

EDGE_TYPE_COLLECTION = "EdgeType_relationship_name"
LOCK_NAMESPACE = "cognee:edge-type-lifecycle:v1"

_LOCK_SQL = "SELECT pg_advisory_lock(hashtext(:namespace), hashtext(:schema))"
_UNLOCK_SQL = "SELECT pg_advisory_unlock(hashtext(:namespace), hashtext(:schema))"


class EdgeTypeLifecycleError(RuntimeError):
    """The lifecycle lock was lost or could not be released as held."""


@dataclass
class EdgeTypeLifecycle:
    """The owner of one acquired lifecycle block."""

    task: asyncio.Task | None
    graph_adapter: Any
    vector_adapter: Any
    graph_engine: Any
    vector_engine: Any
    candidate_ids: set[str] = field(default_factory=set)

    async def add_candidates(
        self,
        edge_identities: Iterable[EdgeIdentity] = (),
        node_ids: Iterable[str] = (),
        candidate_texts: Iterable[str] = (),
    ) -> None:
        """Snapshot what the coming mutation may orphan; call before the mutation."""
        edges = list(edge_identities)
        nodes = [str(node_id) for node_id in node_ids]
        texts = set(candidate_texts)
        if edges:
            plan = await self.graph_engine.get_edge_delete_plan_data(edges)
            texts.update(data.edge_text for data in plan.values())
        if nodes:
            texts.update(await self.graph_engine.get_incident_edge_retrieval_texts(nodes))
        self.candidate_ids.update(str(EdgeType.id_for(text)) for text in texts if text)

    async def remove_orphans(self) -> None:
        """Delete the candidate ids no edge of the current graph produces."""
        if not self.candidate_ids:
            return
        if not await self.vector_engine.has_collection(EDGE_TYPE_COLLECTION):
            return
        live = await self.graph_engine.get_existing_edge_type_ids(set(self.candidate_ids))
        orphan_ids = sorted(self.candidate_ids - live)
        if orphan_ids:
            await self.vector_engine.delete_data_points(EDGE_TYPE_COLLECTION, orphan_ids)
            logger.info("Deleted %d orphaned EdgeType vector(s)", len(orphan_ids))


_owner: ContextVar[EdgeTypeLifecycle | None] = ContextVar("edge_type_lifecycle_owner", default=None)


def _resolve_adapter(engine: Any) -> Any:
    """The real adapter behind an engine handle or a leased cache proxy.

    Handles are created afresh per ``get_*_engine`` call and proxy every
    attribute, so identity is compared on the adapter they resolve to.
    """
    resolve = getattr(type(engine), "_engine", None)
    if resolve is not None:
        engine = resolve(engine)
    return getattr(engine, "__wrapped__", engine)


def _is_supported_pair(graph_engine: Any, vector_engine: Any) -> bool:
    if graph_engine is None or vector_engine is None:
        return False
    try:
        from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter
    except ImportError:  # the neo4j extra is absent, so the graph cannot be Neo4j
        return False
    if not isinstance(graph_engine, Neo4jAdapter):
        return False
    try:
        from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import (
            PGVectorAdapter,
        )
    except ImportError:  # the postgres extra is absent, so the vector store cannot be PGVector
        return False
    return isinstance(vector_engine, PGVectorAdapter)


class _AdvisoryLock:
    """Session-level advisory lock held on its own NullPool connection.

    A waiter must not hold a connection of the vector adapter's working pool, and
    the lock must die with its connection, never outlive a crashed holder.
    """

    def __init__(self, vector_adapter: Any):
        self._db_uri = vector_adapter.db_uri
        self._params = {"namespace": LOCK_NAMESPACE, "schema": vector_adapter.schema or "public"}
        self._engine = None
        self._connection = None

    async def acquire(self) -> None:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        from cognee.infrastructure.databases.relational.config import get_relational_config

        # The same connect args PGVectorAdapter gives its own engines (e.g. SSL).
        connect_args = get_relational_config().database_connect_args
        self._engine = create_async_engine(
            self._db_uri,
            poolclass=NullPool,
            connect_args=dict(connect_args) if connect_args else {},
        )
        try:
            self._connection = await self._engine.connect()
            # Autocommit: the session lock must not keep a transaction open.
            await self._connection.execution_options(isolation_level="AUTOCOMMIT")
            await self._connection.execute(text(_LOCK_SQL), self._params)
        except BaseException:
            await self._close()
            raise

    async def ensure_held(self) -> None:
        """The session lock lives as long as its connection; prove the connection is alive."""
        from sqlalchemy import text

        await self._connection.execute(text("SELECT 1"))

    async def release(self) -> None:
        from sqlalchemy import text

        try:
            result = await self._connection.execute(text(_UNLOCK_SQL), self._params)
            if not result.scalar():
                raise EdgeTypeLifecycleError(
                    "EdgeType lifecycle lock was not held at release; its connection was lost"
                )
        finally:
            await self._close()

    async def _close(self) -> None:
        connection, engine = self._connection, self._engine
        self._connection = self._engine = None
        try:
            if connection is not None:
                await connection.close()
        finally:
            if engine is not None:
                await engine.dispose()


@asynccontextmanager
async def edge_type_lifecycle(
    graph_engine: Any,
    vector_engine: Any,
    *,
    edge_identities: Iterable[EdgeIdentity] = (),
    node_ids: Iterable[str] = (),
    candidate_texts: Iterable[str] = (),
) -> AsyncIterator[EdgeTypeLifecycle | None]:
    """Wrap a mutating block; yields the owner on Neo4j + PGVector, else ``None``.

    Candidates are snapshotted before the block runs. A nested block adds more
    with ``add_candidates`` (or by entering this context again) before its own
    mutation. Orphan cleanup runs only after the outermost block succeeded; any
    failure of the snapshot, the block, the cleanup or the lock is raised.
    """
    if not _is_supported_pair(graph_engine, vector_engine):
        yield None
        return

    graph_adapter = _resolve_adapter(graph_engine)
    vector_adapter = _resolve_adapter(vector_engine)
    task = asyncio.current_task()

    owner = _owner.get()
    if (
        owner is not None
        and owner.task is task
        and owner.graph_adapter is graph_adapter
        and owner.vector_adapter is vector_adapter
    ):
        await owner.add_candidates(edge_identities, node_ids, candidate_texts)
        yield owner
        return

    lock = _AdvisoryLock(vector_adapter)
    await lock.acquire()
    owner = EdgeTypeLifecycle(task, graph_adapter, vector_adapter, graph_engine, vector_engine)
    token = _owner.set(owner)
    try:
        await owner.add_candidates(edge_identities, node_ids, candidate_texts)
        yield owner
        await lock.ensure_held()
        await owner.remove_orphans()
    except BaseException:
        _owner.reset(token)
        try:
            await lock.release()
        except Exception:
            # The original failure is the outcome; the connection is closed either way.
            logger.warning("EdgeType lifecycle lock release failed", exc_info=True)
        raise
    _owner.reset(token)
    await lock.release()

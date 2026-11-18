"""get_superseded_chunk_ids: chunk -> active facts -> dataset-scoped marks -> fact_status.

The graph double answers the filtered read the way the adapters do: the edges
induced by the requested node ids, and the requested nodes.
"""

import json
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.provenance.edge_evidence.lookup import ChunkEdgeSupport
from cognee.modules.retrieval.utils import superseded_chunks

DATASET = uuid4()
OTHER_DATASET = uuid4()


class FakeGraph:
    def __init__(self, edges, conflicts):
        self.edges = edges  # (source, target, relationship, properties)
        self.conflicts = conflicts  # conflict id -> dataset id
        self.reads = []

    async def get_filtered_graph_data(self, filters, *, include_edges=True):
        ids = set(filters[0]["id"])
        self.reads.append((sorted(ids), include_edges))
        nodes = [
            (conflict_id, {"type": "FactConflict", "dataset_id": str(dataset_id)})
            for conflict_id, dataset_id in self.conflicts.items()
            if conflict_id in ids
        ]
        if not include_edges:
            return nodes, []
        return nodes, [edge for edge in self.edges if edge[0] in ids and edge[1] in ids]


class Scenario:
    """One fact per chunk: ``fact(chunk, marks)`` links a new chunk to a new marked edge."""

    def __init__(self):
        self.supports: list[ChunkEdgeSupport] = []
        self.edges: list[tuple] = []
        self.conflicts = {"c-here": DATASET, "c-here2": DATASET, "c-other": OTHER_DATASET}

    def fact(self, marks, *, chunk=None, alias=False, object_id=True):
        chunk = chunk or uuid4()
        source, target, edge_id = uuid4(), uuid4(), uuid4()
        properties = (
            {"conflict_marks_json": json.dumps(marks)} if alias else {"conflict_marks": marks}
        )
        if object_id:
            properties["edge_object_id"] = str(edge_id)
        else:
            edge_id = generate_edge_object_id(str(source), str(target), "has_value")
        self.edges.append((str(source), str(target), "has_value", properties))
        self.supports.append(ChunkEdgeSupport(chunk, edge_id, source, target, "has_value"))
        return str(chunk)

    async def run(self, chunk_ids, graph=None):
        graph = graph or FakeGraph(self.edges, self.conflicts)
        with patch.object(
            superseded_chunks, "get_chunk_edges", AsyncMock(return_value=self.supports)
        ) as lookup:
            result = await superseded_chunks.get_superseded_chunk_ids(graph, DATASET, chunk_ids)
        return result, lookup, graph


def review(status, conflict="c-here"):
    return {"conflict_id": conflict, "status": status}


def user(status, dataset=DATASET, conflict="c-here"):
    return {
        "conflict_id": conflict,
        "status": status,
        "resolution_source": "user",
        "dataset_id": str(dataset),
        "selected_fact": "x",
    }


@pytest.mark.asyncio
async def test_effective_status_decides_which_chunks_are_superseded():
    s = Scenario()
    superseded = s.fact([review("superseded")])
    conflicting = s.fact([review("conflicting")])
    current = s.fact([review("current")])
    # conflicting outranks superseded on one fact (fact_status precedence).
    mixed = s.fact([review("superseded"), review("conflicting", "c-here2")])
    # A mark of another dataset's conflict does not count here.
    foreign = s.fact([review("superseded", "c-other")])
    # A mark whose conflict node is gone has no dataset, so it does not count either.
    dangling = s.fact([review("superseded", "c-missing")])
    # The user's choice outranks the review; the user's rejection is superseded.
    chosen = s.fact([review("superseded"), user("current")])
    rejected = s.fact([user("superseded")])
    neo4j_alias = s.fact([review("superseded")], alias=True)
    derived_id = s.fact([review("superseded")], object_id=False)
    unmarked = s.fact([])
    no_facts = str(uuid4())

    asked = [
        superseded,
        conflicting,
        current,
        mixed,
        foreign,
        dangling,
        chosen,
        rejected,
        neo4j_alias,
        derived_id,
        unmarked,
        no_facts,
    ]
    result, lookup, graph = await s.run(asked)

    assert result == {superseded, rejected, neo4j_alias, derived_id}
    assert lookup.await_args.args[0] == DATASET
    # Two reads: the facts' endpoints with their edges, then the cited conflict nodes.
    assert [include_edges for _, include_edges in graph.reads] == [True, False]
    assert graph.reads[1][0] == ["c-here", "c-here2", "c-missing", "c-other"]


@pytest.mark.asyncio
async def test_a_mixed_chunk_is_excluded_by_any_one_superseded_fact():
    s = Scenario()
    chunk = s.fact([review("current")])
    s.fact([review("superseded")], chunk=uuid4())  # another chunk
    s.fact([review("superseded")], chunk=UUID(chunk))

    result, _, _ = await s.run([chunk])

    assert chunk in result


@pytest.mark.asyncio
async def test_user_decisions_are_scoped_to_the_searched_dataset_not_the_context():
    s = Scenario()
    rejected_elsewhere = s.fact([user("superseded", dataset=OTHER_DATASET, conflict="c-other")])
    chosen_here = s.fact([review("superseded"), user("current")])

    token = current_dataset_id.set(OTHER_DATASET)
    try:
        result, _, _ = await s.run([rejected_elsewhere, chosen_here])
        assert current_dataset_id.get() == OTHER_DATASET
    finally:
        current_dataset_id.reset(token)

    assert result == set()


@pytest.mark.asyncio
async def test_no_supported_edges_reads_no_graph():
    s = Scenario()
    graph = FakeGraph([], {})
    result, _, graph = await s.run([str(uuid4())], graph=graph)

    assert result == set()
    assert graph.reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_read", ["edges", "conflicts"])
async def test_a_failed_graph_read_raises(failing_read):
    s = Scenario()
    chunk = s.fact([review("superseded")])

    class FailingGraph(FakeGraph):
        async def get_filtered_graph_data(self, filters, *, include_edges=True):
            if include_edges == (failing_read == "edges"):
                raise RuntimeError("read failed")
            return await super().get_filtered_graph_data(filters, include_edges=include_edges)

    with pytest.raises(RuntimeError, match="read failed"):
        await s.run([chunk], graph=FailingGraph(s.edges, s.conflicts))


@pytest.mark.asyncio
async def test_a_failed_provenance_read_raises():
    with (
        patch.object(
            superseded_chunks, "get_chunk_edges", AsyncMock(side_effect=RuntimeError("no table"))
        ),
        pytest.raises(RuntimeError, match="no table"),
    ):
        await superseded_chunks.get_superseded_chunk_ids(FakeGraph([], {}), DATASET, [str(uuid4())])

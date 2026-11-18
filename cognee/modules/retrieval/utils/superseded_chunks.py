"""Which chunks support a fact whose effective status in one dataset is ``superseded``.

A fact is a graph edge; the chunks it was extracted from are its active edge
evidence rows. Its status is ``fact_status`` over the conflict marks the edge
carries, counting only marks whose ``FactConflict`` node belongs to the dataset:
a review mark carries no dataset id of its own. Every read propagates its error,
so a caller never mistakes a failed read for a chunk without superseded facts.
"""

from collections.abc import Iterable
from uuid import UUID

from cognee.context_global_variables import current_dataset_id
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.graph.utils.fact_conflicts import fact_status, read_conflict_marks
from cognee.modules.provenance.edge_evidence.lookup import get_chunk_edges


async def get_superseded_chunk_ids(
    graph_engine, dataset_id: UUID, chunk_ids: Iterable[str]
) -> set[str]:
    """The given chunks that support at least one fact superseded in ``dataset_id``."""
    supports = await get_chunk_edges(dataset_id, [UUID(str(chunk_id)) for chunk_id in chunk_ids])
    if not supports:
        return set()

    # The filtered read raises on failure; get_nodes() may return a partial list instead.
    endpoint_ids = {str(s.source_node_id) for s in supports} | {
        str(s.destination_node_id) for s in supports
    }
    _, edges = await graph_engine.get_filtered_graph_data([{"id": sorted(endpoint_ids)}])
    supported_edge_ids = {str(s.edge_id) for s in supports}
    marked_facts: dict[str, list[dict]] = {}
    for source, target, relationship, properties in edges:
        edge_id = str(
            properties.get("edge_object_id")
            or generate_edge_object_id(str(source), str(target), relationship)
        )
        marks = read_conflict_marks(properties)
        if edge_id in supported_edge_ids and marks:
            marked_facts[edge_id] = marks
    if not marked_facts:
        return set()

    conflict_ids = {
        str(mark["conflict_id"])
        for marks in marked_facts.values()
        for mark in marks
        if mark.get("conflict_id")
    }
    if not conflict_ids:
        return set()
    conflicts, _ = await graph_engine.get_filtered_graph_data(
        [{"id": sorted(conflict_ids)}], include_edges=False
    )
    in_scope = {
        str(node_id)
        for node_id, properties in conflicts
        if str(properties.get("dataset_id")) == str(dataset_id)
    }

    # fact_status scopes user decisions by the current dataset; pin it to this one.
    token = current_dataset_id.set(dataset_id)
    try:
        superseded = {
            edge_id
            for edge_id, marks in marked_facts.items()
            if fact_status(
                {
                    "conflict_marks": [
                        mark for mark in marks if str(mark.get("conflict_id")) in in_scope
                    ]
                }
            )
            == "superseded"
        }
    finally:
        current_dataset_id.reset(token)
    return {str(s.chunk_id) for s in supports if str(s.edge_id) in superseded}

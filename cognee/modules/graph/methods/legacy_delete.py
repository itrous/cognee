from uuid import UUID

from cognee.api.v1.exceptions.exceptions import DocumentSubgraphNotFoundError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.infrastructure.engine import DataPoint
from cognee.modules.data.models import Data
from cognee.modules.graph.methods.deleted_graph_elements import DeletedGraphElements
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.graph.utils.prepare_edges_for_storage import get_edge_retrieval_text
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.edge_type_lifecycle import EDGE_TYPE_COLLECTION, edge_type_lifecycle

logger = get_logger()


def _is_contains_edge(edge: dict) -> bool:
    relationship_name = str(edge.get("relationship_name", ""))
    return relationship_name == "contains" or "relationship_name: contains;" in relationship_name


def _get_edge_vector_text(edge: dict) -> str:
    return get_edge_retrieval_text(edge.get("edge_text"), edge.get("relationship_name"))


async def legacy_delete(data: Data, mode: str = "soft"):
    """Delete a single document by its content hash."""

    # Delete from graph database. A fork row's graph node is re-keyed to the
    # canonical id by the rekey_fork_document_ids startup migration; if that
    # migration has not run (ENABLE_AUTO_MIGRATIONS=false without
    # `cognee-cli upgrade`), this raises DocumentSubgraphNotFoundError on
    # purpose — un-migrated stores must not be mutated by guesswork.
    graph_engine = await get_graph_engine()
    vector_engine = await get_vector_engine_async()

    # On Neo4j + PGVector the EdgeType lifecycle snapshots every edge incident to
    # each node before delete_document_subgraph deletes it, and drops the orphaned
    # EdgeType ids after the whole deletion; the id lists below then hold node ids
    # only. Other backends keep collecting contains-edge text ids as before.
    async with edge_type_lifecycle(graph_engine, vector_engine) as lifecycle:
        deleted_node_ids = await delete_document_subgraph(
            data.id, mode, graph_engine=graph_engine, vector_engine=vector_engine
        )
        await _delete_node_vectors(vector_engine, deleted_node_ids, lifecycle is not None)

    return DeletedGraphElements(node_ids={str(node_id) for node_id in deleted_node_ids})


async def _delete_node_vectors(vector_engine, deleted_node_ids, skip_edge_types: bool) -> None:
    """Delete the deleted ids from every vector collection that exists."""
    # Determine vector collections dynamically
    subclasses = get_all_subclasses(DataPoint)
    vector_collections = []

    for subclass in subclasses:
        index_fields = subclass.model_fields["metadata"].default.get("index_fields", [])
        for field_name in index_fields:
            vector_collections.append(f"{subclass.__name__}_{field_name}")

    # If no collections found, use default collections
    if not vector_collections:
        vector_collections = [
            "DocumentChunk_text",
            "EdgeType_relationship_name",
            "EntityType_name",
            "Entity_name",
            "TextDocument_name",
            "TextSummary_text",
        ]

    # Delete records from each vector collection that exists
    for collection in vector_collections:
        if skip_edge_types and collection == EDGE_TYPE_COLLECTION:
            continue
        if await vector_engine.has_collection(collection):
            await vector_engine.delete_data_points(
                collection, [str(node_id) for node_id in deleted_node_ids]
            )


async def delete_document_subgraph(
    document_id: UUID, mode: str = "soft", *, graph_engine=None, vector_engine=None
):
    """Delete a document and all its related nodes in the correct order.

    With a ``vector_engine`` on Neo4j + PGVector, the incident edges of every
    deleted node feed the EdgeType lifecycle (the caller's block when it holds
    one) and only node ids are returned. Without one the deletion stays
    graph-only, as before.
    """
    graph_db = graph_engine or await get_graph_engine()
    subgraph = await graph_db.get_document_subgraph(str(document_id))
    if not subgraph:
        raise DocumentSubgraphNotFoundError(f"Document not found with id: {document_id}")

    # Delete in the correct order to maintain graph integrity
    deletion_order = [
        ("orphan_entities", "orphaned entities"),
        ("orphan_types", "orphaned entity types"),
        (
            "made_from_nodes",
            "made_from nodes",
        ),  # Move before chunks since summaries are connected to chunks
        ("chunks", "document chunks"),
        ("document", "document"),
    ]

    deleted_node_ids = []

    async def delete_nodes(key: str, node_ids: list[str]) -> None:
        # Each group's incident edges are snapshotted right before it is deleted.
        async with edge_type_lifecycle(graph_db, vector_engine, node_ids=node_ids) as lifecycle:
            for node_id in node_ids:
                if key == "chunks" and lifecycle is None:
                    chunk_connections = await graph_db.get_connections(node_id)
                    deleted_node_ids.extend(
                        [
                            str(EdgeType.id_for(_get_edge_vector_text(edge)))
                            for (__, edge, __) in chunk_connections
                            if _is_contains_edge(edge) and _get_edge_vector_text(edge)
                        ]
                    )

                await graph_db.delete_node(node_id)
                deleted_node_ids.append(node_id)

    for key, description in deletion_order:
        nodes = subgraph[key]
        if nodes:
            await delete_nodes(key, [node["id"] for node in nodes])

    # If hard mode, also delete degree-one nodes
    if mode == "hard":
        # Get and delete degree one entity nodes
        degree_one_entity_nodes = await graph_db.get_degree_one_nodes("Entity")
        await delete_nodes("degree_one_entities", [node["id"] for node in degree_one_entity_nodes])

        # Get and delete degree one entity types
        degree_one_entity_types = await graph_db.get_degree_one_nodes("EntityType")
        await delete_nodes("degree_one_types", [node["id"] for node in degree_one_entity_types])

    return deleted_node_ids

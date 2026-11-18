"""CHUNKS filters on real file backends: Ladybug graph, LanceDB vectors, SQLite evidence.

Conflict marks are written by the real review writer, chunks are embedded by the
local fastembed model (no API key; downloaded on first use, like the keyless
e2e suite), and edge evidence lives in SQLite, so the test covers the property
formats and reads the adapters really use.
"""

import importlib
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.markers import mark_graph_provenance_if_empty
from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.unified.unified_store_engine import UnifiedStoreEngine
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.models import Data
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import Entity, FactConflict
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.pipelines.models import PipelineContext, PipelineRun, PipelineRunStatus
from cognee.modules.provenance.edge_evidence import lookup
from cognee.modules.provenance.edge_evidence.models import ProvenanceEdgeEvidence
from cognee.modules.retrieval import chunks_retriever as chunks_module
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.retrieval.utils.superseded_chunks import get_superseded_chunk_ids
from cognee.tasks.memify.review_conflicts import schema, write_conflicts
from cognee.tasks.memify.review_conflicts.accept_review import accept_review_response
from cognee.tasks.memify.review_conflicts.build_review_prompt import ReviewPrompt
from cognee.tasks.memify.review_conflicts.models import (
    AcceptedConflict,
    ConflictsToFinalize,
    ConflictsToWrite,
    ReviewScope,
)
from cognee.tasks.storage.index_data_points import index_data_points
from cognee.tests.integration.tasks.test_review_conflicts import RecordingVectors

TEXTS = {
    "old": "The billing service listens on port 8080.",
    "mixed": "The billing service moved from port 8080 to port 9090.",
    "new": "The billing service now listens on port 9090.",
    "team_a": "Team A owns the billing service.",
    "team_b": "Team B owns the billing service.",
    "foreign": "The billing service runs in the eu region.",
    "plain": "Bananas are yellow.",
    "timing": "The billing test once timed out on port 9090; alone it takes 0.26 to 10.69 s.",
}
QUERY = "Which port does the billing service listen on?"


def fastembed_engine():
    pytest.importorskip("fastembed")
    from cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine import (
        FastembedEmbeddingEngine,
    )

    return FastembedEmbeddingEngine(model="BAAI/bge-small-en-v1.5", dimensions=384)


async def write_review(graph, dataset_id, facts, statuses, *, resolved):
    """Store one conflict over ``facts`` through the real two-phase review write."""
    first, second = facts
    conflict = FactConflict(
        dataset_id=str(dataset_id),
        about_id=first["source"],
        attribute=first["relationship"],
        kind="time_varying" if resolved else "fixed",
        status="resolved" if resolved else "unresolved",
        text=f"{first['relationship']} changed",
        values=[first["target"], second["target"]],
        sources=[source["chunk_id"] for fact in facts for source in fact["sources"]],
    )
    scope = ReviewScope(str(dataset_id), {}, {fact["id"]: fact for fact in facts})
    accepted = AcceptedConflict(conflict, dict(zip([f["id"] for f in facts], statuses)))
    await write_accepted(dataset_id, scope, accepted)


async def write_accepted(dataset_id, scope, accepted):
    ctx = PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
        dataset=SimpleNamespace(id=dataset_id),
        data_item={},
        pipeline_run_id=uuid4(),
    )
    await write_conflicts.write_conflicts(
        [ConflictsToWrite(scope, [accepted], shown_fact_ids=set(scope.facts))], ctx
    )
    await write_conflicts.write_conflicts([ConflictsToFinalize([accepted])], ctx)


@pytest.mark.asyncio
async def test_threshold_and_superseded_filter_on_file_backends(tmp_path, monkeypatch):
    embedding_engine = fastembed_engine()

    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    graph = LadybugAdapter(str(tmp_path / "graph"), kuzu_buffer_pool_size=128 * 1024 * 1024)
    vectors = LanceDBAdapter(str(tmp_path / "lance"), None, embedding_engine)
    writer_engine = UnifiedStoreEngine(
        graph_engine=graph,
        vector_engine=RecordingVectors(),
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )
    search_engine = UnifiedStoreEngine(
        graph_engine=graph,
        vector_engine=vectors,
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )

    async def writer():
        return writer_engine

    async def searcher():
        return search_engine

    monkeypatch.setattr(
        importlib.import_module("cognee.tasks.storage.add_data_points"),
        "get_unified_engine",
        writer,
    )
    monkeypatch.setattr(write_conflicts, "get_unified_engine", writer)
    monkeypatch.setattr(chunks_module, "get_unified_engine", searcher)

    sql_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(sql_engine, expire_on_commit=False)
    async with sql_engine.begin() as connection:
        for table in (Data.__table__, PipelineRun.__table__, ProvenanceEdgeEvidence.__table__):
            await connection.run_sync(lambda sync, table=table: table.create(sync))
    monkeypatch.setattr(
        lookup, "get_relational_engine", lambda: SimpleNamespace(get_async_session=sessions)
    )

    dataset_id, other_dataset_id, run_id = uuid4(), uuid4(), uuid4()
    try:
        await mark_graph_provenance_if_empty(graph)
        document = Document(
            name="Billing notes", raw_data_location="test", external_metadata=None, mime_type="t"
        )
        chunks = {
            name: DocumentChunk(
                text=text,
                chunk_size=len(text.split()),
                chunk_index=index,
                cut_type="test",
                is_part_of=document,
            )
            for index, (name, text) in enumerate(TEXTS.items())
        }
        ids = {name: str(chunk.id) for name, chunk in chunks.items()}
        service = Entity(name="billing service", description="service")
        values = {
            name: Entity(name=name, description=name)
            for name in ("8080", "9090", "team a", "team b", "eu", "us", "timeout", "10.69 s")
        }
        owner = make_source_ref_key(dataset_id, document.id)
        await graph.add_nodes([service, *values.values(), *chunks.values()], source_ref_key=owner)

        def fact(relationship, value, chunk_names, date=None):
            source, target = str(service.id), str(values[value].id)
            fid = str(generate_edge_object_id(source, target, relationship))
            props = {"edge_text": f"billing service {relationship} {value}", "edge_object_id": fid}
            return {
                "id": fid,
                "source": source,
                "target": target,
                "relationship": relationship,
                "properties": props,
                "effective_date": date,
                "observed_at": None,
                "sources": [
                    {
                        "chunk_id": ids[name],
                        "data_id": str(document.id),
                        "document": document.name,
                        "effective_date": date,
                    }
                    for name in chunk_names
                ],
            }

        port_old = fact("has_port", "8080", ["old", "mixed"])
        port_new = fact("has_port", "9090", ["new", "mixed"])
        owner_a = fact("owned_by", "team a", ["team_a"])
        owner_b = fact("owned_by", "team b", ["team_b"])
        region_eu = fact("runs_in", "eu", ["foreign"])
        region_us = fact("runs_in", "us", ["new"])
        # S04: one document states both facts under one date.
        timed_out = fact("failed_with", "timeout", ["timing"], "2026-11-05T14:03:51+00:00")
        duration = fact("has_duration", "10.69 s", ["timing"], "2026-11-05T14:03:51+00:00")
        all_facts = [
            port_old,
            port_new,
            owner_a,
            owner_b,
            region_eu,
            region_us,
            timed_out,
            duration,
        ]
        await graph.add_edges(
            [(f["source"], f["target"], f["relationship"], f["properties"]) for f in all_facts],
            source_ref_key=owner,
        )
        await write_review(
            graph, dataset_id, [port_old, port_new], ["superseded", "current"], resolved=True
        )
        await write_review(
            graph, dataset_id, [owner_a, owner_b], ["conflicting", "conflicting"], resolved=False
        )
        # The review orders S04's two facts; acceptance leaves them conflicting.
        s04_scope = ReviewScope(
            str(dataset_id), {}, {timed_out["id"]: timed_out, duration["id"]: duration}
        )
        s04_prompt = ReviewPrompt(
            text="",
            shown_facts={str(service.id): [timed_out["id"], duration["id"]]},
            node_labels={"n1": str(service.id)},
            fact_labels={"r1": timed_out["id"], "r2": duration["id"]},
            conflict_labels={},
        )
        s04_answer = schema.ReviewResponse(
            reviewed=["n1"],
            conflicts=[
                schema.ReportedConflict(
                    about="n1",
                    attribute="failure status",
                    kind="time_varying",
                    text="The timeout gave way to a measured duration.",
                    current=["r2"],
                    superseded=["r1"],
                )
            ],
        )
        [s04] = accept_review_response(s04_answer, s04_scope, s04_prompt).accepted_conflicts
        assert set(s04.fact_statuses.values()) == {"conflicting"}
        await write_accepted(dataset_id, s04_scope, s04)
        # Another dataset's review of facts this dataset's chunks also support.
        await write_review(
            graph,
            other_dataset_id,
            [region_eu, region_us],
            ["superseded", "current"],
            resolved=True,
        )

        async with sessions() as session:
            session.add(Data(id=document.id, dataset_id=dataset_id, name=document.name))
            session.add(
                PipelineRun(
                    pipeline_run_id=run_id, status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED
                )
            )
            session.add_all(
                ProvenanceEdgeEvidence(
                    user_id=uuid4(),
                    dataset_id=dataset_id,
                    data_id=document.id,
                    pipeline_run_id=run_id,
                    chunk_id=UUID(source["chunk_id"]),
                    edge_id=UUID(f["id"]),
                    source_node_id=UUID(f["source"]),
                    destination_node_id=UUID(f["target"]),
                    relationship_name=f["relationship"],
                )
                for f in all_facts
                for source in f["sources"]
            )
            await session.commit()
        await index_data_points(list(chunks.values()), vector_engine=vectors)

        # The status read alone: only the chunks behind this dataset's superseded fact.
        superseded = await get_superseded_chunk_ids(graph, dataset_id, list(ids.values()))
        assert superseded == {ids["old"], ids["mixed"]}

        async def retrieve(**kwargs):
            found = await ChunksRetriever(**kwargs).get_retrieved_objects(QUERY)
            return [str(chunk.id) for chunk in found], [chunk.score for chunk in found]

        everything, scores = await retrieve(top_k=len(TEXTS))
        assert set(everything) == set(ids.values())
        # Control: the stale chunks are among the nearest without the filter.
        top3, _ = await retrieve(top_k=3)
        assert {ids["old"], ids["mixed"]} & set(top3)

        searched_limits = []
        original_search = vectors.search

        async def spy(*args, **kwargs):
            searched_limits.append(kwargs.get("limit"))
            return await original_search(*args, **kwargs)

        monkeypatch.setattr(vectors, "search", spy)
        filtered, _ = await retrieve(top_k=3, exclude_superseded=True, dataset_id=dataset_id)
        expected = [i for i in everything if i not in superseded][:3]
        assert filtered == expected
        # S04's chunk, with both its facts, stays in the filtered results.
        kept, _ = await retrieve(top_k=len(TEXTS), exclude_superseded=True, dataset_id=dataset_id)
        assert ids["timing"] in kept
        assert searched_limits[0] == 3 and len(searched_limits) > 1

        # Threshold: the boundary distance passes, farther ones do not.
        threshold = 1 - scores[2]
        thresholded, kept_scores = await retrieve(top_k=len(TEXTS), min_similarity=threshold)
        assert thresholded == [i for i, s in zip(everything, scores) if 1 - s >= threshold]
        assert len(thresholded) >= 3 and all(1 - s >= threshold for s in kept_scores)
        assert len(thresholded) < len(everything)

        # A failed provenance read fails the search instead of returning unchecked text.
        broken = async_sessionmaker(create_async_engine("sqlite+aiosqlite:///:memory:"))
        monkeypatch.setattr(
            lookup, "get_relational_engine", lambda: SimpleNamespace(get_async_session=broken)
        )
        with pytest.raises(Exception, match="no such table"):
            await retrieve(top_k=3, exclude_superseded=True, dataset_id=dataset_id)
    finally:
        await graph.close()
        await sql_engine.dispose()

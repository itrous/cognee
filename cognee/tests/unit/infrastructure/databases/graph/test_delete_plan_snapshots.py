"""Delete planning must not materialize graph payloads or lose shared ownership."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.provenance.delete_data import (
    EdgeDeleteData,
    EdgeDeletePlanData,
    EdgeIdentity,
    NodeDeleteData,
    NodeDeletePlanData,
)


@pytest.fixture
def adapter():
    pytest.importorskip("neo4j")
    from cognee.infrastructure.databases.graph.neo4j_driver.adapter import Neo4jAdapter

    instance = Neo4jAdapter("bolt://unused", graph_database_allow_anonymous=True, driver=object())
    instance.query = AsyncMock(return_value=[])
    return instance


def test_plan_models_do_not_carry_full_payload_or_run_history():
    assert {field.name for field in fields(NodeDeletePlanData)} == {
        "node_id",
        "node_type",
        "indexed_fields",
        "node_properties",
        "source_ref_keys",
    }
    assert {field.name for field in fields(EdgeDeletePlanData)} == {
        "edge",
        "edge_text",
        "source_ref_keys",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, expected",
    [
        ("get_node_delete_plan_data", {}),
        ("get_edge_delete_plan_data", {}),
        ("get_existing_edge_texts", set()),
    ],
)
async def test_neo4j_empty_plan_reads_do_not_query(adapter, method, expected):
    assert await getattr(adapter, method)([]) == expected
    adapter.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_neo4j_node_plan_projects_only_planning_fields_and_all_refs(adapter):
    refs = [f"owner-{index}" for index in range(300)] + ["other-dataset/shared-owner"]
    adapter.query.return_value = [
        {
            "id": "node-a",
            "node_type": "NodeSet",
            "metadata": '{"index_fields": ["name", "description"]}',
            "name": "shared set",
            "srk": refs,
        }
    ]

    result = await adapter.get_node_delete_plan_data(["node-a", "missing"])

    assert result == {
        "node-a": NodeDeletePlanData(
            node_id="node-a",
            node_type="NodeSet",
            indexed_fields=["name", "description"],
            node_properties={"name": "shared set"},
            source_ref_keys=refs,
        )
    }
    query, parameters = adapter.query.await_args.args
    assert parameters == {"ids": ["node-a", "missing"]}
    for field in ("n.id", "n.type", "n.metadata", "n.name", "n.source_ref_keys"):
        assert field in query
    for forbidden in ("properties(", "source_run_", "source_dataset_ids", "n.description"):
        assert forbidden not in query
    assert "LIMIT" not in query.upper()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata, expected", [(None, []), ("not-json", []), ({"index_fields": ["name"]}, ["name"])]
)
async def test_neo4j_node_plan_handles_stored_metadata_shapes(adapter, metadata, expected):
    adapter.query.return_value = [
        {"id": "a", "node_type": None, "metadata": metadata, "name": None, "srk": []}
    ]
    snapshot = (await adapter.get_node_delete_plan_data(["a"]))["a"]
    assert snapshot.node_type == ""
    assert snapshot.indexed_fields == expected
    assert snapshot.source_ref_keys == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edge_text, relationship, expected",
    [
        ("  precise text \n", "knows", "precise text"),
        (" \t", " knows ", "knows"),
        (None, "knows", "knows"),
        (None, "  ", ""),
    ],
)
async def test_neo4j_edge_plan_preserves_all_refs_and_retrieval_text(
    adapter, edge_text, relationship, expected
):
    edge = EdgeIdentity("a", "b", relationship)
    refs = [f"owner-{index}" for index in range(300)] + ["shared-owner"]
    adapter.query.return_value = [
        {"s": "a", "t": "b", "rel": relationship, "edge_text": edge_text, "srk": refs}
    ]

    assert await adapter.get_edge_delete_plan_data([edge]) == {
        edge: EdgeDeletePlanData(edge=edge, edge_text=expected, source_ref_keys=refs)
    }
    query, parameters = adapter.query.await_args.args
    assert parameters == {"edges": [{"s": "a", "t": "b", "rel": relationship}]}
    assert "r.edge_text" in query
    assert "r.source_ref_keys" in query
    for forbidden in ("properties(", "source_run_", "source_dataset_ids"):
        assert forbidden not in query
    assert "LIMIT" not in query.upper()


@pytest.mark.asyncio
async def test_neo4j_existing_text_lookup_is_targeted_and_normalizes_fallback(adapter):
    adapter.query.return_value = [
        {"edge_text": " \tprecise text\n", "rel": "ignored"},
        {"edge_text": "\u00a0\u2003", "rel": " knows \n"},
        {"edge_text": "not targeted", "rel": "ignored"},
        {"edge_text": None, "rel": "absent from candidates"},
    ]
    adapter.get_graph_data = AsyncMock(side_effect=AssertionError("must not fetch graph payloads"))
    candidates = ["precise text", "knows", "absent"]

    assert await adapter.get_existing_edge_texts(candidates) == {"precise text", "knows"}

    query, parameters = adapter.query.await_args.args
    assert parameters["texts"] == candidates
    assert all(char in parameters["strip_chars"] for char in " \t\n\r\u00a0\u2003\x1c")
    assert "WHERE" in query.upper()
    assert " IN " in query.upper()
    assert "trim(" in query.lower()
    assert "r.edge_text" in query
    assert "type(r)" in query
    assert "properties(" not in query
    assert "source_run_" not in query
    adapter.get_graph_data.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_value, expected", [(False, "False"), (0, "0"), ([], "[]")])
async def test_neo4j_existing_text_lookup_uses_python_legacy_value_conversion(
    adapter, legacy_value, expected
):
    adapter.query.return_value = [
        {"edge_text": legacy_value, "rel": "fallback-must-not-win"},
        {"edge_text": "unrelated", "rel": expected},
    ]
    assert await adapter.get_existing_edge_texts([expected]) == {expected}


@pytest.mark.asyncio
async def test_full_neo4j_snapshot_apis_still_include_payloads_and_run_history(adapter):
    edge = EdgeIdentity("a", "b", "knows")
    refs, datasets, runs, run_refs = ["owner", "shared"], ["dataset"], ["run"], ["run/owner"]
    provenance = {"srk": refs, "sdi": datasets, "sri": runs, "srr": run_refs}
    adapter.query.side_effect = [
        [
            {
                "id": "a",
                "properties": {"type": "Entity", "name": "A", "description": "large"},
                **provenance,
            }
        ],
        [
            {
                "s": "a",
                "t": "b",
                "rel": "knows",
                "properties": {"edge_text": " A knows B ", "weight": 0.7},
                **provenance,
            }
        ],
    ]

    nodes = await adapter.get_node_delete_data(["a"])
    edges = await adapter.get_edge_delete_data([edge])

    assert nodes["a"].node_properties["description"] == "large"
    assert edges[edge].edge_properties == {"edge_text": " A knows B ", "weight": 0.7}
    assert edges[edge].edge_text == "A knows B"
    for snapshot in (nodes["a"], edges[edge]):
        assert snapshot.source_ref_keys == refs
        assert snapshot.source_dataset_ids == datasets
        assert snapshot.source_run_ids == runs
        assert snapshot.source_run_refs == run_refs
    for call in adapter.query.await_args_list:
        assert "properties(" in call.args[0]
        assert "source_run_refs" in call.args[0]


@pytest.mark.asyncio
async def test_generic_plan_defaults_derive_from_full_snapshots_without_losing_refs():
    refs = [f"owner-{index}" for index in range(300)] + ["shared-owner"]
    edge = EdgeIdentity("a", "b", "knows")
    node = NodeDeleteData(
        "a",
        "NodeSet",
        ["name"],
        {"name": "shared set", "description": "large"},
        refs,
        ["dataset"],
        ["run"],
        ["run/owner"],
    )
    relation = EdgeDeleteData(
        edge, "A knows B", {"weight": 0.7}, refs, ["dataset"], ["run"], ["run/owner"]
    )
    graph = SimpleNamespace(
        get_node_delete_data=AsyncMock(return_value={"a": node}),
        get_edge_delete_data=AsyncMock(return_value={edge: relation}),
    )

    nodes = await GraphDBInterface.get_node_delete_plan_data(graph, ["a", "missing"])
    edges = await GraphDBInterface.get_edge_delete_plan_data(graph, [edge])

    assert isinstance(nodes["a"], NodeDeletePlanData)
    assert nodes["a"].node_id == "a"
    assert nodes["a"].node_type == "NodeSet"
    assert nodes["a"].indexed_fields == ["name"]
    assert nodes["a"].node_properties.get("name") == "shared set"
    assert nodes["a"].source_ref_keys == refs
    assert edges == {edge: EdgeDeletePlanData(edge, "A knows B", refs)}
    graph.get_node_delete_data.assert_awaited_once_with(["a", "missing"])
    graph.get_edge_delete_data.assert_awaited_once_with([edge])
    assert node.node_properties["description"] == "large"
    assert relation.edge_properties == {"weight": 0.7}


@pytest.mark.asyncio
async def test_generic_existing_texts_matches_normalized_text_not_raw_or_substring():
    graph = SimpleNamespace(
        get_graph_data=AsyncMock(
            return_value=(
                [],
                [
                    ("a", "b", "ignored", {"edge_text": " \tprecise text\n"}),
                    ("b", "c", " knows \n", {"edge_text": " \t"}),
                    ("c", "d", "fallback", {}),
                    ("d", "e", "fallback", {"edge_text": None}),
                    ("e", "f", "not-targeted", {"edge_text": "precise text longer"}),
                    ("f", "g", "unrelated", {"edge_text": "unrelated text"}),
                    ("g", "h", " \t", {"edge_text": " "}),
                ],
            )
        )
    )
    candidates = ["precise text", "knows", "fallback", "absent", "precise", ""]

    result = await GraphDBInterface.get_existing_edge_texts(graph, candidates)

    assert result == {"precise text", "knows", "fallback"}
    graph.get_graph_data.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_generic_existing_texts_empty_candidates_does_not_scan_graph():
    graph = SimpleNamespace(get_graph_data=AsyncMock(side_effect=AssertionError("no scan needed")))
    assert await GraphDBInterface.get_existing_edge_texts(graph, []) == set()
    graph.get_graph_data.assert_not_awaited()

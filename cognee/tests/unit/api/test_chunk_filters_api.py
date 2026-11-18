"""POST /api/v1/search and /api/v1/recall: the CHUNKS filters minSimilarity / excludeSuperseded.

The app uses the real exception handlers of cognee/api/client.py, so the status
codes are the deployed ones: 400 when a field has the wrong JSON type (the
request model), 422 for a number outside 0..1, NaN or infinity, and for a filter
sent with a search type it does not apply to (chunk_filter_config in the SDK).
"""

import importlib
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from cognee.exceptions import CogneeApiError

search_router_module = importlib.import_module("cognee.api.v1.search.routers.get_search_router")
recall_router_module = importlib.import_module("cognee.api.v1.recall.routers.get_recall_router")

RESULTS = {
    "/api/v1/search": [{"search_result": [{"id": "c1", "text": "t", "score": 0.2}]}],
    "/api/v1/recall": [
        {
            "source": "graph",
            "kind": "chunk",
            "search_type": "CHUNKS",
            "text": "t",
            "system_prompt": None,
            "score": 0.2,
            "dataset_id": None,
            "dataset_name": None,
            "metadata": {"chunk_id": "c1"},
            "raw": {},
            "structured": None,
        }
    ],
}


@pytest.fixture
def client(monkeypatch):
    from cognee.api.client import exception_handler, request_validation_exception_handler

    app = FastAPI()
    app.include_router(search_router_module.get_search_router(), prefix="/api/v1/search")
    app.include_router(recall_router_module.get_recall_router(), prefix="/api/v1/recall")
    app.add_exception_handler(CogneeApiError, exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    for module in (search_router_module, recall_router_module):
        app.dependency_overrides[module.get_authenticated_user] = lambda: user
        monkeypatch.setattr(module, "send_telemetry", lambda *a, **k: None)
    monkeypatch.setattr("cognee.api.client.send_api_exception_telemetry", lambda *a, **k: None)
    return TestClient(app)


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake(path):
        async def sdk(**kwargs):
            calls.append(kwargs)
            return RESULTS[path]

        return sdk

    # The routers import the SDK functions at call time from their packages.
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.search"), "search", fake("/api/v1/search")
    )
    monkeypatch.setattr(sys.modules["cognee.api.v1.recall"], "recall", fake("/api/v1/recall"))
    return calls


@pytest.mark.parametrize("path", ["/api/v1/search", "/api/v1/recall"])
def test_old_json_and_explicit_defaults_reach_the_sdk_unfiltered(client, captured, path):
    old = {"searchType": "CHUNKS", "query": "q", "topK": 3, "datasets": ["d"]}
    responses = [
        client.post(path, json=old),
        client.post(path, json=old | {"minSimilarity": None, "excludeSuperseded": False}),
        client.post(path, json=old | {"min_similarity": None, "exclude_superseded": False}),
    ]

    for response in responses:
        assert response.status_code == 200, response.text
        assert response.json() == RESULTS[path]
    assert captured[0] == captured[1] == captured[2]
    assert captured[0]["min_similarity"] is None
    assert captured[0]["exclude_superseded"] is False


@pytest.mark.parametrize("path", ["/api/v1/search", "/api/v1/recall"])
@pytest.mark.parametrize(
    "fields",
    [
        {"minSimilarity": 0.6, "excludeSuperseded": True},
        {"min_similarity": 0.6, "exclude_superseded": True},
    ],
)
def test_filters_reach_the_sdk(client, captured, path, fields):
    response = client.post(path, json={"searchType": "CHUNKS", "query": "q", **fields})

    assert response.status_code == 200, response.text
    assert captured[0]["min_similarity"] == 0.6
    assert captured[0]["exclude_superseded"] is True


def post_raw(client, path, fragment):
    body = '{"searchType": "CHUNKS", "query": "q", ' + fragment + "}"
    return client.post(path, content=body, headers={"Content-Type": "application/json"})


@pytest.mark.parametrize("path", ["/api/v1/search", "/api/v1/recall"])
@pytest.mark.parametrize(
    "fragment",
    [
        '"minSimilarity": "0.5"',
        '"minSimilarity": true',
        '"excludeSuperseded": "true"',
        '"excludeSuperseded": 1',
    ],
)
def test_wrongly_typed_values_are_rejected(client, captured, path, fragment):
    response = post_raw(client, path, fragment)

    assert response.status_code == 400, response.text
    assert captured == []


# The real SDK functions run: they reject the value before touching any store.
@pytest.mark.parametrize("path", ["/api/v1/search", "/api/v1/recall"])
@pytest.mark.parametrize(
    "fragment",
    [
        '"minSimilarity": NaN',
        '"minSimilarity": Infinity',
        '"minSimilarity": -Infinity',
        '"minSimilarity": 1.5',
        '"minSimilarity": -0.1',
    ],
)
def test_out_of_range_or_non_finite_thresholds_are_rejected(client, path, fragment):
    response = post_raw(client, path, fragment)

    assert response.status_code == 422, response.text
    assert "InvalidMinSimilarity" in response.text


@pytest.mark.parametrize("path", ["/api/v1/search", "/api/v1/recall"])
@pytest.mark.parametrize("fragment", ['"minSimilarity": 0', '"minSimilarity": 1'])
def test_threshold_bounds_are_accepted(client, captured, path, fragment):
    response = post_raw(client, path, fragment)

    assert response.status_code == 200, response.text
    assert captured[0]["min_similarity"] in (0, 1)


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/v1/search", {"searchType": "GRAPH_COMPLETION", "minSimilarity": 0}),
        ("/api/v1/search", {"searchType": "CHUNKS_LEXICAL", "excludeSuperseded": True}),
        # search's default type is HYBRID_COMPLETION.
        ("/api/v1/search", {"minSimilarity": 0.5}),
        ("/api/v1/recall", {"searchType": "RAG_COMPLETION", "minSimilarity": 0.5}),
        # recall never routes to CHUNKS by itself: the filter needs it pinned.
        ("/api/v1/recall", {"excludeSuperseded": True}),
    ],
)
def test_filters_for_another_search_type_are_rejected(client, path, body):
    response = client.post(path, json={"query": "q", **body})

    assert response.status_code == 422, response.text
    assert "InvalidChunkFilterConfig" in response.text

"""POST /conflicts/{id}/resolve with only_if_unresolved: write only into an unresolved conflict."""

import asyncio
import importlib
from contextlib import asynccontextmanager
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cognee.modules.conflicts import methods
from cognee.modules.conflicts.exceptions import ConflictAlreadyResolvedError
from cognee.tests.unit.modules.conflicts import test_resolve_conflict as base

# The shared fixtures: an in-memory conflict store and the resolve lane over it.
lane, store, resolve = base.lane, base.store, base.resolve

router_module = importlib.import_module("cognee.api.v1.conflicts.routers.get_conflicts_router")


def _facts(lane) -> list[str]:
    return sorted(lane.store.scope.facts)


def _open(lane) -> None:
    """Make the fixture's review-resolved conflict one that no review could order."""
    lane.store.nodes[lane.store.cid]["status"] = "unresolved"


@asynccontextmanager
async def _http(lane):
    app = FastAPI()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: lane.user
    app.include_router(router_module.get_conflicts_router(), prefix="/api/v1/conflicts")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield lambda fact, **body: client.post(
            f"/api/v1/conflicts/{lane.store.cid}/resolve",
            json={"datasetId": str(lane.dataset.id), "selectedFactId": fact, **body},
        )


@pytest.mark.asyncio
async def test_without_the_flag_a_decided_conflict_is_still_revised(lane):
    first, second = _facts(lane)
    await resolve(lane, first)
    async with _http(lane) as post:
        response = await post(second)
    assert response.status_code == 200
    assert response.json()["selectedFactId"] == second
    assert lane.store.nodes[lane.store.cid]["selected_fact_id"] == second


@pytest.mark.asyncio
async def test_flag_on_an_unresolved_conflict_records_the_choice(lane):
    _open(lane)
    fact = _facts(lane)[0]
    async with _http(lane) as post:
        response = await post(fact, onlyIfUnresolved=True)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "resolved" and body["resolutionSource"] == "user"
    assert body["selectedFactId"] == fact
    saved = lane.store.nodes[lane.store.cid]
    assert saved["selected_fact_id"] == fact and saved["resolution_source"] == "user"


@pytest.mark.asyncio
async def test_flag_on_a_user_decided_conflict_answers_409_and_keeps_the_decision(lane):
    _open(lane)
    first, second = _facts(lane)
    await resolve(lane, first, reason="Verified")
    decided = dict(lane.store.nodes[lane.store.cid])
    writes = len(lane.store.calls)
    lane.invalidation.reset_mock()
    async with _http(lane) as post:
        response = await post(second, only_if_unresolved=True, reason="Late")
    assert response.status_code == 409
    assert response.json() == {"error": "already_resolved"}
    assert lane.store.nodes[lane.store.cid] == decided
    assert len(lane.store.calls) == writes
    lane.invalidation.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_on_a_review_resolved_conflict_answers_409(lane):
    # The fixture conflict was resolved by review (status resolved, source review).
    assert lane.store.nodes[lane.store.cid]["status"] == "resolved"
    assert lane.store.nodes[lane.store.cid]["resolution_source"] == "review"
    before = dict(lane.store.nodes[lane.store.cid])
    async with _http(lane) as post:
        response = await post(_facts(lane)[0], onlyIfUnresolved=True)
    assert response.status_code == 409
    assert response.json() == {"error": "already_resolved"}
    assert lane.store.nodes[lane.store.cid] == before
    assert not lane.store.calls


@pytest.mark.asyncio
async def test_two_concurrent_conditional_choices_record_exactly_one(lane, monkeypatch):
    """Both calls read the conflict while it is unresolved unless the check and the write
    share the dataset lock; the slow fact read gives the second call the chance to."""
    _open(lane)
    scope = lane.store.scope

    async def slow_read(*args, **kwargs):
        for _ in range(5):
            await asyncio.sleep(0)
        return scope

    monkeypatch.setattr(methods, "read_entity_facts", slow_read)
    first, second = _facts(lane)
    outcomes = await asyncio.gather(
        resolve(lane, first, only_if_unresolved=True),
        resolve(lane, second, only_if_unresolved=True),
        return_exceptions=True,
    )
    refused = [o for o in outcomes if isinstance(o, ConflictAlreadyResolvedError)]
    recorded = [o for o in outcomes if not isinstance(o, BaseException)]
    assert len(refused) == 1 and len(recorded) == 1, outcomes
    winner = str(recorded[0].selected_fact_id)
    assert lane.store.nodes[lane.store.cid]["selected_fact_id"] == winner
    assert UUID(winner) == recorded[0].selected_fact_id

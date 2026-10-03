"""Server-only I2/I3/I4/I5/I6 smoke on the local isolated stack.

Real upload/add/cognify/preflight/storage with deterministic LLM/embedding
fixtures. No client implementation, deployed server, or live volumes.
"""

import asyncio
import hashlib
import importlib
import json
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select, update

from cognee.tests.e2e.incremental_update import test_add_existing_document as existing_tests
from cognee.tests.e2e.incremental_update.backend_env import reset_backend_state
from cognee.tests.e2e.incremental_update.test_add_existing_document import _upload

add_env = existing_tests.add_env

pytestmark = pytest.mark.asyncio


async def _client(user):
    from cognee.api.client import exception_handler
    from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
    from cognee.exceptions import CogneeApiError
    from cognee.modules.users.methods import get_authenticated_user

    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/api/v1/remember")
    app.add_exception_handler(CogneeApiError, exception_handler)

    async def authenticated():
        return user

    app.dependency_overrides[get_authenticated_user] = authenticated
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _post(client, dataset_id, text, name=None):
    filename = name or "pi-cognee-" + hashlib.sha256(text).hexdigest() + ".txt"
    return await client.post(
        "/api/v1/remember",
        data={
            "datasetId": str(dataset_id),
            "run_in_background": "true",
            "self_improvement": "false",
            "node_set": ["project_docs", "test_notes"],
        },
        files={"data": (filename, text, "text/plain")},
    )


async def _finish(client, accepted):
    assert accepted.status_code == 202, accepted.text
    body = accepted.json()
    assert body["status"] == "running"
    UUID(body["job_id"])
    module = importlib.import_module("cognee.api.v1.remember.remember")
    await asyncio.wait_for(asyncio.gather(*list(module._BACKGROUND_REMEMBER_TASKS)), 60)
    response = await client.get("/api/v1/remember/jobs/" + body["job_id"])
    assert response.status_code == 200, response.text
    return response.json()


async def _stored(user, data_id):
    from cognee.infrastructure.files.utils.open_data_file import open_data_file
    from cognee.modules.data.methods.get_data import get_data

    row = await get_data(user.id, data_id)
    async with open_data_file(row.raw_data_location, mode="r", encoding="utf-8") as file:
        content = file.read()
    return row.id, row.name, row.content_hash, row.node_set, content


async def test_new_jobs_preserve_legacy_documents_and_identical_repeat(add_env):
    import cognee
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.data.models.Data import Data
    from cognee.modules.users.methods import get_default_user

    await reset_backend_state()
    # Legacy same-name rows are fixture history, not a request to rename live data.
    for name, text in [
        ("legacy_one.txt", b"Old ENTOLDONE note.\n"),
        ("legacy_two.txt", b"Old ENTOLDTWO note.\n"),
    ]:
        await cognee.add(_upload(text, name), dataset_name="jobs")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "jobs")
    old_ids = [row.id for row in await get_dataset_data(dataset.id)]
    async with get_relational_engine().get_async_session() as session:
        await session.execute(update(Data).where(Data.id.in_(old_ids)).values(name="project_docs"))
        await session.commit()
    before = [await _stored(user, data_id) for data_id in old_ids]
    async with await _client(user) as client:
        assert (await client.get("/api/v1/remember/capabilities")).json() == {
            "document_job_status_version": 1
        }
        first_text, second_text = b"New ENTFIRST note.\n", b"New ENTSECOND note.\n"
        first = await _finish(client, await _post(client, dataset.id, first_text))
        second = await _finish(client, await _post(client, dataset.id, second_text))
        repeat = await _finish(client, await _post(client, dataset.id, first_text))
        assert first["status"] == second["status"] == repeat["status"] == "completed"
        assert len({first["job_id"], second["job_id"], repeat["job_id"]}) == 3
        assert first["pipeline_run_id"] != first["job_id"]
        rows = await get_dataset_data(dataset.id)
        assert len(rows) == 4
        new_rows = [row for row in rows if row.id not in old_ids]
        assert {row.name for row in new_rows} == {
            "pi-cognee-" + hashlib.sha256(text).hexdigest() for text in (first_text, second_text)
        }
        # Ingestion currently stores node_set as JSON text in a JSON column.
        assert all(
            (json.loads(row.node_set) if isinstance(row.node_set, str) else row.node_set)
            == ["project_docs", "test_notes"]
            for row in new_rows
        )
        assert {(await _stored(user, row.id))[-1].strip() for row in new_rows} == {
            first_text.decode().strip(),
            second_text.decode().strip(),
        }
        assert [await _stored(user, data_id) for data_id in old_ids] == before
        # Real original conflict checker must still reject category-name reuse.
        rejected = await _post(client, dataset.id, b"Changed legacy ENTNEW.\n", "project_docs.txt")
        assert rejected.status_code == 409, rejected.text
        assert "job_id" not in rejected.json()
        assert len(await get_dataset_data(dataset.id)) == 4
        assert [await _stored(user, data_id) for data_id in old_ids] == before
    # Rebuild the reader/engine, rather than accepting an in-memory success.
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    create_relational_engine.cache_clear()
    async with await _client(user) as client:
        saved = await client.get("/api/v1/remember/jobs/" + first["job_id"])
    assert saved.json() == first
    async with get_relational_engine().get_async_session() as session:
        job_rows = list(
            (
                await session.execute(
                    select(cognee.modules.pipelines.models.PipelineRun).where(
                        cognee.modules.pipelines.models.PipelineRun.pipeline_run_id
                        == UUID(first["job_id"])
                    )
                )
            ).scalars()
        )
    assert len(job_rows) == 1


async def test_late_add_conflict_and_partial_cognify_failure_have_specific_job(
    add_env, monkeypatch
):
    import cognee
    from cognee.api.v1.exceptions import DocumentUpdateRequiredError
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    await reset_backend_state()
    await cognee.add(_upload(b"Fixture ENTBASE.\n", "base.txt"), dataset_name="jobs")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "jobs")
    refuse_module = importlib.import_module(
        "cognee.tasks.ingestion.refuse_changed_existing_documents"
    )
    real_refuse, calls = refuse_module.refuse_changed_existing_documents, []

    async def late_conflict(data, owner, target):
        calls.append(target.id)
        if len(calls) == 2:
            raise DocumentUpdateRequiredError(
                [{"name": "late.txt", "data_id": dataset.id}], dataset.id
            )
        return await real_refuse(data, owner, target)

    async with await _client(user) as client:
        with monkeypatch.context() as patch:
            patch.setattr(refuse_module, "refuse_changed_existing_documents", late_conflict)
            patch.setattr(
                importlib.import_module("cognee.api.v1.add.add"),
                "refuse_changed_existing_documents",
                late_conflict,
            )
            failed = await _finish(client, await _post(client, dataset.id, b"Late ENTLATE.\n"))
        assert calls == [dataset.id, dataset.id], "preflight and add both check"
        assert failed["status"] == "errored"
        assert failed["error_class"] == "DocumentUpdateRequiredError"
        assert failed["error_http_status"] == 409 and failed["error"]
        assert len(await get_dataset_data(dataset.id)) == 1

        async def cognify_broke(**kwargs):
            raise RuntimeError("injected cognify outage")

        with monkeypatch.context() as patch:
            patch.setattr("cognee.api.v1.cognify.cognify", cognify_broke)
            partial = await _finish(
                client, await _post(client, dataset.id, b"Partial ENTPARTIAL.\n")
            )
        assert partial["status"] == "errored"
        assert partial["error_class"] == "RuntimeError" and partial["error"]
        assert len(await get_dataset_data(dataset.id)) == 2, "add can land before cognify fails"
        success = await _finish(client, await _post(client, dataset.id, b"Success ENTSUCCESS.\n"))
        assert success["status"] == "completed"
        assert (await client.get("/api/v1/remember/jobs/" + failed["job_id"])).json() == failed
        assert (await client.get("/api/v1/remember/jobs/" + partial["job_id"])).json() == partial


async def test_real_dataset_write_permission_is_checked_before_acceptance(add_env, monkeypatch):
    from uuid import uuid4

    import cognee
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.users.methods import create_user, get_default_user

    await reset_backend_state()
    await cognee.add(_upload(b"Owner ENTOWNER.\n", "owner.txt"), dataset_name="private")
    owner = await get_default_user()
    dataset = next(d for d in await get_datasets(owner.id) if d.name == "private")
    intruder = await create_user(f"intruder_{uuid4().hex}@example.com", "test-only-password")
    called = []

    async def forbidden_add(**kwargs):
        called.append("add")
        raise AssertionError("must authorize before add")

    monkeypatch.setattr("cognee.api.v1.add.add", forbidden_add)
    async with await _client(intruder) as client:
        response = await _post(client, dataset.id, b"Unauthorized ENTNO.\n")
    assert response.status_code == 403, response.text
    assert "job_id" not in response.json()
    assert called == []

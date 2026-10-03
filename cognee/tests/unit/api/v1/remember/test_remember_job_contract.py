"""Wire and stage regressions for plan S / I3-I7, I11.

Pipelines are controlled doubles; the job store is real isolated SQLite, as
configured by the unit-suite fixture. The production exception handler is used
for pre-acceptance HTTP errors, without starting the API lifespan.
"""

import asyncio
import importlib
from io import BytesIO
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from cognee.exceptions import CogneeApiError
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunAlreadyCompleted
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user

from . import test_remember_job_status as job_tests
from .test_remember_job_status import (
    _app,
    _background,
    _client,
    _completed,
    _errored,
    _no_db_setup,
    _user,
    job_module,
    remember_module,
)

stages = job_tests.stages
code_remember_env = importlib.import_module(
    "cognee.tests.unit.tasks.code_graph.test_remember_code_graph"
).code_remember_env
fake_import = importlib.import_module(
    "cognee.tests.unit.migration.test_remember_router_cogx"
).fake_import

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("background", [False, True], ids=["blocking", "background"])
@pytest.mark.parametrize("stage", ["add", "cognify"])
@pytest.mark.parametrize(
    "outcome",
    [
        lambda: RuntimeError("stage raised for alice@example.com"),
        lambda: _errored("StageBroke", "stage failed for alice@example.com"),
        lambda: _completed(data_ingestion_info=[{"run_info": _errored("ItemBroke")}]),
        lambda: SimpleNamespace(status="PipelineRunStarted"),
        lambda: SimpleNamespace(status="unexpected"),
        lambda: None,
        lambda: {"good": _completed(), "bad": _errored()},
        lambda: _errored(None, None),
    ],
    ids=["raised", "returned", "per-item", "running", "unknown", "missing", "mixed", "no-detail"],
)
async def test_stage_failure_never_becomes_document_success(stages, background, stage, outcome):
    stages[stage] = outcome
    user = _user()
    # Blocking raised exceptions retain the existing loud contract even with
    # raise_on_error=False (which only controls returned pipeline failures).
    if isinstance(outcome(), BaseException) and not background:
        with pytest.raises(RuntimeError):
            await remember_module.remember(
                "note",
                dataset_id=uuid4(),
                user=user,
                self_improvement=False,
                raise_on_error=False,
            )
    else:
        result = await remember_module.remember(
            "note",
            dataset_id=uuid4(),
            user=user,
            run_in_background=background,
            self_improvement=False,
            raise_on_error=False,
        )
        await result
        body = (
            await job_module.get_job_snapshot(result.job_id, user)
            if background
            else result.to_dict()
        )
        assert body["status"] == "errored"
        assert body["error"] and body["error_class"] and body["error_http_status"]
        if getattr(outcome(), "status", None) in ("PipelineRunStarted", "unexpected"):
            assert body["error_class"] == "RememberStageNotTerminal"
        if background:
            assert "alice@example.com" not in body["error"]
    assert stages["calls"] == (["add"] if stage == "add" else ["add", "cognify"])


async def test_deduplicated_terminal_success_can_have_zero_items(stages):
    def already():
        return PipelineRunAlreadyCompleted(
            pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="ds"
        )

    stages["add"] = already
    stages["cognify"] = lambda: {"ds": already()}
    user = _user()
    result = await _background(user)
    await result
    body = await job_module.get_job_snapshot(result.job_id, user)
    assert body["status"] == "completed"
    assert body["items_processed"] == 0
    assert body["pipeline_run_id"] != body["job_id"]


async def test_job_stays_running_until_optional_improve_finishes(stages, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def improve(**kwargs):
        entered.set()
        await release.wait()
        raise RuntimeError("optional improvement failed")

    monkeypatch.setattr(importlib.import_module("cognee.api.v1.improve"), "improve", improve)
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.auto_improve_debounce"),
        "auto_improve_enabled",
        lambda: True,
    )
    user = _user()
    result = await remember_module.remember(
        "note",
        dataset_id=uuid4(),
        user=user,
        run_in_background=True,
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        body = await job_module.get_job_snapshot(result.job_id, user)
        assert body["status"] == "running"
        assert "improve_error" not in body
    finally:
        release.set()
        await result
    body = await job_module.get_job_snapshot(result.job_id, user)
    assert body["status"] == "completed"
    assert body["improve_error"] == "optional improvement failed"


async def test_closed_request_upload_is_owned_by_background_task(stages, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    read = []

    async def add(data, **kwargs):
        entered.set()
        await release.wait()
        read.append((data.filename, data.file.read()))
        return _completed()

    monkeypatch.setattr("cognee.api.v1.add.add", add)
    upload = UploadFile(file=BytesIO("заметка\n".encode()), filename="note.txt")
    result = await remember_module.remember(
        upload,
        dataset_id=uuid4(),
        user=_user(),
        run_in_background=True,
        self_improvement=False,
    )
    try:
        await upload.close()
        await asyncio.wait_for(entered.wait(), 5)
    finally:
        release.set()
        await result
    assert result.status == "completed"
    assert read == [("note.txt", "заметка\n".encode())]


async def test_http_blocking_success_and_failure_preserve_legacy_fields(stages):
    user = _user()
    async with _client(user) as client:
        success = await client.post(
            "/remember",
            data={
                "datasetId": str(uuid4()),
                "raw_data": "note",
                "self_improvement": "false",
            },
        )
        assert success.status_code == 200
        body = success.json()
        assert body["status"] == "completed"
        assert "job_id" not in body
        assert {
            "dataset_name",
            "dataset_id",
            "pipeline_run_id",
            "items_processed",
            "elapsed_seconds",
        } <= body.keys()
        stages["cognify"] = lambda: {"ds": _errored()}
        failed = await client.post(
            "/remember",
            data={
                "datasetId": str(uuid4()),
                "raw_data": "note",
                "self_improvement": "false",
            },
        )
    assert failed.status_code == 409
    assert failed.json()["status"] == "errored"
    assert failed.json()["error"]


@pytest.mark.parametrize("kind", ["conflict", "no-write", "invalid", "initial-store"])
async def test_http_rejection_never_accepts_or_starts_a_task(stages, monkeypatch, kind):
    from cognee.api.client import exception_handler
    from cognee.api.v1.exceptions import DocumentUpdateRequiredError

    app = _app(_user())
    app.add_exception_handler(CogneeApiError, exception_handler)
    inserted = []
    original = job_module.insert_running_job

    async def insert(context, result):
        inserted.append(context.operation_id)
        if kind == "initial-store":
            raise RuntimeError("test database unavailable")
        await original(context, result)

    monkeypatch.setattr(job_module, "insert_running_job", insert)

    async def conflict(*args):
        raise DocumentUpdateRequiredError([{"name": "note.txt", "data_id": uuid4()}], uuid4())

    async def denied(**kwargs):
        raise PermissionDeniedError()

    if kind == "conflict":
        monkeypatch.setattr(
            "cognee.tasks.ingestion.refuse_changed_existing_documents.refuse_changed_existing_documents",
            conflict,
        )
    if kind == "no-write":
        monkeypatch.setattr(
            "cognee.modules.pipelines.layers.resolve_authorized_user_dataset.resolve_authorized_user_dataset",
            denied,
        )
    before = set(remember_module._BACKGROUND_REMEMBER_TASKS)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        response = await client.post(
            "/remember",
            data={
                "datasetId": str(uuid4()),
                "run_in_background": "true",
                **({} if kind == "invalid" else {"raw_data": "note"}),
            },
        )
    assert (
        response.status_code
        == {"conflict": 409, "no-write": 403, "invalid": 400, "initial-store": 409}[kind]
    )
    assert "job_id" not in response.json()
    assert stages["calls"] == []
    assert set(remember_module._BACKGROUND_REMEMBER_TASKS) == before
    assert len(inserted) == (1 if kind == "initial-store" else 0)


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("path", ["/remember/capabilities", "/remember/jobs/" + str(uuid4())])
async def test_job_routes_use_existing_auth_dependency(status, path):
    app = _app(_user())

    async def rejected():
        raise HTTPException(status_code=status)

    app.dependency_overrides[get_authenticated_user] = rejected
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        response = await client.get(path)
    assert response.status_code == status


async def test_foreign_tenant_unknown_and_legacy_jobs_have_same_http_404(stages):
    from cognee.modules.operations import record_operation

    owner = _user(tenant_id=uuid4())
    result = await _background(owner)
    await result
    async with record_operation("remember", user=owner) as legacy:
        pass
    async with (
        _client(owner) as own_client,
        _client(SimpleNamespace(id=owner.id, tenant_id=uuid4())) as foreign_client,
    ):
        responses = [
            await own_client.get(f"/remember/jobs/{uuid4()}"),
            await own_client.get(f"/remember/jobs/{legacy.operation_id}"),
            await foreign_client.get(f"/remember/jobs/{result.job_id}"),
        ]
        own = await own_client.get(f"/remember/jobs/{result.job_id}")
    assert own.status_code == 200
    assert len({(r.status_code, r.text) for r in responses}) == 1
    assert responses[0].status_code == 404


@pytest.mark.parametrize("special", ["session", "skills"])
async def test_special_remember_paths_do_not_create_document_jobs(stages, monkeypatch, special):
    from contextlib import asynccontextmanager

    async def store_forbidden(*args, **kwargs):
        raise AssertionError("special paths must not create document jobs")

    monkeypatch.setattr(job_module, "insert_running_job", store_forbidden)
    owner = _user()
    kwargs = {
        "dataset_id": uuid4(),
        "user": owner,
        "run_in_background": True,
        "self_improvement": False,
    }
    if special == "session":

        async def to_session(*args, **kwargs):
            return None

        monkeypatch.setattr(remember_module, "_add_to_session", to_session)
        kwargs["session_id"] = "test-session"
    else:
        dataset = SimpleNamespace(id=kwargs["dataset_id"], name="skills", owner_id=owner.id)

        async def resolve(*args, **kwargs):
            return owner, [dataset]

        @asynccontextmanager
        async def context(*args, **kwargs):
            yield

        async def add_skills(*args, **kwargs):
            return [SimpleNamespace(name="skill", declared_tools=[])]

        monkeypatch.setattr(remember_module, "resolve_authorized_user_datasets", resolve)
        monkeypatch.setattr(
            "cognee.context_global_variables.set_database_global_context_variables", context
        )
        monkeypatch.setattr("cognee.modules.tools.add_skills", add_skills)
        kwargs["content_type"] = "skills"
    result = await remember_module.remember("note", **kwargs)
    assert result.status == ("session_stored" if special == "session" else "completed")
    assert result.job_id is None
    assert "job_id" not in result.to_dict()
    assert stages["calls"] == []


async def test_returned_improve_error_keeps_completed_job_and_warning(stages, monkeypatch):
    from cognee.modules.improve import ImproveResult, StageResult

    warning = ImproveResult(
        stages=[StageResult.errored("triplet_enrichment", "embedding backend down")],
        memify_run={},
    )

    async def improve(**kwargs):
        return warning

    monkeypatch.setattr(importlib.import_module("cognee.api.v1.improve"), "improve", improve)
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.auto_improve_debounce"),
        "auto_improve_enabled",
        lambda: True,
    )
    owner = _user()
    result = await remember_module.remember(
        "note",
        dataset_id=uuid4(),
        user=owner,
        run_in_background=True,
    )
    await result
    async with _client(owner) as client:
        response = await client.get(f"/remember/jobs/{result.job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["improve"] == warning.model_dump(mode="json")
    assert body["improve_error"] == "triplet_enrichment: embedding backend down"
    assert "error" not in body


async def test_background_code_http_keeps_legacy_non_job_contract(code_remember_env, monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("code must not create document jobs")

    monkeypatch.setattr(job_module, "insert_running_job", forbidden)
    async with _client(_user()) as client:
        response = await client.post(
            "/remember",
            data={
                "content_type": "code",
                "run_in_background": "true",
                "raw_data": "/test/repo",
                "datasetName": "my_code",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert "job_id" not in body
    assert body["dataset_id"] == str(code_remember_env["dataset"].id)
    await asyncio.gather(*list(remember_module._BACKGROUND_REMEMBER_TASKS))
    code_remember_env["pipeline"].assert_awaited_once()


async def test_background_archive_http_keeps_legacy_non_job_contract(
    fake_import, tmp_path, monkeypatch
):
    from cognee.tests.unit.migration.test_remember_router_cogx import _packed_archive_bytes

    async def forbidden(*args, **kwargs):
        raise AssertionError("archives must not create document jobs")

    monkeypatch.setattr(job_module, "insert_running_job", forbidden)
    async with _client(_user()) as client:
        response = await client.post(
            "/remember",
            data={
                "content_type": "cogx-archive",
                "run_in_background": "true",
                "datasetName": "archive",
            },
            files={
                "data": ("sample.cogx.tar.gz", _packed_archive_bytes(tmp_path), "application/gzip")
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert "job_id" not in body
    assert body["items_processed"] == 3
    assert len(fake_import.calls) == 1 and fake_import.calls[0]["run_in_background"] is True

"""Durable job status of a background document remember (job status v1).

A background ``remember()`` of ordinary permanent data stores a job in the
remember operation's ``pipeline_runs`` row before it starts and its terminal
state after the whole coroutine finished; ``GET /v1/remember/jobs/{job_id}``
reads it back scoped to the owner. Stage results that *return* a failure (an
errored add run info, a non-terminal result) fail the job and stop the chain.

The store tests use the unit suite's real SQLite database (see
``cognee/tests/unit/conftest.py``); the pipelines are stubbed.
"""

import asyncio
import importlib
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunAlreadyCompleted,
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
)
from cognee.modules.users.methods import get_authenticated_user

remember_module = importlib.import_module("cognee.api.v1.remember.remember")
job_module = importlib.import_module("cognee.api.v1.remember.remember_job")


def _user(tenant_id=None):
    return SimpleNamespace(id=uuid4(), tenant_id=tenant_id, email="u@example.com")


def _completed(**kwargs):
    return PipelineRunCompleted(
        pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="ds", **kwargs
    )


def _errored(error_class="AddBroke", error_message="add item failed"):
    return PipelineRunErrored(
        pipeline_run_id=uuid4(),
        dataset_id=uuid4(),
        dataset_name="ds",
        error_class=error_class,
        error_message=error_message,
    )


@pytest.fixture(autouse=True)
def _no_db_setup(monkeypatch):
    async def _noop_setup():
        return None

    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", _noop_setup)


@pytest.fixture
def stages(monkeypatch, stub_document_preflight):
    """Stub add()/cognify(); tests set what each returns or raises."""
    state = {
        "add": lambda: _completed(),
        "cognify": lambda: {"ds": _completed()},
        "calls": [],
        "release": None,
    }

    async def fake_add(*args, **kwargs):
        state["calls"].append("add")
        outcome = state["add"]()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def fake_cognify(*args, **kwargs):
        state["calls"].append("cognify")
        if state["release"] is not None:
            await state["release"].wait()
        outcome = state["cognify"]()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr("cognee.api.v1.add.add", fake_add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", fake_cognify)
    return state


async def _background(user, **kwargs):
    return await remember_module.remember(
        "note",
        dataset_id=uuid4(),
        run_in_background=True,
        self_improvement=False,
        user=user,
        **kwargs,
    )


# --- stage classification --------------------------------------------------


@pytest.mark.parametrize(
    "stage_result",
    [
        _completed(),
        {"ds": _completed()},
        PipelineRunAlreadyCompleted(pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="ds"),
    ],
)
def test_terminal_success_is_not_a_failure(stage_result):
    assert remember_module._stage_failure("add", stage_result) is None


@pytest.mark.parametrize(
    "stage_result, error_class",
    [
        (None, "RememberStageNoResult"),
        ({}, "RememberStageNoResult"),
        (_errored(), "AddBroke"),
        ({"ds": _errored()}, "AddBroke"),
        (
            PipelineRunStarted(pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="ds"),
            "RememberStageNotTerminal",
        ),
        (
            _completed(data_ingestion_info=[{"run_info": _errored("ItemBroke", "bad item")}]),
            "ItemBroke",
        ),
    ],
)
def test_anything_but_terminal_success_is_a_failure(stage_result, error_class):
    failure = remember_module._stage_failure("add", stage_result)
    assert failure is not None
    assert failure.error_class == error_class
    assert failure.message.startswith("add: ")


# --- background jobs ---------------------------------------------------------


@pytest.mark.asyncio
async def test_background_job_runs_then_completes(stages):
    user = _user()
    stages["release"] = asyncio.Event()

    result = await _background(user)

    assert result.status == "running"
    assert result.job_id
    running = await job_module.get_job_snapshot(result.job_id, user)
    assert running["status"] == "running"
    assert running["job_id"] == result.job_id
    assert "error" not in running

    stages["release"].set()
    await result
    done = await job_module.get_job_snapshot(result.job_id, user)
    assert done["status"] == "completed"
    assert done["pipeline_run_id"] == result.pipeline_run_id
    assert done["pipeline_run_id"] != result.job_id
    assert stages["calls"] == ["add", "cognify"]


@pytest.mark.asyncio
async def test_returned_add_failure_stops_the_chain_and_errors_the_job(stages):
    user = _user()
    stages["add"] = lambda: _errored("DocumentUpdateRequiredError", "name holds other content")

    result = await _background(user)
    await result

    assert stages["calls"] == ["add"]
    snapshot = await job_module.get_job_snapshot(result.job_id, user)
    assert snapshot["status"] == "errored"
    assert snapshot["error_class"] == "DocumentUpdateRequiredError"
    assert snapshot["error"] == "add: name holds other content"
    assert snapshot["error_http_status"] == 409


@pytest.mark.asyncio
async def test_returned_cognify_failure_errors_the_job(stages):
    user = _user()
    stages["cognify"] = lambda: {"ds": _errored("LLMDown", "provider refused")}

    result = await _background(user)
    await result

    snapshot = await job_module.get_job_snapshot(result.job_id, user)
    assert snapshot["status"] == "errored"
    assert snapshot["error_class"] == "LLMDown"


@pytest.mark.asyncio
async def test_empty_cognify_result_is_not_completed(stages):
    user = _user()
    stages["cognify"] = dict

    result = await _background(user)
    await result

    snapshot = await job_module.get_job_snapshot(result.job_id, user)
    assert snapshot["status"] == "errored"
    assert snapshot["error_class"] == "RememberStageNoResult"


@pytest.mark.asyncio
async def test_raised_stage_error_is_scrubbed(stages):
    user = _user()
    stages["add"] = lambda: RuntimeError("boom for alice@example.com")

    result = await _background(user)
    await result

    snapshot = await job_module.get_job_snapshot(result.job_id, user)
    assert snapshot["status"] == "errored"
    assert snapshot["error_class"] == "RuntimeError"
    assert "alice@example.com" not in snapshot["error"]
    assert snapshot["error_http_status"] == 409


@pytest.mark.asyncio
async def test_cancelled_job_is_errored(stages):
    user = _user()
    stages["release"] = asyncio.Event()

    result = await _background(user)
    await asyncio.sleep(0)
    result._task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await result._task

    snapshot = await job_module.get_job_snapshot(result.job_id, user)
    assert snapshot["status"] == "errored"
    assert snapshot["error_class"] == "CancelledError"
    assert snapshot["error"]


@pytest.mark.asyncio
async def test_improve_failure_keeps_the_job_completed(stages, monkeypatch):
    user = _user()

    async def failing_improve(**kwargs):
        raise RuntimeError("enrichment exploded")

    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.improve"), "improve", failing_improve
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.auto_improve_debounce"),
        "auto_improve_enabled",
        lambda: True,
    )

    result = await remember_module.remember(
        "note", dataset_id=uuid4(), run_in_background=True, user=user
    )
    await result

    snapshot = await job_module.get_job_snapshot(result.job_id, user)
    assert snapshot["status"] == "completed"
    assert snapshot["improve_error"] == "enrichment exploded"


@pytest.mark.asyncio
async def test_failed_initial_insert_starts_nothing(stages, monkeypatch):
    async def broken_insert(context, result):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(job_module, "insert_running_job", broken_insert)

    with pytest.raises(RuntimeError, match="database is locked"):
        await _background(_user())
    await asyncio.sleep(0)
    assert stages["calls"] == []


@pytest.mark.asyncio
async def test_failed_terminal_write_never_reads_completed(stages, monkeypatch):
    user = _user()

    class _BrokenSession:
        async def __aenter__(self):
            raise RuntimeError("store went away")

        async def __aexit__(self, *exc):
            return False

    result = await _background(user)
    monkeypatch.setattr(
        job_module,
        "get_relational_engine",
        lambda: SimpleNamespace(get_async_session=lambda: _BrokenSession()),
    )
    await result
    monkeypatch.undo()

    snapshot = await job_module.get_job_snapshot(result.job_id, user)
    assert result.status == "completed"
    assert snapshot["status"] == "running"


@pytest.mark.asyncio
async def test_terminal_state_is_written_once(stages):
    user = _user()
    stages["add"] = lambda: _errored()

    result = await _background(user)
    await result
    context = SimpleNamespace(
        operation_id=__import__("uuid").UUID(result.job_id),
        run_info=None,
        usage=SimpleNamespace(tokens_in=0, tokens_out=0),
    )
    result.status = "completed"
    assert await job_module.finish_job(context, result) is False

    snapshot = await job_module.get_job_snapshot(result.job_id, user)
    assert snapshot["status"] == "errored"


@pytest.mark.asyncio
async def test_job_is_scoped_to_its_user_and_tenant(stages):
    tenant = uuid4()
    owner = _user(tenant_id=tenant)

    result = await _background(owner)
    await result

    same_owner = await job_module.get_job_snapshot(result.job_id, owner)
    other_user = await job_module.get_job_snapshot(result.job_id, _user(tenant_id=tenant))
    other_tenant = await job_module.get_job_snapshot(
        result.job_id, SimpleNamespace(id=owner.id, tenant_id=uuid4())
    )
    assert same_owner["status"] == "completed"
    assert other_user is None
    assert other_tenant is None


@pytest.mark.asyncio
async def test_legacy_remember_row_is_not_a_job():
    from cognee.modules.operations import record_operation

    user = _user()
    async with record_operation("remember", user=user) as context:
        context.set_background(True)

    assert await job_module.get_job_snapshot(context.operation_id, user) is None


@pytest.mark.asyncio
async def test_job_survives_a_new_engine(stages):
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    user = _user()
    result = await _background(user)
    await result

    create_relational_engine.cache_clear()
    snapshot = await job_module.get_job_snapshot(result.job_id, user)
    assert snapshot["status"] == "completed"


@pytest.mark.asyncio
async def test_blocking_add_failure_is_errored_without_cognify(stages):
    stages["add"] = lambda: _errored()

    result = await remember_module.remember(
        "note",
        dataset_id=uuid4(),
        self_improvement=False,
        user=_user(),
        raise_on_error=False,
    )

    assert result.status == "errored"
    assert result.job_id is None
    assert stages["calls"] == ["add"]


@pytest.mark.asyncio
async def test_blocking_add_failure_raises_when_loud(stages):
    from cognee.api.v1.exceptions import RememberStageFailedError

    stages["add"] = lambda: _errored()

    with pytest.raises(RememberStageFailedError):
        await remember_module.remember(
            "note", dataset_id=uuid4(), self_improvement=False, user=_user()
        )
    assert stages["calls"] == ["add"]


# --- HTTP ----------------------------------------------------------------------


def _app(user):
    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/remember")

    async def override_user():
        return user

    app.dependency_overrides[get_authenticated_user] = override_user
    return app


def _client(user):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(user)), base_url="http://t")


@pytest.mark.asyncio
async def test_capabilities_advertise_job_status_v1():
    async with _client(_user()) as client:
        response = await client.get("/remember/capabilities")
    assert response.status_code == 200
    assert response.json() == {"document_job_status_version": 1}


@pytest.mark.asyncio
async def test_background_post_is_202_and_job_is_readable(stages):
    user = _user()
    async with _client(user) as client:
        accepted = await client.post(
            "/remember",
            data={"datasetName": "docs", "run_in_background": "true", "self_improvement": "false"},
            files={"data": ("note.txt", b"hello", "text/plain")},
        )
        assert accepted.status_code == 202
        body = accepted.json()
        assert body["status"] == "running"
        assert {
            "dataset_name",
            "dataset_id",
            "pipeline_run_id",
            "items_processed",
            "elapsed_seconds",
        } <= body.keys()
        job_id = body["job_id"]
        assert body["pipeline_run_id"] != job_id

        await asyncio.gather(*list(remember_module._BACKGROUND_REMEMBER_TASKS))
        polled = await client.get(f"/remember/jobs/{job_id}")

    assert polled.status_code == 200
    assert polled.json()["status"] == "completed"
    assert polled.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_errored_job_is_200_with_error_body(stages):
    user = _user()
    stages["add"] = lambda: _errored()
    async with _client(user) as client:
        accepted = await client.post(
            "/remember",
            data={"datasetName": "docs", "run_in_background": "true", "self_improvement": "false"},
            files={"data": ("note.txt", b"hello", "text/plain")},
        )
        await asyncio.gather(*list(remember_module._BACKGROUND_REMEMBER_TASKS))
        polled = await client.get(f"/remember/jobs/{accepted.json()['job_id']}")

    assert polled.status_code == 200
    body = polled.json()
    assert body["status"] == "errored"
    assert body["error"] and body["error_class"] and body["error_http_status"] == 409


@pytest.mark.asyncio
async def test_unknown_foreign_and_malformed_jobs_are_404(stages):
    owner = _user()
    result = await _background(owner)
    await result

    async with _client(_user()) as client:
        foreign = await client.get(f"/remember/jobs/{result.job_id}")
        unknown = await client.get(f"/remember/jobs/{uuid4()}")
        malformed = await client.get("/remember/jobs/not-a-uuid")

    assert foreign.status_code == 404
    assert unknown.status_code == 404
    assert malformed.status_code == 404


@pytest.mark.asyncio
async def test_preflight_conflict_is_409_before_acceptance(stages, monkeypatch):
    from cognee.api.v1.exceptions import DocumentUpdateRequiredError

    async def conflict(data, user, dataset):
        raise DocumentUpdateRequiredError([{"name": "note.txt", "data_id": uuid4()}], dataset.id)

    monkeypatch.setattr(
        "cognee.tasks.ingestion.refuse_changed_existing_documents"
        ".refuse_changed_existing_documents",
        conflict,
    )
    inserted = []

    async def tracking_insert(context, result):
        inserted.append(context.operation_id)

    monkeypatch.setattr(job_module, "insert_running_job", tracking_insert)

    with pytest.raises(DocumentUpdateRequiredError):
        await _background(_user())
    assert inserted == []
    assert stages["calls"] == []


@pytest.mark.asyncio
async def test_job_read_error_is_5xx(monkeypatch):
    async def broken(job_id, user):
        raise RuntimeError("db down")

    monkeypatch.setattr(job_module, "get_job_snapshot", broken)
    transport = httpx.ASGITransport(app=_app(_user()), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.get(f"/remember/jobs/{uuid4()}")
    assert response.status_code >= 500


@pytest.mark.asyncio
async def test_blocking_stage_failure_is_recorded_as_failed_operation(stages):
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.pipelines.models import PipelineRun

    user = _user()
    stages["add"] = lambda: _errored()

    await remember_module.remember(
        "note", dataset_id=uuid4(), self_improvement=False, user=user, raise_on_error=False
    )

    async with get_relational_engine().get_async_session() as session:
        outcomes = (
            (
                await session.execute(
                    select(PipelineRun.outcome).where(
                        PipelineRun.user_id == user.id, PipelineRun.operation_name == "remember"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert outcomes == ["failed"]


@pytest.mark.asyncio
async def test_loud_stage_failure_answers_409(stages):
    from cognee.api.v1.exceptions import RememberStageFailedError

    stages["add"] = lambda: _errored()

    with pytest.raises(RememberStageFailedError) as raised:
        await remember_module.remember(
            "note", dataset_id=uuid4(), self_improvement=False, user=_user()
        )
    assert raised.value.status_code == 409


@pytest.mark.asyncio
async def test_improve_error_is_scrubbed_in_the_job(stages, monkeypatch):
    user = _user()

    async def failing_improve(**kwargs):
        raise RuntimeError("auth failed for bob@example.com")

    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.improve"), "improve", failing_improve
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.auto_improve_debounce"),
        "auto_improve_enabled",
        lambda: True,
    )

    result = await remember_module.remember(
        "note", dataset_id=uuid4(), run_in_background=True, user=user
    )
    await result

    snapshot = await job_module.get_job_snapshot(result.job_id, user)
    assert snapshot["status"] == "completed"
    assert snapshot["improve_error"]
    assert "bob@example.com" not in snapshot["improve_error"]

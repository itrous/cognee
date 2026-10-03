"""Real relational contracts of job status v1 (S / I4, I6, I7, I11).

No schema/model doubles: INSERT, UPDATE and new-session reads use the existing
pipeline_runs table created by the isolated unit-suite migration fixture.
"""

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.operations import record_operation
from cognee.modules.operations.usage_accumulator import get_active_operation_usage
from cognee.modules.pipelines.methods.get_pipeline_run_by_dataset import (
    get_latest_pipeline_runs_by_datasets,
    get_unterminated_pipeline_runs,
)
from cognee.modules.pipelines.models import PipelineRun, PipelineRunStatus

from . import test_remember_job_status as job_tests
from .test_remember_job_status import (
    _background,
    _completed,
    _errored,
    _no_db_setup,
    _user,
    job_module,
    remember_module,
)

stages = job_tests.stages

pytestmark = pytest.mark.asyncio


async def _rows(job_id):
    async with get_relational_engine().get_async_session() as session:
        return list(
            (
                await session.execute(
                    select(PipelineRun).where(
                        PipelineRun.pipeline_run_id == UUID(job_id),
                    )
                )
            ).scalars()
        )


async def test_job_updates_one_operation_row_with_attribution_and_tokens(stages, monkeypatch):
    owner = _user(tenant_id=uuid4())
    dataset_id, cognify_id = uuid4(), uuid4()
    started, release = asyncio.Event(), asyncio.Event()

    async def add(**kwargs):
        get_active_operation_usage().add(17, 3)
        return _completed()

    async def cognify(**kwargs):
        started.set()
        await release.wait()
        get_active_operation_usage().add(23, 5)
        return {
            dataset_id: SimpleNamespace(
                status="PipelineRunCompleted",
                pipeline_run_id=cognify_id,
                dataset_name="docs",
                payload=[SimpleNamespace(id=uuid4(), content_hash="content")],
            )
        }

    monkeypatch.setattr("cognee.api.v1.add.add", add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", cognify)
    async with record_operation("parent", user=owner) as parent:
        result = await remember_module.remember(
            "note",
            dataset_id=dataset_id,
            user=owner,
            run_in_background=True,
            self_improvement=False,
        )
    try:
        rows = await _rows(result.job_id)
        assert len(rows) == 1
        row = rows[0]
        row_id = row.id
        assert row.outcome is None and row.ended_at is None
        assert row.status is None and row.pipeline_name is None and row.pipeline_id is None
        assert row.operation_name == "remember"
        assert row.user_id == owner.id and row.tenant_id == owner.tenant_id
        assert row.dataset_id == dataset_id
        assert row.parent_operation_id == parent.operation_id
        assert row.background is True
        assert row.started_at is not None
        snapshot = row.run_info["remember_job"]
        assert snapshot["version"] == 1 and snapshot["status"] == "running"
        assert snapshot["job_id"] == result.job_id
        assert snapshot["dataset_id"] == str(dataset_id)
        assert snapshot["pipeline_run_id"] is None
        await asyncio.wait_for(started.wait(), 5)
    finally:
        release.set()
        await result
    rows = await _rows(result.job_id)
    assert len(rows) == 1 and rows[0].id == row_id
    row = rows[0]
    assert row.outcome == "succeeded" and row.ended_at >= row.started_at
    assert (row.tokens_in, row.tokens_out) == (40, 8)
    assert row.parent_operation_id == parent.operation_id
    assert row.status is None and row.pipeline_name is None
    snapshot = await job_module.get_job_snapshot(result.job_id, owner)
    assert snapshot["pipeline_run_id"] == str(cognify_id)
    assert snapshot["items_processed"] == 1
    assert snapshot["items"][0]["content_hash"] == snapshot["content_hash"] == "content"
    assert {
        "job_id",
        "status",
        "dataset_name",
        "dataset_id",
        "pipeline_run_id",
        "items_processed",
        "elapsed_seconds",
    } <= snapshot.keys()
    assert "version" not in snapshot


async def test_same_dataset_jobs_never_borrow_each_others_result(stages, monkeypatch):
    owner, dataset_id = _user(), uuid4()
    entered, release = asyncio.Event(), asyncio.Event()

    async def add(data, **kwargs):
        if data == "first":
            entered.set()
            await release.wait()
            return _errored("FirstFailed", "first did not land")
        return _completed()

    monkeypatch.setattr("cognee.api.v1.add.add", add)
    first = await remember_module.remember(
        "first",
        dataset_id=dataset_id,
        user=owner,
        run_in_background=True,
        self_improvement=False,
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        second = await remember_module.remember(
            "second",
            dataset_id=dataset_id,
            user=owner,
            run_in_background=True,
            self_improvement=False,
        )
        await second
        assert first.job_id != second.job_id
        assert (await job_module.get_job_snapshot(second.job_id, owner))["status"] == "completed"
        assert (await job_module.get_job_snapshot(first.job_id, owner))["status"] == "running"
    finally:
        release.set()
        await first
    one = await job_module.get_job_snapshot(first.job_id, owner)
    two = await job_module.get_job_snapshot(second.job_id, owner)
    assert one["status"] == "errored" and one["error_class"] == "FirstFailed"
    assert two["status"] == "completed" and "error" not in two


async def test_operation_jobs_do_not_replace_dataset_latest_pipeline_status(stages):
    dataset_id, owner, pipeline_id = uuid4(), _user(), uuid4()
    async with get_relational_engine().get_async_session() as session:
        session.add(
            PipelineRun(
                pipeline_run_id=pipeline_id,
                pipeline_name="cognify_pipeline",
                dataset_id=dataset_id,
                status=PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
            )
        )
        await session.commit()
    result = await remember_module.remember(
        "note",
        dataset_id=dataset_id,
        user=owner,
        run_in_background=True,
        self_improvement=False,
    )
    await result
    latest = await get_latest_pipeline_runs_by_datasets([dataset_id], "cognify_pipeline")
    assert latest[dataset_id].pipeline_run_id == pipeline_id
    assert all(
        row.pipeline_run_id != UUID(result.job_id) for row in await get_unterminated_pipeline_runs()
    )


@pytest.mark.parametrize("terminal", [False, True], ids=["insert-commit", "update-commit"])
async def test_actual_commit_failure_cannot_publish_success(stages, monkeypatch, terminal):
    owner = _user()
    real_engine = get_relational_engine()

    class SessionProxy:
        async def __aenter__(self):
            self.cm = real_engine.get_async_session()
            self.session = await self.cm.__aenter__()
            return self

        async def __aexit__(self, *args):
            return await self.cm.__aexit__(*args)

        def add(self, row):
            self.session.add(row)

        async def execute(self, *args, **kwargs):
            return await self.session.execute(*args, **kwargs)

        async def commit(self):
            raise RuntimeError("injected commit failure")

    if terminal:
        result = await _background(owner)
    with monkeypatch.context() as patch:
        patch.setattr(
            job_module,
            "get_relational_engine",
            lambda: SimpleNamespace(
                get_async_session=SessionProxy,
            ),
        )
        if terminal:
            await result
        else:
            with pytest.raises(RuntimeError, match="injected commit failure"):
                await _background(owner)
    if terminal:
        body = await job_module.get_job_snapshot(result.job_id, owner)
        assert body["status"] == "running"
        row = (await _rows(result.job_id))[0]
        assert row.outcome is None and row.ended_at is None
    else:
        assert stages["calls"] == []
        async with real_engine.get_async_session() as session:
            rows = list(
                (
                    await session.execute(
                        select(PipelineRun).where(
                            PipelineRun.user_id == owner.id,
                        )
                    )
                ).scalars()
            )
        assert not any((row.run_info or {}).get("remember_job") for row in rows)


@pytest.mark.parametrize("status", ["completed", "errored"])
async def test_both_terminal_states_are_immutable(stages, status):
    owner = _user()
    if status == "errored":
        stages["add"] = lambda: _errored()
    result = await _background(owner)
    await result
    before = await job_module.get_job_snapshot(result.job_id, owner)
    result.status = "errored" if status == "completed" else "completed"
    context = SimpleNamespace(
        operation_id=UUID(result.job_id),
        run_info=None,
        usage=SimpleNamespace(tokens_in=999, tokens_out=999),
    )
    assert await job_module.finish_job(context, result) is False
    assert await job_module.get_job_snapshot(result.job_id, owner) == before


async def test_failed_job_row_has_safe_error_and_failed_terminal_outcome(stages):
    owner = _user()
    stages["add"] = lambda: _errored("WriteFailed", "failure for alice@example.com")
    result = await _background(owner)
    await result
    rows = await _rows(result.job_id)
    assert len(rows) == 1
    row = rows[0]
    body = await job_module.get_job_snapshot(result.job_id, owner)
    assert row.outcome == "failed" and row.ended_at is not None
    assert row.error_class == body["error_class"] == "WriteFailed"
    assert row.error_message == body["error"]
    assert row.error_message and "alice@example.com" not in row.error_message
    assert row.status is None and row.pipeline_name is None


async def test_orphan_running_row_is_not_promoted_by_a_new_reader(stages):
    from cognee.api.v1.remember.remember import RememberResult
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.modules.operations import OperationContext

    owner = _user()
    context = OperationContext("remember", user_id=owner.id, dataset_id=uuid4())
    result = RememberResult(
        status="running", dataset_name="docs", dataset_id=str(context.dataset_id)
    )
    await job_module.insert_running_job(context, result)
    create_relational_engine.cache_clear()
    body = await job_module.get_job_snapshot(context.operation_id, owner)
    assert body["status"] == "running"
    assert body["pipeline_run_id"] is None
    assert "error" not in body
    assert stages["calls"] == []

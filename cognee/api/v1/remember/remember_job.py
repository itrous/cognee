"""Durable status of a background document remember (job status v1).

A background ``remember()`` of ordinary permanent data (add + cognify) used to
answer "running" and keep its outcome in memory only: a caller that got the
acceptance had no way to learn whether *its* write landed. This module keeps
that outcome in the remember operation's own ``pipeline_runs`` row, so it
survives the request and is readable from any API worker:

- before the background task starts, the row is INSERTed strictly (no
  swallowed errors) with ``outcome``/``ended_at`` NULL and a versioned
  ``run_info["remember_job"]`` snapshot whose status is ``running``;
- when the task's coroutine has finished, the same row is UPDATEd once with
  the terminal outcome and the final snapshot (``completed`` / ``errored``).

The row is the one ``record_operation("remember")`` would have written
(``pipeline_run_id`` = the operation id, ``status``/``pipeline_name`` NULL),
so legacy pipeline-status readers keep ignoring it; the operation defers its
close so the recorder never writes a second row. Rows without the
``remember_job`` payload (historical acceptance records) are not jobs.

``running`` means "accepted; no terminal result stored", not "a process is
proven alive": a process killed mid-run leaves its last stored state behind,
and nothing here sweeps or re-runs such rows.
"""

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select, update

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

if TYPE_CHECKING:
    from cognee.modules.operations import OperationContext
    from cognee.modules.users.models import User

logger = get_logger("remember_job")

REMEMBER_OPERATION_NAME = "remember"
REMEMBER_JOB_KEY = "remember_job"
# Version of the stored snapshot and of the HTTP contract advertised by
# GET /api/v1/remember/capabilities as ``document_job_status_version``.
REMEMBER_JOB_VERSION = 1

JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_ERRORED = "errored"


def _json_safe(payload: dict) -> dict:
    """Round-trip through JSON so the JSON column never sees a non-JSON value."""
    return json.loads(json.dumps(payload, default=str))


def build_job_snapshot(result, job_id: UUID) -> dict:
    """The stored/served view of one job, built from its ``RememberResult``.

    The legacy ``RememberResult`` fields are kept as they are; a job that did
    not finish successfully is ``errored`` and always carries a non-empty
    ``error``, ``error_class`` and ``error_http_status``.
    """
    snapshot = result.to_dict()
    snapshot.pop("job_id", None)
    status = snapshot.get("status")
    if status not in (JOB_RUNNING, JOB_COMPLETED):
        status = JOB_ERRORED
    snapshot["status"] = status
    if status == JOB_ERRORED:
        from cognee.modules.operations import scrub_error_message

        snapshot["error"] = (
            getattr(result, "_safe_error", None)
            or scrub_error_message(getattr(result, "error", None))
            or "remember failed"
        )
        snapshot["error_class"] = getattr(result, "error_class", None) or "RememberFailedError"
        snapshot["error_http_status"] = getattr(result, "error_http_status", None) or 409
    else:
        for key in ("error", "error_class", "error_http_status"):
            snapshot.pop(key, None)
    if snapshot.get("improve_error"):
        # Exception text, like ``error``: scrubbed before it is stored and served.
        from cognee.modules.operations import scrub_error_message

        snapshot["improve_error"] = scrub_error_message(snapshot["improve_error"])
    return _json_safe(
        {
            "version": REMEMBER_JOB_VERSION,
            "job_id": str(job_id),
            **snapshot,
        }
    )


async def insert_running_job(context: "OperationContext", result) -> None:
    """Persist the job as ``running`` and commit; raise on any failure.

    Must succeed before the background task is created: a job the store does
    not know about is never acknowledged.
    """
    from cognee.modules.operations import get_operation_origin
    from cognee.modules.pipelines.models import PipelineRun

    snapshot = build_job_snapshot(result, context.operation_id)
    snapshot["status"] = JOB_RUNNING
    run_info = {**(context.run_info or {}), REMEMBER_JOB_KEY: snapshot}

    row = PipelineRun(
        status=None,
        pipeline_run_id=context.operation_id,
        pipeline_name=None,
        pipeline_id=None,
        dataset_id=context.dataset_id,
        run_info=run_info,
        user_id=context.user_id,
        tenant_id=context.tenant_id,
        operation_name=REMEMBER_OPERATION_NAME,
        started_at=context.started_at,
        ended_at=None,
        outcome=None,
        error_class=None,
        error_message=None,
        tokens_in=None,
        tokens_out=None,
        origin=get_operation_origin(),
        session_id=context.session_id,
        parent_operation_id=context.parent_operation_id,
        background=context.background,
    )

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        session.add(row)
        await session.commit()

    context.run_info = run_info


async def finish_job(context: "OperationContext", result) -> bool:
    """Store the terminal state of a job once; never raise.

    Only a row still without an outcome is updated, so a terminal state is
    never overwritten. A failed write is logged and leaves the stored state
    as it was (``running``): the in-memory result is never published as the
    job's outcome in its place. Returns whether the terminal state was stored.
    """
    from cognee.modules.pipelines.models import OperationOutcome, PipelineRun

    try:
        snapshot = build_job_snapshot(result, context.operation_id)
        failed = snapshot["status"] != JOB_COMPLETED
        run_info = {**(context.run_info or {}), REMEMBER_JOB_KEY: snapshot}
        values: dict[str, Any] = {
            "run_info": run_info,
            "ended_at": datetime.now(timezone.utc),
            "outcome": (OperationOutcome.FAILED if failed else OperationOutcome.SUCCEEDED).value,
            "error_class": snapshot.get("error_class") if failed else None,
            "error_message": snapshot.get("error") if failed else None,
            "tokens_in": context.usage.tokens_in,
            "tokens_out": context.usage.tokens_out,
        }

        db_engine = get_relational_engine()
        async with db_engine.get_async_session() as session:
            update_result = await session.execute(
                update(PipelineRun)
                .where(
                    PipelineRun.pipeline_run_id == context.operation_id,
                    PipelineRun.operation_name == REMEMBER_OPERATION_NAME,
                    PipelineRun.outcome.is_(None),
                )
                .values(**values)
            )
            await session.commit()
        if update_result.rowcount != 1:
            logger.warning(
                "remember job %s: terminal state not stored (%s rows matched)",
                context.operation_id,
                update_result.rowcount,
            )
            return False
        context.run_info = run_info
        return True
    except Exception:
        logger.exception("remember job %s: failed to store terminal state", context.operation_id)
        return False


async def get_job_snapshot(job_id: UUID | str, user: "User") -> dict | None:
    """The stored snapshot of ``job_id`` when it is a job of ``user``, else None.

    Scoped to the requesting user and tenant; another user's job, a legacy
    remember row without the job payload and an unknown id all read as None.
    Database errors propagate to the caller.
    """
    from cognee.modules.pipelines.models import PipelineRun

    job_id = job_id if isinstance(job_id, UUID) else UUID(str(job_id))
    tenant_id = getattr(user, "tenant_id", None)
    tenant_filter = (
        PipelineRun.tenant_id == tenant_id if tenant_id else PipelineRun.tenant_id.is_(None)
    )

    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        rows = (
            await session.execute(
                select(PipelineRun.run_info, PipelineRun.started_at, PipelineRun.outcome).where(
                    PipelineRun.pipeline_run_id == job_id,
                    PipelineRun.operation_name == REMEMBER_OPERATION_NAME,
                    PipelineRun.status.is_(None),
                    PipelineRun.user_id == user.id,
                    tenant_filter,
                )
            )
        ).fetchall()

    for run_info, started_at, outcome in rows:
        if isinstance(run_info, str):
            run_info = json.loads(run_info)
        payload = run_info.get(REMEMBER_JOB_KEY) if isinstance(run_info, dict) else None
        if not isinstance(payload, dict) or payload.get("version") != REMEMBER_JOB_VERSION:
            continue
        snapshot = dict(payload)
        snapshot.pop("version", None)
        if outcome is None:
            # No terminal state stored: whatever the payload says, the job is
            # "accepted, not finished" as far as the store knows.
            snapshot["status"] = JOB_RUNNING
            for key in ("error", "error_class", "error_http_status"):
                snapshot.pop(key, None)
            if started_at is not None:
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)
                snapshot["elapsed_seconds"] = (
                    datetime.now(timezone.utc) - started_at
                ).total_seconds()
        return snapshot
    return None


__all__ = [
    "JOB_COMPLETED",
    "JOB_ERRORED",
    "JOB_RUNNING",
    "REMEMBER_JOB_VERSION",
    "build_job_snapshot",
    "finish_job",
    "get_job_snapshot",
    "insert_running_job",
]

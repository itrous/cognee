"""Exception text never reaches a stored or served job snapshot unscrubbed (job status v1).

A failed run inside the improve result carries the exception's repr in its ``payload``, and
errors repeat per stage, per rerun pass, per memify run and per ingested item. The snapshot is
scrubbed as a whole, at any depth, while the payload of a run that succeeded is user data and
stays exactly as it was.
"""

import importlib
import json
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from cognee.api.v1.remember.remember import RememberResult
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.improve import ImproveResult, StageResult
from cognee.modules.pipelines.models import PipelineRun
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
)

from . import test_remember_job_status as job_tests
from .test_remember_job_status import _no_db_setup, _user, job_module, remember_module

stages = job_tests.stages

pytestmark = pytest.mark.asyncio

# Secret-shaped text that only ever sits in an exception text, in the three shapes the scrubber
# knows: an email address, a provider key and a home directory.
LEAKED_EMAIL = "eve@example.com"
LEAKED_KEY = "sk-abcdefghijklmnop1234"
LEAKED_HOME = "/home/eve/project"
LEAK = f"boom for {LEAKED_EMAIL} with {LEAKED_KEY} in {LEAKED_HOME}/data.txt"
LEAKS = (LEAKED_EMAIL, LEAKED_KEY, "/home/eve")

# User data of a run that succeeded. It looks like an exception text on purpose: only the
# status of the run, not the shape of the text, decides what is scrubbed.
USER_EMAIL = "carol@example.com"
SUCCESS_TEXT = f"stored the note for {USER_EMAIL} under /home/carol/notes.txt"
SUCCESS_PAYLOAD = [{"id": "d1", "note": f"write to {USER_EMAIL}", "path": "/home/carol/notes.txt"}]


def _errored_run(**kwargs):
    return PipelineRunErrored(
        pipeline_run_id=uuid4(),
        dataset_id=uuid4(),
        dataset_name="ds",
        payload=f"RuntimeError({LEAK!r})",
        error_class="RuntimeError",
        error_message=LEAK,
        **kwargs,
    )


def _completed_run(payload=SUCCESS_PAYLOAD):
    return PipelineRunCompleted(
        pipeline_run_id=uuid4(), dataset_id=uuid4(), dataset_name="ds", payload=payload
    )


def _stage(name, status, **fields):
    return StageResult(stage=name, status=status, **fields)


def _improve_with_leaks() -> ImproveResult:
    errored_stage = _stage("persist_session_qa", "errored", error=LEAK, run=_errored_run())
    # An errored run reported by a stage, with the leak also in an ingested item's run info.
    nested = _errored_run(
        data_ingestion_info=[{"run_info": {"status": "PipelineRunErrored", "payload": LEAK}}]
    )
    deep_stage = _stage("distill_sessions", "errored", error=LEAK, run=nested)
    ok_stage = _stage("feedback_weights", "completed", run=_completed_run())
    # A succeeded run whose payload is text, the same type the scrubber handles for a failed one.
    text_stage = _stage("update_user_preferences", "completed", run=_completed_run(SUCCESS_TEXT))
    return ImproveResult(
        stages=[errored_stage, ok_stage, deep_stage, text_stage],
        rerun_passes=[[_stage("persist_session_qa", "errored", error=LEAK, run=_errored_run())]],
        error=LEAK,
        memify_run={str(uuid4()): _errored_run()},
    )


def _result_with_improve(improve: ImproveResult) -> RememberResult:
    result = RememberResult(status="completed", dataset_name="ds")
    result.improve = improve
    result.improve_error = LEAK
    return result


def _assert_no_leak(snapshot: dict) -> None:
    text = json.dumps(snapshot, ensure_ascii=False)
    for leak in LEAKS:
        assert leak not in text, f"{leak!r} reached the job snapshot"


def _runs(snapshot: dict) -> list[dict]:
    improve = snapshot["improve"]
    passes = [improve["stages"], *improve["rerun_passes"]]
    found = [stage["run"] for stage_list in passes for stage in stage_list if stage.get("run")]
    found.extend(improve["memify_run"].values())
    return found


def test_snapshot_scrubs_every_exception_text_at_any_depth():
    snapshot = job_module.build_job_snapshot(_result_with_improve(_improve_with_leaks()), uuid4())

    _assert_no_leak(snapshot)
    improve = snapshot["improve"]
    # The fields are scrubbed, not dropped: an operator still sees that something failed.
    assert snapshot["improve_error"] and "[email]" in snapshot["improve_error"]
    assert improve["error"] and "[email]" in improve["error"]
    assert improve["stages"][0]["error"] and "[email]" in improve["stages"][0]["error"]
    assert improve["rerun_passes"][0][0]["error"] != LEAK
    errored = [run for run in _runs(snapshot) if run["status"] == "PipelineRunErrored"]
    assert len(errored) == 4  # stage, deep stage, rerun pass, memify run
    assert all("[email]" in run["payload"] for run in errored)
    deep = snapshot["improve"]["stages"][2]["run"]
    assert "[email]" in deep["data_ingestion_info"][0]["run_info"]["payload"]


def test_snapshot_keeps_the_payload_of_a_run_that_succeeded():
    snapshot = job_module.build_job_snapshot(_result_with_improve(_improve_with_leaks()), uuid4())

    completed = [run for run in _runs(snapshot) if run["status"] == "PipelineRunCompleted"]
    assert [run["payload"] for run in completed] == [SUCCESS_PAYLOAD, SUCCESS_TEXT]
    assert USER_EMAIL in json.dumps(snapshot)


def test_snapshot_scrubs_an_errored_payload_without_touching_empty_ones():
    improve = ImproveResult(
        stages=[
            _stage(
                "a", "errored", error=LEAK, run=_errored_run().model_copy(update={"payload": None})
            ),
            _stage("b", "completed"),
        ]
    )
    snapshot = job_module.build_job_snapshot(_result_with_improve(improve), uuid4())

    assert snapshot["improve"]["stages"][0]["run"]["payload"] is None
    assert snapshot["improve"]["stages"][1].get("run") is None
    _assert_no_leak(snapshot)


async def test_stored_and_served_job_carry_no_exception_text(stages, monkeypatch):
    # A stage that *returned* a failure (not raised): the improve result with its errored run is
    # stored on the job row and served by the job route, and must come out scrubbed from both.
    leaking_improve = _improve_with_leaks()

    async def improve(**kwargs):
        return leaking_improve

    monkeypatch.setattr(importlib.import_module("cognee.api.v1.improve"), "improve", improve)
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.remember.auto_improve_debounce"),
        "auto_improve_enabled",
        lambda: True,
    )
    owner = _user()
    result = await remember_module.remember(
        "note", dataset_id=uuid4(), user=owner, run_in_background=True
    )
    await result

    served = await job_module.get_job_snapshot(result.job_id, owner)
    assert served["status"] == "completed"
    _assert_no_leak(served)
    completed = [run for run in _runs(served) if run["status"] == "PipelineRunCompleted"]
    assert [run["payload"] for run in completed] == [SUCCESS_PAYLOAD, SUCCESS_TEXT]
    # The row itself, not only the reader, holds the scrubbed text.
    async with get_relational_engine().get_async_session() as session:
        rows = list(
            (
                await session.execute(
                    select(PipelineRun).where(PipelineRun.pipeline_run_id == UUID(result.job_id))
                )
            ).scalars()
        )
    assert len(rows) == 1
    _assert_no_leak(rows[0].run_info["remember_job"])

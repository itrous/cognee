import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j.exceptions import DatabaseUnavailable, Neo4jError

from cognee.infrastructure.databases.graph.neo4j_driver.deadlock_retry import deadlock_retry


@pytest.mark.asyncio
async def test_deadlock_retry_errored():
    mock_return = asyncio.Future()
    mock_return.set_result(True)
    mock_function = MagicMock(
        side_effect=[Neo4jError("DeadlockDetected"), Neo4jError("DeadlockDetected"), mock_return]
    )

    wrapped_function = deadlock_retry(max_retries=1)(mock_function)

    with pytest.raises(Neo4jError):
        await wrapped_function(self=None)


@pytest.mark.asyncio
async def test_deadlock_retry():
    mock_return = asyncio.Future()
    mock_return.set_result(True)
    mock_function = MagicMock(side_effect=[Neo4jError("DeadlockDetected"), mock_return])

    wrapped_function = deadlock_retry(max_retries=2)(mock_function)

    result = await wrapped_function(self=None)
    assert result, "Function should have succeded on second time"


@pytest.mark.asyncio
async def test_deadlock_retry_exhaustive():
    mock_return = asyncio.Future()
    mock_return.set_result(True)
    mock_function = MagicMock(
        side_effect=[Neo4jError("DeadlockDetected"), Neo4jError("DeadlockDetected"), mock_return]
    )

    wrapped_function = deadlock_retry(max_retries=2)(mock_function)

    result = await wrapped_function(self=None)
    assert result, "Function should have succeded on second time"


@pytest.mark.asyncio
async def test_database_unavailable_retry_succeeds():
    """DatabaseUnavailable should retry and succeed within max_retries."""
    mock_function = AsyncMock(side_effect=[DatabaseUnavailable(), True])

    wrapped_function = deadlock_retry(max_retries=2)(mock_function)

    result = await wrapped_function(self=None)
    assert result, "Function should have succeeded after one DatabaseUnavailable retry"


@pytest.mark.asyncio
async def test_database_unavailable_exhausts_retries():
    """DatabaseUnavailable should re-raise once max_retries is exhausted."""
    mock_function = AsyncMock(side_effect=[DatabaseUnavailable(), DatabaseUnavailable()])

    wrapped_function = deadlock_retry(max_retries=1)(mock_function)

    with pytest.raises(DatabaseUnavailable):
        await wrapped_function(self=None)


@pytest.mark.asyncio
async def test_database_unavailable_retries_match_neo4j_error():
    """DatabaseUnavailable and Neo4jError should both allow exactly max_retries attempts.

    This test directly validates the fix from issue #3757: the DatabaseUnavailable
    handler previously used >= max_retries (re-raising one attempt too early),
    while Neo4jError used > max_retries. Both now use > max_retries.
    """
    # With max_retries=2, we should be able to fail twice and succeed on the 3rd attempt.
    mock_function = AsyncMock(side_effect=[DatabaseUnavailable(), DatabaseUnavailable(), True])

    wrapped_function = deadlock_retry(max_retries=2)(mock_function)

    result = await wrapped_function(self=None)
    assert result, "Function should have succeeded on the last allowed retry attempt"


if __name__ == "__main__":

    async def main():
        await test_deadlock_retry()
        await test_deadlock_retry_errored()
        await test_deadlock_retry_exhaustive()
        await test_database_unavailable_retry_succeeds()
        await test_database_unavailable_exhausts_retries()
        await test_database_unavailable_retries_match_neo4j_error()

    asyncio.run(main())


@pytest.mark.asyncio
async def test_memory_pool_out_of_memory_is_not_retried():
    """MemoryPoolOutOfMemoryError arrives as Neo.TransientError.General.*, but it
    is deterministic: retrying the identical query re-allocates the same
    transaction memory, so each retry only amplifies the memory pressure."""
    mock_function = AsyncMock(
        side_effect=Neo4jError(
            "Neo.TransientError.General.MemoryPoolOutOfMemoryError: "
            "dbms.memory.transaction.total.max threshold reached"
        )
    )

    wrapped_function = deadlock_retry(max_retries=5)(mock_function)

    with pytest.raises(Neo4jError, match="MemoryPoolOutOfMemoryError"):
        await wrapped_function(self=None)

    assert mock_function.await_count == 1, "the OOM error must not be retried"


# --- Connection retry opt-in (N1 / N2) ---------------------------------------
# Input that tells the variants apart: ServiceUnavailable / SessionExpired (not
# Neo4jError subclasses, is_retryable() True) followed by a success. The default
# decorator re-raises them after one call; only the opt-in retries them.

from types import SimpleNamespace  # noqa: E402

from neo4j.exceptions import (  # noqa: E402
    AuthError,
    CypherSyntaxError,
    IncompleteCommit,
    ServiceUnavailable,
    SessionExpired,
)

import cognee.infrastructure.databases.graph.neo4j_driver.deadlock_retry as deadlock_retry_module  # noqa: E402


@pytest.fixture
def recorded_sleep(monkeypatch):
    """Backoff waits are recorded, never slept."""
    sleep = AsyncMock()
    monkeypatch.setattr(deadlock_retry_module, "asyncio", SimpleNamespace(sleep=sleep))
    return sleep


@pytest.mark.asyncio
@pytest.mark.parametrize("error_class", [ServiceUnavailable, SessionExpired])
async def test_connection_retry_opt_in(recorded_sleep, error_class):
    opted_in = AsyncMock(side_effect=[error_class("probe"), "ok"])
    result = await deadlock_retry(max_retries=1, retry_connection_errors=True)(opted_in)(None)
    assert result == "ok"
    assert opted_in.await_count == 2
    assert recorded_sleep.await_count == 1

    # Control: the same input under the default decorator is one call and the error.
    default = AsyncMock(side_effect=[error_class("probe"), "ok"])
    with pytest.raises(error_class):
        await deadlock_retry(max_retries=1)(default)(None)
    assert default.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        IncompleteCommit("commit outcome unknown"),
        AuthError("unauthorized"),
        CypherSyntaxError("bad syntax"),
        Neo4jError(
            "Neo.TransientError.General.MemoryPoolOutOfMemoryError: "
            "dbms.memory.transaction.total.max threshold reached"
        ),
        asyncio.CancelledError(),
    ],
    ids=["incomplete-commit", "auth", "syntax", "oom", "cancelled"],
)
async def test_nonretryable_connection_outcomes(recorded_sleep, error):
    # IncompleteCommit is a ServiceUnavailable subclass: a blanket
    # ``except ServiceUnavailable: retry`` would replay a possibly committed write.
    assert not isinstance(error, IncompleteCommit) or isinstance(error, ServiceUnavailable)
    function = AsyncMock(side_effect=[error, "ok"])

    with pytest.raises(type(error)) as raised:
        await deadlock_retry(max_retries=5, retry_connection_errors=True)(function)(None)

    assert raised.value is error
    assert function.await_count == 1
    recorded_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_nonretryable_service_unavailable_is_not_retried(recorded_sleep, monkeypatch):
    error = ServiceUnavailable("routing table says no")
    monkeypatch.setattr(error, "is_retryable", lambda: False)
    function = AsyncMock(side_effect=[error, "ok"])

    with pytest.raises(ServiceUnavailable):
        await deadlock_retry(max_retries=5, retry_connection_errors=True)(function)(None)
    assert function.await_count == 1


@pytest.mark.asyncio
async def test_connection_budget_exhaustion(recorded_sleep):
    """Ten retries after the first attempt: 11 calls, then the last original error."""
    errors = [SessionExpired(f"attempt {index}") for index in range(11)]
    function = AsyncMock(side_effect=[*errors, "never reached"])

    with pytest.raises(SessionExpired) as raised:
        await deadlock_retry(retry_connection_errors=True)(function)(None)

    assert raised.value is errors[-1]
    assert function.await_count == 11
    assert recorded_sleep.await_count == 10

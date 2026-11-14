import asyncio
from functools import wraps

from cognee.infrastructure.utils.calculate_backoff import calculate_backoff
from cognee.shared.logging_utils import get_logger

logger = get_logger("deadlock_retry")


def deadlock_retry(max_retries=10, retry_connection_errors=False):
    """
    Decorator that automatically retries an asynchronous function when Neo4j deadlocks,
    transient errors, or database unavailability occur.

    This decorator implements an exponential backoff strategy with jitter
    to handle these errors efficiently.

    Args:
        max_retries: Maximum number of retry attempts (default is 10).
        retry_connection_errors: Also retry a lost connection (``ServiceUnavailable``,
            ``SessionExpired`` whose ``is_retryable()`` is true). Only for callables
            that redo one whole safe read or one idempotent write per attempt.
            ``IncompleteCommit`` is never retried: its commit outcome is unknown.

    Returns:
        The decorated async function.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            from neo4j.exceptions import (
                DatabaseUnavailable,
                IncompleteCommit,
                Neo4jError,
                ServiceUnavailable,
                SessionExpired,
            )

            attempt = 0

            async def wait(error):
                backoff_time = calculate_backoff(attempt)
                logger.warning(
                    f"Neo4j {type(error).__name__}, retrying in {backoff_time:.2f}s "
                    f"Attempt {attempt}/{max_retries}"
                )
                await asyncio.sleep(backoff_time)

            while attempt <= max_retries:
                try:
                    attempt += 1
                    return await func(self, *args, **kwargs)
                except IncompleteCommit:
                    raise  # The commit may have landed: replaying could apply it twice.
                except (ServiceUnavailable, SessionExpired) as error:
                    if not retry_connection_errors or not error.is_retryable():
                        raise
                    if attempt > max_retries:
                        raise  # Re-raise the original error

                    await wait(error)
                except DatabaseUnavailable as error:
                    if attempt > max_retries:
                        raise  # Re-raise the original error

                    await wait(error)
                except Neo4jError as error:
                    if attempt > max_retries:
                        raise  # Re-raise the original error

                    error_str = str(error)
                    if "MemoryPoolOutOfMemoryError" in error_str:
                        # Deterministic: retrying the identical query re-allocates
                        # the same transaction memory and fails the same way, so
                        # each retry only amplifies the memory pressure.
                        raise
                    if "DeadlockDetected" in error_str or "Neo.TransientError" in error_str:
                        await wait(error)
                    else:
                        raise  # Re-raise the original error

        return wrapper

    return decorator

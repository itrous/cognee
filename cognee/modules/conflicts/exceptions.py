from fastapi import status

from cognee.exceptions import CogneeApiError


class ConflictAlreadyResolvedError(CogneeApiError):
    """A conditional choice found the conflict already resolved and wrote nothing."""

    def __init__(
        self,
        message: str = "Conflict is already resolved; the choice was not recorded.",
        name: str = "ConflictAlreadyResolvedError",
        status_code=status.HTTP_409_CONFLICT,
    ):
        # An expected outcome of a conditional choice, not a server fault.
        super().__init__(message, name, status_code, log_level="INFO")

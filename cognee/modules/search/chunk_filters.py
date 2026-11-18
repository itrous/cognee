"""The server-side filters of an explicit CHUNKS search: one validation for every entry point."""

from numbers import Real

from cognee.exceptions import CogneeValidationError
from cognee.modules.search.types import SearchType


def chunk_filter_config(
    query_type: SearchType, min_similarity=None, exclude_superseded=False
) -> dict:
    """Validate ``min_similarity`` / ``exclude_superseded``; return the config keys they set.

    Unset means ``None`` / ``False``, and an unset filter adds no key, so a search
    without them reaches ``ChunksRetriever`` exactly as before. A set filter is only
    valid for an explicit CHUNKS search, ``min_similarity=0`` included.
    """
    # The range check also rejects NaN and infinities, and compares a huge int
    # without converting it to float.
    if min_similarity is not None and (
        isinstance(min_similarity, bool)
        or not isinstance(min_similarity, Real)
        or not 0 <= min_similarity <= 1
    ):
        raise CogneeValidationError(
            message="min_similarity must be a finite number from 0 to 1.",
            name="InvalidMinSimilarity",
        )
    if not isinstance(exclude_superseded, bool):
        raise CogneeValidationError(
            message="exclude_superseded must be a boolean.",
            name="InvalidExcludeSuperseded",
        )
    config = {}
    if min_similarity is not None:
        config["min_similarity"] = float(min_similarity)
    if exclude_superseded:
        config["exclude_superseded"] = True
    if config and query_type is not SearchType.CHUNKS:
        raise CogneeValidationError(
            message="min_similarity and exclude_superseded require query_type=SearchType.CHUNKS.",
            name="InvalidChunkFilterConfig",
        )
    return config

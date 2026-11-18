"""chunk_filter_config and its wiring through the retriever factory."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.retrieval.chunks_retriever import ChunksRetriever
from cognee.modules.search.chunk_filters import chunk_filter_config
from cognee.modules.search.methods.get_search_type_retriever_instance import (
    get_search_type_retriever_instance,
)
from cognee.modules.search.types import SearchType


@pytest.mark.parametrize(
    ("min_similarity", "exclude_superseded", "expected"),
    [
        (None, False, {}),
        (0, False, {"min_similarity": 0.0}),
        (1, False, {"min_similarity": 1.0}),
        (0.6, True, {"min_similarity": 0.6, "exclude_superseded": True}),
        (None, True, {"exclude_superseded": True}),
    ],
)
def test_valid_filters_for_chunks(min_similarity, exclude_superseded, expected):
    assert chunk_filter_config(SearchType.CHUNKS, min_similarity, exclude_superseded) == expected


@pytest.mark.parametrize(
    "min_similarity",
    [float("nan"), float("inf"), -float("inf"), -0.01, 1.01, 10**400, "0.5", True, [0.5]],
)
def test_invalid_threshold_is_rejected(min_similarity):
    with pytest.raises(CogneeValidationError, match="min_similarity"):
        chunk_filter_config(SearchType.CHUNKS, min_similarity, False)


@pytest.mark.parametrize("exclude_superseded", ["true", 1, None])
def test_non_boolean_flag_is_rejected(exclude_superseded):
    with pytest.raises(CogneeValidationError, match="exclude_superseded"):
        chunk_filter_config(SearchType.CHUNKS, None, exclude_superseded)


@pytest.mark.parametrize(
    "query_type",
    [SearchType.HYBRID_COMPLETION, SearchType.RAG_COMPLETION, SearchType.CHUNKS_LEXICAL, None],
)
@pytest.mark.parametrize(("min_similarity", "exclude_superseded"), [(0, False), (None, True)])
def test_set_filters_need_an_explicit_chunks_search(query_type, min_similarity, exclude_superseded):
    with pytest.raises(CogneeValidationError, match="SearchType.CHUNKS"):
        chunk_filter_config(query_type, min_similarity, exclude_superseded)


def test_unset_filters_are_valid_for_any_search_type():
    assert chunk_filter_config(SearchType.GRAPH_COMPLETION, None, False) == {}
    assert chunk_filter_config(None) == {}


@pytest.mark.asyncio
async def test_factory_passes_filters_and_the_explicit_dataset_id_to_chunks():
    dataset = SimpleNamespace(id=uuid4(), name="ds", tenant_id=None)
    retriever = await get_search_type_retriever_instance(
        SearchType.CHUNKS,
        "q",
        top_k=4,
        dataset=dataset,
        retriever_specific_config={"min_similarity": 0.6, "exclude_superseded": True},
    )

    assert isinstance(retriever, ChunksRetriever)
    assert (retriever.top_k, retriever.min_similarity, retriever.exclude_superseded) == (
        4,
        0.6,
        True,
    )
    assert retriever.dataset_id == dataset.id


@pytest.mark.asyncio
async def test_factory_without_filters_builds_the_unfiltered_retriever():
    retriever = await get_search_type_retriever_instance(SearchType.CHUNKS, "q", top_k=4)

    assert retriever.min_similarity is None
    assert retriever.exclude_superseded is False


@pytest.mark.asyncio
async def test_factory_rejects_superseded_filter_without_a_dataset():
    with pytest.raises(CogneeValidationError, match="one dataset"):
        await get_search_type_retriever_instance(
            SearchType.CHUNKS, "q", retriever_specific_config={"exclude_superseded": True}
        )


@pytest.mark.asyncio
async def test_factory_rejects_filters_for_another_search_type():
    with pytest.raises(CogneeValidationError, match="SearchType.CHUNKS"):
        await get_search_type_retriever_instance(
            SearchType.SUMMARIES, "q", retriever_specific_config={"min_similarity": 0.5}
        )

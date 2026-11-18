"""ChunksRetriever's opt-in server-side filters: min_similarity and exclude_superseded.

Each test's vector store is a fixed list of hits in distance order; ``search``
returns its first ``limit`` rows, as the real adapters do, so widening is visible
as the sequence of limits the retriever asked for.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.modules.retrieval.chunks_retriever import ChunksRetriever, passes_min_similarity
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError

MODULE = "cognee.modules.retrieval.chunks_retriever"
DATASET_ID = uuid4()


def hit(name: str, score):
    return SimpleNamespace(id=name, score=score, payload={"id": name, "text": f"text {name}"})


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.limits = []
        self.search = AsyncMock(side_effect=self._search)
        self.embedding_engine = SimpleNamespace(embed_text=AsyncMock(return_value=[[0.1, 0.2]]))

    async def _search(self, collection, query_text, query_vector=None, limit=None, **kwargs):
        self.limits.append(limit)
        return list(self.hits if limit is None else self.hits[:limit])


def unified_for(store):
    return SimpleNamespace(vector=store, graph=SimpleNamespace(name="graph"))


async def retrieve(retriever, store, superseded=None):
    lookup = AsyncMock(side_effect=lambda graph, dataset_id, ids: set(superseded or ()) & set(ids))
    with (
        patch(f"{MODULE}.get_unified_engine", AsyncMock(return_value=unified_for(store))),
        patch(f"{MODULE}.get_superseded_chunk_ids", lookup),
        patch(f"{MODULE}.attach_chunk_conflicts", AsyncMock()),
    ):
        found = await retriever.get_retrieved_objects("q")
    return [chunk.id for chunk in found], lookup


@pytest.mark.parametrize(
    ("score", "threshold", "passes"),
    [
        (0.25, 0.75, True),  # equality passes
        (0.2500001, 0.75, False),
        (0.5, 0.5, True),
        (-0.3, 1.0, True),  # clamped to 0: similarity 1
        (2.5, 0.0, False),  # clamped to 2: similarity -1
        (2.0, 0.0, False),
        (1.0, 0.0, True),
        (None, 0.0, False),  # a missing distance never passes, not even threshold 0
        (float("nan"), 0.0, False),
        (float("inf"), 0.0, False),
        ("0.1", 0.0, False),
        (True, 0.0, False),
    ],
)
def test_threshold_formula(score, threshold, passes):
    assert passes_min_similarity(score, threshold) is passes


def test_exclude_superseded_needs_an_explicit_dataset():
    with pytest.raises(QueryValidationError):
        ChunksRetriever(top_k=3, exclude_superseded=True, dataset_id=None)


@pytest.mark.asyncio
async def test_without_filters_the_old_single_query_runs_and_no_status_is_read():
    store = FakeStore([hit("a", 0.9), hit("b", 0.95)])
    ids, lookup = await retrieve(ChunksRetriever(top_k=5, dataset_id=DATASET_ID), store)

    assert ids == ["a", "b"]
    store.search.assert_awaited_once_with(
        "DocumentChunk_text",
        "q",
        limit=5,
        include_payload=True,
        node_name=None,
        node_name_filter_operator="OR",
    )
    store.embedding_engine.embed_text.assert_not_awaited()
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_threshold_keeps_boundary_drops_farther_and_stops_at_the_first_failing_tail():
    store = FakeStore([hit("a", 0.1), hit("b", 0.5), hit("c", 0.5000001), hit("d", 0.7)])
    ids, _ = await retrieve(ChunksRetriever(top_k=4, min_similarity=0.5), store)

    assert ids == ["a", "b"]
    # The farthest of the first window already fails, so nothing farther is fetched.
    assert store.limits == [4]
    # Control: the same store without the threshold returns the low-similarity chunks.
    unfiltered, _ = await retrieve(ChunksRetriever(top_k=4), FakeStore(store.hits))
    assert unfiltered == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_non_numeric_distance_is_dropped_and_top_k_is_refilled():
    store = FakeStore([hit("a", None), hit("b", 0.1), hit("c", "x"), hit("d", 0.2), hit("e", 0.3)])
    ids, _ = await retrieve(ChunksRetriever(top_k=2, min_similarity=0.0), store)

    assert ids == ["b", "d"]
    assert store.limits == [2, 4]
    # One embedding for every round; the store is searched by vector, not text.
    store.embedding_engine.embed_text.assert_awaited_once_with(["q"])
    assert store.search.await_args.args[1] is None
    assert store.search.await_args.kwargs["query_vector"] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_superseded_chunks_are_excluded_and_the_next_nearest_fill_their_places():
    store = FakeStore([hit(name, 0.1 * i) for i, name in enumerate("abcdefg")])
    retriever = ChunksRetriever(top_k=3, exclude_superseded=True, dataset_id=DATASET_ID)
    ids, lookup = await retrieve(retriever, store, superseded={"a", "c", "d"})

    assert ids == ["b", "e", "f"]
    assert store.limits == [3, 6]
    # Each chunk's status is read once, with the explicit dataset id.
    assert [call.args[2] for call in lookup.await_args_list] == [["a", "b", "c"], ["d", "e", "f"]]
    assert all(call.args[1] == DATASET_ID for call in lookup.await_args_list)
    # Control: without the flag the superseded chunks are returned.
    unfiltered, _ = await retrieve(ChunksRetriever(top_k=3), FakeStore(store.hits), {"a"})
    assert unfiltered == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_widening_stops_when_the_store_is_exhausted():
    store = FakeStore([hit("a", 0.1), hit("b", 0.2), hit("c", 0.3)])
    retriever = ChunksRetriever(top_k=2, exclude_superseded=True, dataset_id=DATASET_ID)
    ids, _ = await retrieve(retriever, store, superseded={"a", "b"})

    assert ids == ["c"]
    assert store.limits == [2, 4]


@pytest.mark.asyncio
async def test_both_filters_apply_together_and_keep_distance_order():
    store = FakeStore([hit("a", 0.1), hit("b", 0.2), hit("c", 0.3), hit("d", 0.6)])
    retriever = ChunksRetriever(
        top_k=3, min_similarity=0.5, exclude_superseded=True, dataset_id=DATASET_ID
    )
    ids, lookup = await retrieve(retriever, store, superseded={"b"})

    assert ids == ["a", "c"]
    # Only chunks that pass the threshold have their facts read.
    assert lookup.await_args_list[0].args[2] == ["a", "b", "c"]
    assert store.limits == [3, 6]


@pytest.mark.asyncio
async def test_duplicate_hits_count_once():
    store = FakeStore([hit("a", 0.1), hit("a", 0.1), hit("b", 0.2), hit("c", 0.3)])
    ids, _ = await retrieve(ChunksRetriever(top_k=2, min_similarity=0.0), store)

    assert ids == ["a", "b"]


@pytest.mark.asyncio
async def test_top_k_none_reads_every_candidate_once():
    store = FakeStore([hit("a", 0.1), hit("b", 0.9)])
    ids, _ = await retrieve(ChunksRetriever(top_k=None, min_similarity=0.5), store)

    assert ids == ["a"]
    assert store.limits == [None]


@pytest.mark.asyncio
async def test_a_failed_status_read_fails_the_search():
    store = FakeStore([hit("a", 0.1)])
    retriever = ChunksRetriever(top_k=1, exclude_superseded=True, dataset_id=DATASET_ID)
    failing = AsyncMock(side_effect=RuntimeError("graph down"))
    with (
        patch(f"{MODULE}.get_unified_engine", AsyncMock(return_value=unified_for(store))),
        patch(f"{MODULE}.get_superseded_chunk_ids", failing),
        patch(f"{MODULE}.attach_chunk_conflicts", AsyncMock()),
        pytest.raises(RuntimeError, match="graph down"),
    ):
        await retriever.get_retrieved_objects("q")


@pytest.mark.asyncio
async def test_node_filters_ride_every_widening_round():
    store = FakeStore([hit("a", None), hit("b", None), hit("c", 0.1)])
    retriever = ChunksRetriever(
        top_k=1, node_name=["proj"], node_name_filter_operator="AND", min_similarity=0.1
    )
    ids, _ = await retrieve(retriever, store)

    assert ids == ["c"]
    assert store.limits == [1, 2, 4]
    for call in store.search.await_args_list:
        assert call.kwargs["node_name"] == ["proj"]
        assert call.kwargs["node_name_filter_operator"] == "AND"


@pytest.mark.asyncio
async def test_filtered_payload_and_score_are_unchanged():
    store = FakeStore([hit("a", 0.25)])
    retriever = ChunksRetriever(top_k=1, min_similarity=0.75)
    with (
        patch(f"{MODULE}.get_unified_engine", AsyncMock(return_value=unified_for(store))),
        patch(f"{MODULE}.attach_chunk_conflicts", AsyncMock()),
    ):
        found = await retriever.get_retrieved_objects("q")
    completion = await retriever.get_completion_from_context("q", found, None)

    assert completion == [{"id": "a", "text": "text a", "score": 0.25}]

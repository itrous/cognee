"""Second client of the EdgeType lifecycle integration tests; not a test module.

Imported by ``test_edge_type_lifecycle.py`` for the shared fixed-embedding helpers and
run by it as ``python edge_type_lifecycle_peer.py '<json job>'``: a separate process
with its own engines, pools and lock connection against the same Neo4j + PGVector
pair. It reports through stdout lines prefixed ``PEER:`` so the parent orders its
steps by what the peer says it did, never by elapsed time:

``PEER:ATTEMPT``  printed right before the lifecycle lock is requested
``PEER:DONE``     the write returned
``PEER:ERROR ...`` the write raised
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import sys

DIMENSIONS = 2048


def text_vector(text: str, dimensions: int = DIMENSIONS) -> list[float]:
    """Distinct, non-zero, reproducible direction per text (cosine search works on it)."""
    rng = random.Random(text)
    vector = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


class EmbedLog:
    """Every text the fixed embedding function was asked for."""

    def __init__(self):
        self.texts: list[str] = []


def install_fixed_embeddings(setattr_fn, log: EmbedLog, fail_on: set[str] | None = None) -> None:
    """Replace the embedding call; ``setattr_fn(obj, name, value)`` is monkeypatch.setattr or setattr."""
    from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
    from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
        LiteLLMEmbeddingEngine,
    )

    async def embed_text(self, text):
        if fail_on and any(item in fail_on for item in text):
            raise RuntimeError("injected embedding failure")
        log.texts.extend(text)
        return EmbeddingResult(
            vectors=[text_vector(item, self.dimensions or DIMENSIONS) for item in text],
            tokens=0,
        )

    setattr_fn(LiteLLMEmbeddingEngine, "embed_text", embed_text)


def say(line: str) -> None:
    print("PEER:" + line, flush=True)


async def run(job: dict) -> None:
    import cognee.tasks.storage.edge_type_lifecycle as lifecycle
    from cognee.modules.engine.models.Entity import Entity
    from cognee.tasks.storage import add_data_points

    install_fixed_embeddings(setattr, EmbedLog())

    if job["no_lock"]:

        class NoLock:
            """The control: the same interleaving without any mutual exclusion."""

            def __init__(self, vector_adapter):
                pass

            async def acquire(self):
                say("ATTEMPT")

            async def ensure_held(self):
                pass

            async def release(self):
                pass

        lifecycle._AdvisoryLock = NoLock
    else:
        acquire = lifecycle._AdvisoryLock.acquire

        async def announced_acquire(self):
            say("ATTEMPT")
            await acquire(self)

        lifecycle._AdvisoryLock.acquire = announced_acquire

    nodes = [Entity(name=name, description=f"{name} is an entity") for name in job["entities"]]
    ids = {node.name: str(node.id) for node in nodes}
    source, target, relationship, text = job["edge"]
    await add_data_points(
        nodes, custom_edges=[(ids[source], ids[target], relationship, {"edge_text": text})]
    )
    say("DONE")


def main() -> int:
    job = json.loads(sys.argv[1])
    try:
        asyncio.run(run(job))
    except Exception as error:  # noqa: BLE001 - the parent must see why the peer stopped
        say(f"ERROR {type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

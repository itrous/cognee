from types import SimpleNamespace
from uuid import uuid4

import pytest


@pytest.fixture
def stub_document_preflight(monkeypatch):
    """Skip the dataset resolution and name-conflict check a background
    document remember runs before acknowledging; they need real datasets."""

    async def _resolve(dataset_name=None, dataset_id=None, user=None):
        return user, SimpleNamespace(id=dataset_id or uuid4(), name=dataset_name or "ds")

    async def _no_conflicts(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "cognee.modules.pipelines.layers.resolve_authorized_user_dataset"
        ".resolve_authorized_user_dataset",
        _resolve,
    )
    monkeypatch.setattr(
        "cognee.tasks.ingestion.refuse_changed_existing_documents"
        ".refuse_changed_existing_documents",
        _no_conflicts,
    )

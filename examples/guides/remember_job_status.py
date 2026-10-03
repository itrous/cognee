"""Remember in the background over HTTP and wait for the outcome of that one write.

A background ``POST /api/v1/remember`` is answered with 202 and a ``job_id`` once the
input, permissions and name conflicts are checked. ``GET /api/v1/remember/jobs/{job_id}``
then reports ``running`` until the add + cognify of this write finished, and then
``completed`` or ``errored`` (with ``error``, ``error_class`` and ``error_http_status``).
A failed write is a 200 on the job endpoint: read ``status``, not the HTTP code.

Prerequisites:
  * A running Cognee API: uv run python -m cognee.api.client
  * COGNEE_API_URL (default http://localhost:8000) and, when the server requires
    authentication, COGNEE_API_KEY (sent as X-Api-Key).

Run: uv run python examples/guides/remember_job_status.py
"""

import asyncio
import os
import time

import httpx

API_URL = os.environ.get("COGNEE_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("COGNEE_API_KEY")
DEADLINE_SECONDS = 120


async def main():
    headers = {"X-Api-Key": API_KEY} if API_KEY else {}
    async with httpx.AsyncClient(base_url=API_URL, headers=headers, timeout=30) as client:
        capabilities = await client.get("/api/v1/remember/capabilities")
        if capabilities.status_code != 200 or (
            capabilities.json().get("document_job_status_version") != 1
        ):
            print("This server has no remember job status; use a blocking remember instead.")
            return

        accepted = await client.post(
            "/api/v1/remember",
            data={"datasetName": "job_status_demo", "run_in_background": "true"},
            files={"data": ("einstein.txt", b"Einstein was born in Ulm.", "text/plain")},
        )
        accepted.raise_for_status()  # a name conflict is a 409 here, before any job exists
        job_id = accepted.json()["job_id"]
        print(f"accepted ({accepted.status_code}), job {job_id}")

        deadline = time.monotonic() + DEADLINE_SECONDS
        while time.monotonic() < deadline:
            job = (await client.get(f"/api/v1/remember/jobs/{job_id}")).json()
            if job["status"] != "running":
                break
            await asyncio.sleep(1)
        else:
            print("still running when the deadline passed; the job may finish later")
            return

        if job["status"] == "completed":
            print(
                f"completed: {job['items_processed']} item(s), cognify run {job['pipeline_run_id']}"
            )
        else:
            print(f"errored ({job['error_class']}, {job['error_http_status']}): {job['error']}")


if __name__ == "__main__":
    asyncio.run(main())

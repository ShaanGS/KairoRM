from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from ingestion.types import Chunk, CompressedContext, Ok, RankedChunk
from output import qa_server


def _compressed() -> CompressedContext:
    return CompressedContext(content="repo analysis here", token_count=4, truncated=False)


def _chunk() -> RankedChunk:
    c = Chunk(
        chunk_id="id_a",
        file_path=Path("/repo/auth.py"),
        language="python",
        unit_type="function",
        name="verify",
        start_line=1,
        end_line=3,
        content="def verify(): ...",
        token_count=6,
        imports=(),
        calls=(),
        context_header="# python | function verify",
    )
    return RankedChunk(chunk=c, importance_score=0.9)


def test_ask_valid_question_returns_answer() -> None:
    app = qa_server.create_app(
        _compressed(), repo_name="myrepo", repo_id="r" * 64, db_path=Path("/tmp/db")
    )
    with (
        patch.object(
            qa_server.retriever, "retrieve", new=AsyncMock(return_value=Ok([_chunk()]))
        ),
        patch.object(
            qa_server, "_complete_text", new=AsyncMock(return_value=Ok("Auth verifies tokens."))
        ),
    ):
        client = TestClient(app)
        resp = client.post("/ask", json={"question": "how does auth work"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Auth verifies tokens."
    assert body["chunks_used"] == 1


def test_ask_empty_question_returns_422() -> None:
    app = qa_server.create_app(_compressed(), repo_name="myrepo")
    client = TestClient(app)
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_missing_question_returns_422() -> None:
    app = qa_server.create_app(_compressed(), repo_name="myrepo")
    client = TestClient(app)
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_without_store_skips_retrieval() -> None:
    # No repo_id/db_path → retrieval is skipped, chunks_used == 0, still answers.
    app = qa_server.create_app(_compressed(), repo_name="myrepo")
    retrieve_mock = AsyncMock(return_value=Ok([_chunk()]))
    with (
        patch.object(qa_server.retriever, "retrieve", new=retrieve_mock),
        patch.object(qa_server, "_complete_text", new=AsyncMock(return_value=Ok("answer"))),
    ):
        client = TestClient(app)
        resp = client.post("/ask", json={"question": "anything"})

    assert resp.status_code == 200
    assert resp.json()["chunks_used"] == 0
    retrieve_mock.assert_not_called()

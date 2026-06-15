from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from ingestion.types import (
    Chunk,
    CompressedContext,
    Ok,
    RankedChunk,
    SynthesisEntryPoint,
    SynthesisModule,
    SynthesisResult,
)
from output import qa_server


def _compressed() -> CompressedContext:
    return CompressedContext(content="repo analysis here", token_count=4, truncated=False)


def _result() -> SynthesisResult:
    return SynthesisResult(
        repo_id="r" * 64,
        architecture_summary="A layered pipeline. Each stage feeds the next.",
        modules=[
            SynthesisModule(
                name="ingestion", path="/repo/ingestion", responsibility="Fetches code."
            )
        ],
        key_dependencies=["click", "rich"],
        circular_risks=["auth -> db -> auth"],
        entry_points=[
            SynthesisEntryPoint(name="main", file="cli/main.py", description="Entry point.")
        ],
        contributor_quickstart=["Clone the repo", "Run the tests"],
        complexity_score=7,
        generated_at=datetime(2026, 6, 15, tzinfo=UTC),
    )


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
        patch.object(qa_server.retriever, "retrieve", new=AsyncMock(return_value=Ok([_chunk()]))),
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


def test_report_route_serves_html_with_result() -> None:
    app = qa_server.create_app(
        _compressed(),
        repo_name="myrepo",
        result=_result(),
        stats={"files": 12, "chunks": 30, "languages": {"python": 12}},
    )
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    # Repo data made it into the injected payload.
    assert "myrepo" in body
    assert "ingestion" in body
    assert "auth -> db -> auth" in body
    # The data placeholder was replaced, not left raw.
    assert "__KAIRO_DATA__" not in body


def test_report_route_absent_without_result() -> None:
    # No result → no report page is mounted, only the /ask API exists.
    app = qa_server.create_app(_compressed(), repo_name="myrepo")
    client = TestClient(app)
    assert client.get("/").status_code == 404

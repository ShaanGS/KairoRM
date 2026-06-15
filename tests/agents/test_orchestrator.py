from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agents import base, orchestrator
from agents.orchestrator import run_all
from indexing.retriever import RetrieveError
from ingestion.types import AgentOutputs, Chunk, Err, Ok, RankedChunk


def _chunk(name: str = "a", importance: float = 0.5) -> RankedChunk:
    c = Chunk(
        chunk_id=f"id_{name}",
        file_path=Path(f"/repo/{name}.py"),
        language="python",
        unit_type="function",
        name=name,
        start_line=1,
        end_line=2,
        content="def a(): pass",
        token_count=4,
        imports=(),
        calls=(),
        context_header="# python | function a",
    )
    return RankedChunk(chunk=c, importance_score=importance)


@pytest.mark.asyncio
async def test_all_agents_succeed_no_none_fields(tmp_path: Path) -> None:
    async def fake_retrieve(query, *, repo_id, db_path):  # noqa: ANN001, ANN202
        return Ok([_chunk()])

    async def fake_run(self, chunks):  # noqa: ANN001, ANN202
        return Ok({self.name: "ok"})

    with (
        patch.object(orchestrator.retriever, "retrieve", new=fake_retrieve),
        patch.object(base.BaseAgent, "run", new=fake_run),
    ):
        result = await run_all([_chunk()], repo_id="r" * 64, db_path=tmp_path)

    assert result.is_ok()
    out = result.unwrap()
    assert isinstance(out, AgentOutputs)
    assert out.modules == {"module": "ok"}
    assert out.arch == {"arch": "ok"}
    assert out.deps == {"deps": "ok"}
    assert out.contributor == {"contributor": "ok"}


@pytest.mark.asyncio
async def test_one_agent_raises_only_that_field_is_none(tmp_path: Path) -> None:
    async def fake_retrieve(query, *, repo_id, db_path):  # noqa: ANN001, ANN202
        return Ok([_chunk()])

    async def fake_run(self, chunks):  # noqa: ANN001, ANN202
        if self.name == "arch":
            raise RuntimeError("arch agent exploded")
        return Ok({self.name: "ok"})

    with (
        patch.object(orchestrator.retriever, "retrieve", new=fake_retrieve),
        patch.object(base.BaseAgent, "run", new=fake_run),
    ):
        result = await run_all([_chunk()], repo_id="r" * 64, db_path=tmp_path)

    assert result.is_ok()
    out = result.unwrap()
    assert out.arch is None  # the one that exploded
    assert out.modules == {"module": "ok"}
    assert out.deps == {"deps": "ok"}
    assert out.contributor == {"contributor": "ok"}


@pytest.mark.asyncio
async def test_all_retrievals_run_before_any_agent(tmp_path: Path) -> None:
    call_log: list[str] = []

    async def fake_retrieve(query, *, repo_id, db_path):  # noqa: ANN001, ANN202
        call_log.append("retrieve")
        return Ok([_chunk()])

    async def fake_run(self, chunks):  # noqa: ANN001, ANN202
        call_log.append("run")
        return Ok({self.name: "ok"})

    with (
        patch.object(orchestrator.retriever, "retrieve", new=fake_retrieve),
        patch.object(base.BaseAgent, "run", new=fake_run),
    ):
        await run_all([_chunk()], repo_id="r" * 64, db_path=tmp_path)

    assert call_log.count("retrieve") == 4
    assert call_log.count("run") == 4
    # Every retrieval completes before any agent starts.
    assert call_log[:4] == ["retrieve"] * 4
    assert call_log[4:] == ["run"] * 4


@pytest.mark.asyncio
async def test_retrieval_failure_falls_back_to_provided_chunks(tmp_path: Path) -> None:
    seen_chunks: dict[str, list] = {}

    async def failing_retrieve(query, *, repo_id, db_path):  # noqa: ANN001, ANN202
        return Err(RetrieveError(reason="no collection"))

    async def fake_run(self, chunks):  # noqa: ANN001, ANN202
        seen_chunks[self.name] = chunks
        return Ok({self.name: "ok"})

    fallback = [_chunk("fallback", 0.9)]
    with (
        patch.object(orchestrator.retriever, "retrieve", new=failing_retrieve),
        patch.object(base.BaseAgent, "run", new=fake_run),
    ):
        result = await run_all(fallback, repo_id="r" * 64, db_path=tmp_path)

    assert result.is_ok()
    # Each agent got the fallback chunk list when its retrieval failed.
    for name in ("module", "arch", "deps", "contributor"):
        assert seen_chunks[name] == fallback

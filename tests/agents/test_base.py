from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agents import base
from agents.base import AgentError, BaseAgent
from ingestion.types import Chunk, Err, Ok, RankedChunk


class _DummyAgent(BaseAgent):
    name = "dummy"
    query = "dummy query"
    system_prompt = "Return JSON."


def _chunk(name: str = "foo") -> RankedChunk:
    c = Chunk(
        chunk_id=f"id_{name}",
        file_path=Path(f"/repo/{name}.py"),
        language="python",
        unit_type="function",
        name=name,
        start_line=1,
        end_line=2,
        content="def foo(): pass",
        token_count=5,
        imports=(),
        calls=(),
        context_header=f"# python | function {name}",
    )
    return RankedChunk(chunk=c, importance_score=0.5)


@pytest.mark.asyncio
async def test_successful_call_returns_parsed_dict() -> None:
    with patch.object(
        base, "_complete_text", new=AsyncMock(side_effect=[Ok('{"modules": [1, 2]}')])
    ) as mock:
        result = await _DummyAgent().run([_chunk()])
    assert result.is_ok()
    assert result.unwrap() == {"modules": [1, 2]}
    assert mock.await_count == 1


@pytest.mark.asyncio
async def test_invalid_json_retries_once_then_succeeds() -> None:
    with patch.object(
        base,
        "_complete_text",
        new=AsyncMock(side_effect=[Ok("here you go, no json"), Ok('{"ok": true}')]),
    ) as mock:
        result = await _DummyAgent().run([_chunk()])
    assert result.is_ok()
    assert result.unwrap() == {"ok": True}
    assert mock.await_count == 2  # original + one retry


@pytest.mark.asyncio
async def test_invalid_json_twice_returns_agent_error() -> None:
    with patch.object(
        base,
        "_complete_text",
        new=AsyncMock(side_effect=[Ok("not json"), Ok("still not json")]),
    ) as mock:
        result = await _DummyAgent().run([_chunk()])
    assert not result.is_ok()
    assert isinstance(result.error, AgentError)
    assert "valid JSON" in result.error.reason
    assert mock.await_count == 2


@pytest.mark.asyncio
async def test_groq_missing_falls_back_to_gemini(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")

    groq_mock = AsyncMock()
    gemini_mock = AsyncMock(return_value='{"from": "gemini"}')
    with (
        patch.object(base, "_call_groq", new=groq_mock),
        patch.object(base, "_call_gemini", new=gemini_mock),
    ):
        result = await base._complete_text("dummy", "sys", "user")

    assert result.is_ok()
    assert result.unwrap() == '{"from": "gemini"}'
    groq_mock.assert_not_called()  # no GROQ key → Groq never attempted
    gemini_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_backend_returns_agent_error(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = await base._complete_text("dummy", "sys", "user")
    assert not result.is_ok()
    assert isinstance(result.error, AgentError)


@pytest.mark.asyncio
async def test_timeout_returns_agent_error(monkeypatch) -> None:
    monkeypatch.setattr(base, "AGENT_TIMEOUT", 0.05)

    async def _slow(*args, **kwargs):  # noqa: ANN002, ANN003
        await asyncio.sleep(1.0)
        return Ok('{"never": "returned"}')

    with patch.object(base, "_complete_text", new=_slow):
        result = await _DummyAgent().run([_chunk()])

    assert not result.is_ok()
    assert isinstance(result.error, AgentError)
    assert "timed out" in result.error.reason


@pytest.mark.asyncio
async def test_llm_backend_error_propagates_as_agent_error() -> None:
    err = Err(AgentError(agent="dummy", reason="Gemini call failed: boom"))
    with patch.object(base, "_complete_text", new=AsyncMock(side_effect=[err])):
        result = await _DummyAgent().run([_chunk()])
    assert not result.is_ok()
    assert isinstance(result.error, AgentError)
    assert "boom" in result.error.reason

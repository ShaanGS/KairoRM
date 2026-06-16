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


def test_rate_limit_wait_detects_429_and_caps() -> None:
    # A 429/quota error yields a bounded wait; a non-rate-limit error yields None.
    rate_err = Exception("429 You exceeded your current quota. Please retry in 27s.")
    wait = base._rate_limit_wait(rate_err, attempt=0)
    assert wait is not None
    # Honoured hint (27s) is capped at RATE_LIMIT_MAX_WAIT, plus <=1.5s jitter.
    assert wait <= base.RATE_LIMIT_MAX_WAIT + 1.5
    assert base._rate_limit_wait(Exception("connection reset"), attempt=0) is None


@pytest.mark.asyncio
async def test_call_gemini_retries_on_429_then_succeeds() -> None:
    # First call 429s, second succeeds — the retry loop must recover without raising.
    calls = {"n": 0}

    async def _gen(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("429 rate limit exceeded")

        class _Resp:
            text = '{"ok": true}'

        return _Resp()

    fake_model = type("M", (), {"generate_content_async": staticmethod(_gen)})()
    fake_genai = type(
        "G",
        (),
        {
            "configure": staticmethod(lambda **k: None),
            "GenerativeModel": staticmethod(lambda *a, **k: fake_model),
        },
    )()
    with (
        patch.dict("sys.modules", {"google.generativeai": fake_genai}),
        patch.object(base.asyncio, "sleep", new=AsyncMock()),
    ):
        out = await base._call_gemini("sys", "msg", "key")

    assert out == '{"ok": true}'
    assert calls["n"] == 2


def test_rate_limit_wait_fails_fast_on_daily_quota() -> None:
    # A per-minute burst → retry (bounded wait). A daily/TPD cap → don't retry (None),
    # so the run fails fast instead of burning the agent timeout on doomed retries.
    per_minute = Exception("429 rate limit reached ... please try again in 8.5s")
    assert base._rate_limit_wait(per_minute, 0) is not None

    daily = Exception(
        "429 Rate limit reached ... tokens per day (TPD): Limit 100000, Used 98217. "
        "Please try again in 20m18s"
    )
    assert base._rate_limit_wait(daily, 0) is None

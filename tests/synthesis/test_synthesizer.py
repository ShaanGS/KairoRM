from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ingestion.types import (
    AgentOutputs,
    Chunk,
    Ok,
    RankedChunk,
    SynthesisEntryPoint,
    SynthesisModule,
    SynthesisResult,
)
from synthesis import synthesizer
from synthesis.synthesizer import SynthesisError, synthesize

REPO_ID = "a" * 64


def _chunk(path: str, name: str, importance: float = 0.5) -> RankedChunk:
    c = Chunk(
        chunk_id=f"id_{name}",
        file_path=Path(path),
        language="python",
        unit_type="function",
        name=name,
        start_line=1,
        end_line=3,
        content=f"def {name}(): pass",
        token_count=6,
        imports=(),
        calls=(),
        context_header=f"# python | function {name}",
    )
    return RankedChunk(chunk=c, importance_score=importance)


def _outputs(**overrides) -> AgentOutputs:
    base = {
        "modules": {"modules": [{"name": "auth", "path": "auth.py", "responsibility": "x"}]},
        "arch": {"architecture": {"pattern": "library"}},
        "deps": {"dependencies": {"external": [{"name": "db", "purpose": "y"}]}},
        "contributor": {"contributor": {"test_command": "pytest"}},
    }
    base.update(overrides)
    return AgentOutputs(**base)


def _full_llm_json(modules=None, entry_points=None) -> str:
    return json.dumps(
        {
            "architecture_summary": "A small auth library. It hashes and verifies. Clean.",
            "modules": modules
            if modules is not None
            else [{"name": "auth", "path": "auth.py", "responsibility": "auth logic"}],
            "key_dependencies": ["db"],
            "circular_risks": [],
            "entry_points": entry_points
            if entry_points is not None
            else [{"name": "verify", "file": "auth.py", "description": "entry"}],
            "contributor_quickstart": ["pip install -e .", "pytest"],
            "complexity_score": 4,
        }
    )


@pytest.mark.asyncio
async def test_all_outputs_populated_returns_full_result() -> None:
    chunks = [_chunk("auth.py", "verify", 0.9)]
    with patch.object(
        synthesizer, "_complete_text", new=AsyncMock(side_effect=[Ok(_full_llm_json())])
    ):
        result = await synthesize(_outputs(), chunks, repo_id=REPO_ID)

    assert result.is_ok()
    res = result.unwrap()
    assert isinstance(res, SynthesisResult)
    assert res.repo_id == REPO_ID
    assert res.architecture_summary
    assert res.modules and isinstance(res.modules[0], SynthesisModule)
    assert res.entry_points and isinstance(res.entry_points[0], SynthesisEntryPoint)
    assert res.key_dependencies == ["db"]
    assert 1 <= res.complexity_score <= 10
    assert res.generated_at is not None


@pytest.mark.asyncio
async def test_partial_inputs_two_none_still_synthesizes() -> None:
    chunks = [_chunk("auth.py", "verify", 0.9)]
    outputs = _outputs(modules=None, arch=None)  # only deps + contributor present
    with patch.object(
        synthesizer, "_complete_text", new=AsyncMock(side_effect=[Ok(_full_llm_json())])
    ):
        result = await synthesize(outputs, chunks, repo_id=REPO_ID)
    assert result.is_ok()
    assert isinstance(result.unwrap(), SynthesisResult)


@pytest.mark.asyncio
async def test_all_none_returns_error_without_llm_call() -> None:
    outputs = AgentOutputs(modules=None, arch=None, deps=None, contributor=None)
    mock = AsyncMock()
    with patch.object(synthesizer, "_complete_text", new=mock):
        result = await synthesize(outputs, [_chunk("auth.py", "v")], repo_id=REPO_ID)
    assert not result.is_ok()
    assert isinstance(result.error, SynthesisError)
    assert result.error.reason == "all agents failed"
    mock.assert_not_called()


@pytest.mark.asyncio
async def test_hallucinated_paths_are_removed() -> None:
    chunks = [_chunk("auth.py", "verify", 0.9)]  # only auth.py is real
    llm = _full_llm_json(
        modules=[
            {"name": "auth", "path": "auth.py", "responsibility": "real"},
            {"name": "ghost", "path": "ghost.py", "responsibility": "hallucinated"},
        ],
        entry_points=[
            {"name": "verify", "file": "auth.py", "description": "real"},
            {"name": "phantom", "file": "nowhere.py", "description": "hallucinated"},
        ],
    )
    with patch.object(synthesizer, "_complete_text", new=AsyncMock(side_effect=[Ok(llm)])):
        result = await synthesize(_outputs(), chunks, repo_id=REPO_ID)

    assert result.is_ok()
    res = result.unwrap()
    module_paths = [m.path for m in res.modules]
    entry_files = [e.file for e in res.entry_points]
    assert "ghost.py" not in module_paths
    assert "auth.py" in module_paths
    assert "nowhere.py" not in entry_files
    assert "auth.py" in entry_files


@pytest.mark.asyncio
async def test_all_real_modules_covered_even_when_llm_omits_them() -> None:
    # Chunks span three real top-level dirs; the LLM only describes one and invents a
    # fake one. The result must contain every real module and no fabricated one.
    chunks = [
        _chunk("ingestion/fetch.py", "fetch", 0.9),
        _chunk("indexing/store.py", "store", 0.5),
        _chunk("agents/run.py", "run", 0.3),
    ]
    llm = _full_llm_json(
        modules=[
            {
                "name": "ingestion",
                "path": "ingestion",
                "responsibility": "clones and filters repos",
            },
            {"name": "ghost", "path": "ghost", "responsibility": "not real"},
        ]
    )
    with patch.object(synthesizer, "_complete_text", new=AsyncMock(side_effect=[Ok(llm)])):
        result = await synthesize(_outputs(), chunks, repo_id=REPO_ID)

    res = result.unwrap()
    by = {m.name: m.responsibility for m in res.modules}
    assert set(by) == {"ingestion", "indexing", "agents"}  # all real dirs, no "ghost"
    assert by["ingestion"] == "clones and filters repos"  # LLM prose kept
    assert by["indexing"] and by["agents"]  # omitted modules get a non-empty fallback


@pytest.mark.asyncio
async def test_invalid_json_retries_once_then_succeeds() -> None:
    chunks = [_chunk("auth.py", "verify", 0.9)]
    mock = AsyncMock(side_effect=[Ok("not json at all"), Ok(_full_llm_json())])
    with patch.object(synthesizer, "_complete_text", new=mock):
        result = await synthesize(_outputs(), chunks, repo_id=REPO_ID)
    assert result.is_ok()
    assert mock.await_count == 2


@pytest.mark.asyncio
async def test_timeout_returns_synthesis_error(monkeypatch) -> None:
    monkeypatch.setattr(synthesizer, "SYNTH_TIMEOUT", 0.05)

    async def _slow(*args, **kwargs):  # noqa: ANN002, ANN003
        await asyncio.sleep(1.0)
        return Ok(_full_llm_json())

    with patch.object(synthesizer, "_complete_text", new=_slow):
        result = await synthesize(_outputs(), [_chunk("auth.py", "v")], repo_id=REPO_ID)

    assert not result.is_ok()
    assert isinstance(result.error, SynthesisError)
    assert "timed out" in result.error.reason

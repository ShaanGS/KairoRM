from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agents import base
from agents.arch_agent import ArchAgent
from agents.contributor_agent import ContributorAgent
from agents.dep_agent import DepAgent
from agents.module_agent import ModuleAgent
from ingestion.types import Chunk, Ok, RankedChunk


def _chunk(name: str, importance: float) -> RankedChunk:
    c = Chunk(
        chunk_id=f"id_{name}",
        file_path=Path(f"/repo/{name}.py"),
        language="python",
        unit_type="function",
        name=name,
        start_line=1,
        end_line=4,
        content=f"def {name}(): pass",
        token_count=6,
        imports=(),
        calls=(),
        context_header=f"# python | function {name}",
    )
    return RankedChunk(chunk=c, importance_score=importance)


def test_agent_query_strings_are_correct() -> None:
    assert ModuleAgent.query == "folder structure module responsibilities entry points"
    assert ArchAgent.query == "architecture patterns system design data flow dependencies"
    assert DepAgent.query == "imports dependencies external packages call graph entry points"
    assert ContributorAgent.query == "setup configuration entry point how to run tests build"


def test_agent_names_map_to_outputs() -> None:
    assert ModuleAgent.name == "module"
    assert ArchAgent.name == "arch"
    assert DepAgent.name == "deps"
    assert ContributorAgent.name == "contributor"


@pytest.mark.asyncio
async def test_module_agent_returns_modules_key() -> None:
    payload = json.dumps(
        {
            "modules": [
                {
                    "name": "cli",
                    "path": "cli/",
                    "responsibility": "x",
                    "key_files": ["main.py"],
                    "exports": ["run"],
                }
            ]
        }
    )
    with patch.object(base, "_complete_text", new=AsyncMock(side_effect=[Ok(payload)])):
        result = await ModuleAgent().run([_chunk("a", 0.1)])
    assert result.is_ok()
    assert "modules" in result.unwrap()


@pytest.mark.asyncio
async def test_arch_agent_returns_architecture_key() -> None:
    payload = json.dumps(
        {
            "architecture": {
                "pattern": "layered",
                "layers": [],
                "data_flow": "x",
                "key_decisions": [],
            }
        }
    )
    with patch.object(base, "_complete_text", new=AsyncMock(side_effect=[Ok(payload)])):
        result = await ArchAgent().run([_chunk("a", 0.1)])
    assert result.is_ok()
    assert "architecture" in result.unwrap()
    assert result.unwrap()["architecture"]["pattern"] == "layered"


@pytest.mark.asyncio
async def test_contributor_agent_returns_contributor_key() -> None:
    payload = json.dumps(
        {
            "contributor": {
                "setup_steps": ["pip install"],
                "entry_points": [],
                "test_command": "pytest",
                "build_command": "make",
                "first_pr_advice": "start small",
            }
        }
    )
    with patch.object(base, "_complete_text", new=AsyncMock(side_effect=[Ok(payload)])):
        result = await ContributorAgent().run([_chunk("a", 0.1)])
    assert result.is_ok()
    assert "contributor" in result.unwrap()


@pytest.mark.asyncio
async def test_dep_agent_hotspots_use_chunk_scores_not_llm() -> None:
    # LLM tries to hallucinate hotspots with bogus scores; postprocess must overwrite them.
    payload = json.dumps(
        {
            "dependencies": {
                "external": [{"name": "requests", "purpose": "http"}],
                "internal_hotspots": [
                    {"file": "WRONG.py", "reason": "hallucinated", "importance_score": 9.99}
                ],
                "circular_risks": [],
            }
        }
    )
    chunks = [
        _chunk("low", 0.10),
        _chunk("highest", 0.90),
        _chunk("mid", 0.50),
        _chunk("high", 0.70),
        _chunk("lowest", 0.05),
        _chunk("tiny", 0.01),
    ]
    with patch.object(base, "_complete_text", new=AsyncMock(side_effect=[Ok(payload)])):
        result = await DepAgent().run(chunks)

    assert result.is_ok()
    hotspots = result.unwrap()["dependencies"]["internal_hotspots"]
    # Top 5 by importance, descending — exactly from chunk data.
    assert [h["importance_score"] for h in hotspots] == [0.90, 0.70, 0.50, 0.10, 0.05]
    assert [Path(h["file"]).stem for h in hotspots] == ["highest", "high", "mid", "low", "lowest"]
    # The LLM's bogus hotspot must be gone; external deps preserved.
    assert all(h["file"] != "WRONG.py" for h in hotspots)
    assert result.unwrap()["dependencies"]["external"] == [{"name": "requests", "purpose": "http"}]


@pytest.mark.asyncio
async def test_dep_agent_circular_risks_grounded_not_hallucinated() -> None:
    # The LLM invents a textbook cycle; postprocess must discard it and use only the
    # real broken back-edges from the ranker — dropping same-file self-loops.
    payload = json.dumps(
        {
            "dependencies": {
                "external": [],
                "internal_hotspots": [],
                "circular_risks": ["auth -> db -> auth"],
            }
        }
    )
    cycles = [
        ("/repo/a.py", "/repo/b.py"),  # real cross-file cycle — kept
        ("/repo/x.py", "/repo/x.py"),  # same-file self-loop — dropped
        ("/repo/a.py", "/repo/b.py"),  # duplicate — deduped
    ]
    with patch.object(base, "_complete_text", new=AsyncMock(side_effect=[Ok(payload)])):
        result = await DepAgent(cycles=cycles).run([_chunk("a", 0.5)])

    assert result.is_ok()
    risks = result.unwrap()["dependencies"]["circular_risks"]
    assert risks == ["a.py -> b.py"]
    assert "auth -> db -> auth" not in risks


@pytest.mark.asyncio
async def test_dep_agent_no_cycles_means_no_risks() -> None:
    # No real cycles → circular_risks is empty, even if the LLM offered some.
    payload = json.dumps(
        {
            "dependencies": {
                "external": [],
                "internal_hotspots": [],
                "circular_risks": ["a -> b -> a"],
            }
        }
    )
    with patch.object(base, "_complete_text", new=AsyncMock(side_effect=[Ok(payload)])):
        result = await DepAgent().run([_chunk("a", 0.5)])

    assert result.is_ok()
    assert result.unwrap()["dependencies"]["circular_risks"] == []

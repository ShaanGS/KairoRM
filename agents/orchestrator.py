"""Parallel agent orchestration.

Fans out to all four specialist agents at once. Each agent first gets its own
retrieval (the four `retriever.retrieve` calls run concurrently), then the four
agents run concurrently via `asyncio.gather(return_exceptions=True)`. A single agent
crashing — or its retrieval failing — nulls only that one field; the other three
still come back. The whole pipeline degrades gracefully, never hard-fails on one
flaky LLM call.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from agents.arch_agent import ArchAgent
from agents.base import BaseAgent
from agents.contributor_agent import ContributorAgent
from agents.dep_agent import DepAgent
from agents.module_agent import ModuleAgent
from indexing import retriever
from ingestion.types import AgentOutputs, Err, Ok, RankedChunk, Result

log = logging.getLogger("kairo")


@dataclass(frozen=True, slots=True)
class OrchestratorError:
    reason: str


def _build_agents(cycles: list[tuple[str, str]]) -> list[BaseAgent]:
    # Order is meaningful: it maps positionally onto AgentOutputs fields.
    # DepAgent is given the broken back-edges so it can ground circular_risks.
    return [ModuleAgent(), ArchAgent(), DepAgent(cycles=cycles), ContributorAgent()]


async def _retrieve_for(
    agent: BaseAgent,
    *,
    repo_id: str,
    db_path: Path,
    fallback: list[RankedChunk],
) -> list[RankedChunk]:
    try:
        result = await retriever.retrieve(agent.query, repo_id=repo_id, db_path=db_path)
    except Exception as exc:
        log.warning("Retrieval raised for %s (%s); using fallback chunks", agent.name, exc)
        return fallback
    if not result.is_ok():
        log.warning(
            "Retrieval failed for %s (%s); using fallback chunks", agent.name, result.error.reason
        )
        return fallback
    retrieved = result.unwrap()
    return retrieved if retrieved else fallback


async def run_all(
    chunks: list[RankedChunk],
    *,
    repo_id: str,
    db_path: Path,
    cycles: list[tuple[str, str]] | None = None,
) -> Result[AgentOutputs, OrchestratorError]:
    """Run all four agents concurrently and collect their outputs.

    `chunks` is the full ranked set; it serves as a fallback when an agent's
    retrieval comes back empty or errors, so agents always have something to work on.
    `cycles` are the broken back-edges from `RankResult.cycles`; callers (the CLI
    layer) thread `rank_result.cycles` through so the DepAgent can ground
    circular_risks. Defaults to empty for backwards compatibility.

    Note: the spec described `cycles` as `field(default_factory=list)`, which is a
    dataclass construct — for a plain function we use the idiomatic `| None = None`
    sentinel and normalize to `[]` below to avoid a mutable default argument.
    """
    try:
        agents = _build_agents(cycles or [])

        # Phase 1: four retrieval calls, concurrently.
        retrieved_lists = await asyncio.gather(
            *(
                _retrieve_for(agent, repo_id=repo_id, db_path=db_path, fallback=chunks)
                for agent in agents
            )
        )

        # Phase 2: four agents, concurrently — one failure must not sink the rest.
        results = await asyncio.gather(
            *(
                agent.run(retrieved)
                for agent, retrieved in zip(agents, retrieved_lists, strict=True)
            ),
            return_exceptions=True,
        )

        outputs: list[dict | None] = []
        for agent, result in zip(agents, results, strict=True):
            if isinstance(result, BaseException):
                log.warning("Agent %s raised: %s", agent.name, result)
                outputs.append(None)
            elif not result.is_ok():
                log.warning("Agent %s failed: %s", agent.name, result.error.reason)
                outputs.append(None)
            else:
                outputs.append(result.unwrap())

        return Ok(
            AgentOutputs(
                modules=outputs[0],
                arch=outputs[1],
                deps=outputs[2],
                contributor=outputs[3],
            )
        )
    except Exception as exc:
        return Err(OrchestratorError(reason=str(exc)))

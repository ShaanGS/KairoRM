"""Dependency agent: external packages, internal hotspots, circular risks.

Two fields are NOT trusted to the LLM — both are filled deterministically:
- `internal_hotspots` comes from the PageRank `importance_score` on each chunk, so the
  most central files are reported with their true scores, not a hallucinated ranking.
- `circular_risks` comes from the real back-edges the ranker broke in the call graph.
  Left to the LLM it invents textbook cycles like "auth -> db -> auth" for repos that
  have no such modules, so its output for this field is discarded entirely.
"""

from __future__ import annotations

from pathlib import Path

from agents.base import BaseAgent
from ingestion.types import RankedChunk

_SYSTEM_PROMPT = (
    "You are a dependency analyst. Given code chunks from a repository, analyze its "
    "dependencies. Return ONLY a single valid JSON object — no preamble, no markdown "
    "fences — matching exactly this schema:\n"
    '{"dependencies": {"external": [{"name": str, "purpose": str}], '
    '"internal_hotspots": [{"file": str, "reason": str, "importance_score": number}], '
    '"circular_risks": [str]}}\n'
    "Rules: `external` lists third-party packages and their purpose. Leave "
    "`internal_hotspots` as an empty list — it is filled from PageRank data, not by you. "
    "Leave `circular_risks` as an empty list too — it is filled from the call graph, not "
    "by you. Do NOT guess circular dependencies."
)

_HOTSPOT_LIMIT = 5


class DepAgent(BaseAgent):
    name = "deps"
    query = "imports dependencies external packages call graph entry points"
    system_prompt = _SYSTEM_PROMPT

    def __init__(self, cycles: list[tuple[str, str]] | None = None) -> None:
        # Back-edges broken by the ranker as (file_path, file_path). Passed in from
        # RankResult.cycles; these are the sole source of circular_risks (see below).
        self._cycles: list[tuple[str, str]] = cycles or []

    def _postprocess(self, data: dict, chunks: list[RankedChunk]) -> dict:
        # Override the LLM's internal_hotspots with the true PageRank ranking.
        top = sorted(chunks, key=lambda rc: rc.importance_score, reverse=True)[:_HOTSPOT_LIMIT]
        hotspots = [
            {
                "file": str(rc.chunk.file_path),
                "reason": f"{rc.chunk.unit_type} {rc.chunk.name} — central in the call graph",
                "importance_score": rc.importance_score,
            }
            for rc in top
        ]
        deps = data.get("dependencies")
        if not isinstance(deps, dict):
            deps = {}
        deps["internal_hotspots"] = hotspots
        deps.setdefault("external", [])
        # Ground circular_risks solely in the real broken back-edges, discarding whatever
        # the LLM produced. Drop same-file self-loops (intra-file recursion is not a
        # circular *dependency*) and dedupe, preserving order.
        grounded = [
            f"{Path(caller).name} -> {Path(callee).name}"
            for caller, callee in self._cycles
            if caller != callee
        ]
        deps["circular_risks"] = list(dict.fromkeys(grounded))
        data["dependencies"] = deps
        return data

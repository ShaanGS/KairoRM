"""Dependency agent: external packages, internal hotspots, circular risks.

`internal_hotspots` is NOT trusted to the LLM — it is filled deterministically from
the PageRank `importance_score` already attached to each chunk, so the most central
files are reported with their true scores rather than a hallucinated ranking.
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
    "`circular_risks` notes any modules that look mutually dependent."
)

_HOTSPOT_LIMIT = 5


class DepAgent(BaseAgent):
    name = "deps"
    query = "imports dependencies external packages call graph entry points"
    system_prompt = _SYSTEM_PROMPT

    def __init__(self, cycles: list[tuple[str, str]] | None = None) -> None:
        # Back-edges broken by the ranker (caller_chunk_id, callee_chunk_id). Passed
        # in from RankResult.cycles so the agent can ground circular_risks instead of
        # asking the LLM to guess them.
        self._cycles: list[tuple[str, str]] = cycles or []

    def _format_context(self, chunks: list[RankedChunk]) -> str:
        context = super()._format_context(chunks)
        if not self._cycles:
            return context
        risk_lines = "\n".join(
            f"- {Path(caller).name} -> {Path(callee).name}" for caller, callee in self._cycles
        )
        return (
            f"{context}\n\n"
            "## Known circular dependencies (detected in the call graph)\n"
            "These back-edges were broken during ranking; report them as circular_risks:\n"
            f"{risk_lines}"
        )

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
        deps.setdefault("circular_risks", [])
        # Ground circular_risks in the actual broken back-edges, merging with anything
        # the LLM surfaced (deduped, order-preserving).
        if self._cycles:
            grounded = [
                f"{Path(caller).name} -> {Path(callee).name}" for caller, callee in self._cycles
            ]
            existing = deps["circular_risks"] if isinstance(deps["circular_risks"], list) else []
            merged = list(dict.fromkeys([*existing, *grounded]))
            deps["circular_risks"] = merged
        data["dependencies"] = deps
        return data

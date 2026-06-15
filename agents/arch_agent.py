"""Architecture agent: identifies the overall system pattern and layering."""

from __future__ import annotations

from agents.base import BaseAgent

_SYSTEM_PROMPT = (
    "You are a software architecture analyst. Given code chunks from a repository, "
    "describe its high-level architecture. Return ONLY a single valid JSON object — no "
    "preamble, no markdown fences — matching exactly this schema:\n"
    '{"architecture": {"pattern": str, "layers": [{"name": str, "responsibility": str}], '
    '"data_flow": str, "key_decisions": [str]}}\n'
    "Rules: `pattern` must be exactly one of: monolith, layered, microservices, "
    "event-driven, pipeline, library, unknown. `data_flow` is two sentences max. "
    "`key_decisions` lists notable design choices."
)


class ArchAgent(BaseAgent):
    name = "arch"
    query = "architecture patterns system design data flow dependencies"
    system_prompt = _SYSTEM_PROMPT

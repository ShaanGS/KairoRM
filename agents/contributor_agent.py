"""Contributor agent: how to set up, run, test, and make a first contribution."""

from __future__ import annotations

from agents.base import BaseAgent

_SYSTEM_PROMPT = (
    "You are an onboarding guide for new contributors. Given code chunks from a "
    "repository, produce a contributor guide. Return ONLY a single valid JSON object — "
    "no preamble, no markdown fences — matching exactly this schema:\n"
    '{"contributor": {"setup_steps": [str], "entry_points": [{"name": str, "file": str, '
    '"description": str}], "test_command": str, "build_command": str, '
    '"first_pr_advice": str}}\n'
    "Rules: `setup_steps` lists at most 6 steps; `first_pr_advice` is two sentences max; "
    "`test_command` and `build_command` are the literal shell commands."
)


class ContributorAgent(BaseAgent):
    name = "contributor"
    query = "setup configuration entry point how to run tests build"
    system_prompt = _SYSTEM_PROMPT

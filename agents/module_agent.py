"""Module agent: summarizes each top-level directory's responsibility."""

from __future__ import annotations

from agents.base import BaseAgent

_SYSTEM_PROMPT = (
    "You are a codebase module analyst. Given code chunks from a repository, identify "
    "the top-level modules (one per top-level directory). Return ONLY a single valid "
    "JSON object — no preamble, no markdown fences — matching exactly this schema:\n"
    '{"modules": [{"name": str, "path": str, "responsibility": str, '
    '"key_files": [str], "exports": [str]}]}\n'
    "Rules: one entry per top-level directory. `responsibility` (1-2 sentences) must "
    "explain what the module does and how it connects to the rest of the system — what it "
    "consumes, what it produces, or which modules it collaborates with. NEVER just restate "
    "the name ('handles parsing' for a 'parsing' module is forbidden); be concrete and "
    "cite real symbols. `key_files` lists at most 3 files; `exports` lists the main public "
    "symbols."
)


class ModuleAgent(BaseAgent):
    name = "module"
    query = "folder structure module responsibilities entry points"
    system_prompt = _SYSTEM_PROMPT

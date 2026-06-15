"""Synthesis: merge the four specialist reports into one grounded result.

Takes `AgentOutputs` (any field may be `None`), folds the surviving reports plus the
top code chunks into a single deduplicated context, and asks one senior-staff-engineer
LLM call to produce a unified `SynthesisResult`. The LLM's JSON is never trusted
blindly: every file path it returns is checked against the actual evidence chunks and
hallucinated paths are dropped before the result leaves this module.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

# Reuse the agents' LLM dispatch + JSON parsing — do not reimplement the fallback.
from agents.base import _complete_text, _extract_json
from ingestion.types import (
    AgentOutputs,
    Err,
    Ok,
    RankedChunk,
    Result,
    SynthesisEntryPoint,
    SynthesisModule,
    SynthesisResult,
)

SYNTH_TIMEOUT = 45.0  # seconds — synthesis context is larger than a single agent's
EVIDENCE_CHUNK_LIMIT = 10

console = Console(stderr=True)

_SYSTEM_PROMPT = (
    "You are a senior staff engineer writing the canonical overview of a codebase. "
    "You are given four specialist reports (modules, architecture, dependencies, "
    "contributor) and the most important code chunks as evidence. Synthesize them into "
    "one coherent picture. Return ONLY a single valid JSON object — no preamble, no "
    "markdown fences — matching exactly this schema:\n"
    "{"
    '"architecture_summary": str, '
    '"modules": [{"name": str, "path": str, "responsibility": str}], '
    '"key_dependencies": [str], '
    '"circular_risks": [str], '
    '"entry_points": [{"name": str, "file": str, "description": str}], '
    '"contributor_quickstart": [str], '
    '"complexity_score": int'
    "}\n"
    "Rules: `architecture_summary` is three sentences max. `contributor_quickstart` is "
    "at most 6 ordered steps. `complexity_score` is an integer 1-10. Only reference file "
    "paths that appear in the evidence chunks."
)

_JSON_REMINDER = (
    "Your previous response was not valid JSON. Return ONLY a single valid JSON object "
    "matching the schema. No preamble, no markdown fences."
)


@dataclass(frozen=True, slots=True)
class SynthesisError:
    reason: str


def _merge_and_dedupe(outputs: AgentOutputs, top_chunks: list[RankedChunk]) -> str:
    """Build the labelled, deduplicated context string sent to the LLM."""
    sections: list[tuple[str, dict]] = []
    if outputs.modules is not None:
        sections.append(("Modules", outputs.modules))
    if outputs.arch is not None:
        sections.append(("Architecture", outputs.arch))
    if outputs.deps is not None:
        sections.append(("Dependencies", outputs.deps))
    if outputs.contributor is not None:
        sections.append(("Contributor", outputs.contributor))

    parts: list[str] = []
    seen_values: set[str] = set()
    for title, data in sections:
        parts.append(f"## {title}")
        block = json.dumps(data, indent=2, default=str)
        for line in block.splitlines():
            stripped = line.strip()
            # Dedupe repeated scalar lines (file paths, module/dep names) that recur
            # across reports; leave structural braces untouched.
            if '"' in stripped:
                if stripped in seen_values:
                    continue
                seen_values.add(stripped)
            parts.append(line)

    parts.append("\n## Evidence — top code chunks by importance")
    top = sorted(top_chunks, key=lambda rc: rc.importance_score, reverse=True)[
        :EVIDENCE_CHUNK_LIMIT
    ]
    for rc in top:
        c = rc.chunk
        parts.append(
            f"- {c.file_path} :: {c.unit_type} {c.name} "
            f"(importance={rc.importance_score:.4f})"
        )
    return "\n".join(parts)


async def _call_llm(user_message: str) -> Result[dict, SynthesisError]:
    first = await _complete_text("synthesis", _SYSTEM_PROMPT, user_message)
    if not first.is_ok():
        return Err(SynthesisError(reason=first.error.reason))
    parsed = _extract_json(first.unwrap())
    if parsed is not None:
        return Ok(parsed)

    second = await _complete_text(
        "synthesis", _SYSTEM_PROMPT, f"{user_message}\n\n{_JSON_REMINDER}"
    )
    if not second.is_ok():
        return Err(SynthesisError(reason=second.error.reason))
    parsed = _extract_json(second.unwrap())
    if parsed is not None:
        return Ok(parsed)
    return Err(SynthesisError(reason="LLM did not return valid JSON after one retry"))


def _valid_paths(top_chunks: list[RankedChunk]) -> tuple[set[str], set[str]]:
    full = {str(rc.chunk.file_path) for rc in top_chunks}
    names = {Path(str(rc.chunk.file_path)).name for rc in top_chunks}
    return full, names


def _normalize_path(path: str, repo_name: str) -> str:
    """Strip KairoRM's clone-cache prefix so paths display repo-relative.

    `kairomap-output/.cache/karpathy__micrograd/micrograd/engine.py` → `micrograd/engine.py`.
    Already-relative paths are returned unchanged.
    """
    p = str(path).replace("\\", "/")
    marker = "/.cache/"
    if marker in p:
        # After the marker: "<slug>/<repo-relative...>"; drop the slug directory.
        after = p.split(marker, 1)[1]
        return after.split("/", 1)[1] if "/" in after else after
    # Local repos (no cache): strip up to and including a "/<repo_name>/" segment.
    if repo_name:
        needle = f"/{repo_name}/"
        if needle in p:
            return p.split(needle, 1)[1]
    return p


def _is_grounded(path_str: str, full: set[str], names: set[str]) -> bool:
    if not path_str:
        return False
    if path_str in full or Path(path_str).name in names:
        return True
    # Tolerate dir/abs-vs-rel differences via substring overlap either direction.
    return any(path_str in f or f in path_str for f in full)


def _build_result(
    data: dict, top_chunks: list[RankedChunk], repo_id: str, repo_name: str | None
) -> SynthesisResult:
    full, names = _valid_paths(top_chunks)
    name = repo_name or ""

    modules: list[SynthesisModule] = []
    for m in data.get("modules", []) or []:
        path = str(m.get("path", ""))
        # Ground against the raw (cache) path, but store the clean repo-relative form.
        if _is_grounded(path, full, names):
            modules.append(
                SynthesisModule(
                    name=str(m.get("name", "")),
                    path=_normalize_path(path, name),
                    responsibility=str(m.get("responsibility", "")),
                )
            )
        else:
            console.log(f"[yellow]Dropping hallucinated module path: {path!r}[/]")

    entry_points: list[SynthesisEntryPoint] = []
    for e in data.get("entry_points", []) or []:
        file = str(e.get("file", ""))
        if _is_grounded(file, full, names):
            entry_points.append(
                SynthesisEntryPoint(
                    name=str(e.get("name", "")),
                    file=_normalize_path(file, name),
                    description=str(e.get("description", "")),
                )
            )
        else:
            console.log(f"[yellow]Dropping hallucinated entry-point file: {file!r}[/]")

    try:
        complexity = int(data.get("complexity_score", 5))
    except (TypeError, ValueError):
        complexity = 5
    complexity = max(1, min(10, complexity))

    return SynthesisResult(
        repo_id=repo_id,
        architecture_summary=str(data.get("architecture_summary", "")),
        modules=modules,
        key_dependencies=[str(d) for d in (data.get("key_dependencies", []) or [])],
        circular_risks=[str(c) for c in (data.get("circular_risks", []) or [])],
        entry_points=entry_points,
        contributor_quickstart=[
            str(s) for s in (data.get("contributor_quickstart", []) or [])
        ][:6],
        complexity_score=complexity,
        generated_at=datetime.now(UTC),
    )


async def synthesize(
    outputs: AgentOutputs,
    top_chunks: list[RankedChunk],
    *,
    repo_id: str,
    repo_name: str | None = None,
) -> Result[SynthesisResult, SynthesisError]:
    """Merge four agent reports into one grounded `SynthesisResult`.

    `repo_name`, when given, is used to strip the clone-cache prefix from file paths
    so the result reads `micrograd/engine.py` rather than the full cache path.
    """
    if (
        outputs.modules is None
        and outputs.arch is None
        and outputs.deps is None
        and outputs.contributor is None
    ):
        return Err(SynthesisError(reason="all agents failed"))

    user_message = _merge_and_dedupe(outputs, top_chunks)

    try:
        llm_result = await asyncio.wait_for(_call_llm(user_message), timeout=SYNTH_TIMEOUT)
    except TimeoutError:
        return Err(SynthesisError(reason=f"synthesis timed out after {SYNTH_TIMEOUT}s"))

    if not llm_result.is_ok():
        return Err(llm_result.error)

    return Ok(_build_result(llm_result.unwrap(), top_chunks, repo_id, repo_name))

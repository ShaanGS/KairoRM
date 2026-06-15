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
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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

log = logging.getLogger("kairo")

_SYSTEM_PROMPT = (
    "You are a senior staff engineer writing the canonical overview of a codebase. "
    "You are given four specialist reports (modules, architecture, dependencies, "
    "contributor), an AUTHORITATIVE MODULE LIST derived from the real file tree, and the "
    "most important code chunks as evidence. Synthesize them into one coherent picture. "
    "Return ONLY a single valid JSON object — no preamble, no markdown fences — matching "
    "exactly this schema:\n"
    "{"
    '"architecture_summary": str, '
    '"modules": [{"name": str, "path": str, "responsibility": str}], '
    '"key_dependencies": [str], '
    '"circular_risks": [str], '
    '"entry_points": [{"name": str, "file": str, "description": str}], '
    '"contributor_quickstart": [str], '
    '"complexity_score": int'
    "}\n"
    "Rules:\n"
    "- `modules` MUST contain one entry for EVERY name in the authoritative module list, "
    "using those exact names — do not invent, drop, or merge modules.\n"
    "- Each `responsibility` must explain what the module DOES and how it connects to the "
    "rest of the system — what it consumes, what it produces, or which modules it works "
    "with. NEVER merely restate the name: 'handles parsing' for a module named 'parsing' "
    "is forbidden. Be concrete and reference real symbols/files when useful. One or two "
    "sentences.\n"
    "- `architecture_summary` (3 sentences max) names the overall pattern and traces how "
    "data flows end to end across the modules.\n"
    "- `contributor_quickstart` is at most 6 ordered steps. `complexity_score` is an "
    "integer 1-10. Only reference file paths that appear in the evidence chunks."
)

_JSON_REMINDER = (
    "Your previous response was not valid JSON. Return ONLY a single valid JSON object "
    "matching the schema. No preamble, no markdown fences."
)


@dataclass(frozen=True, slots=True)
class SynthesisError:
    reason: str


def _repo_root(top_chunks: list[RankedChunk]) -> str:
    """The longest common directory of all chunk paths — the root we make paths relative
    to. Robust for local clones, temp dirs, and the fetch cache alike, unlike matching a
    repo name that may not appear in the path at all (e.g. `/var/folders/...`)."""
    paths = [str(rc.chunk.file_path) for rc in top_chunks]
    if not paths:
        return ""
    if len(paths) == 1:
        return str(Path(paths[0]).parent)
    try:
        return os.path.commonpath(paths)
    except ValueError:  # mixed absolute/relative, or different drives
        return ""


def _relpath(path: str, root: str) -> str:
    try:
        rel = os.path.relpath(path, root) if root else path
    except ValueError:
        rel = path
    return rel.replace("\\", "/")


def _module_of(path: str, root: str) -> tuple[str, str]:
    """Map a file path to its (module_name, module_path), relative to the repo root.

    A file in a subdirectory belongs to its top-level directory (`ingestion/x.py` →
    `ingestion`). A file at the repo root is its own module (`engine.py` → `engine`),
    so flat repos still produce a module map.
    """
    segs = [s for s in _relpath(path, root).split("/") if s and s not in (".", "..")]
    if len(segs) >= 2:
        return segs[0], segs[0]
    leaf = segs[-1] if segs else path
    return leaf.rsplit(".", 1)[0], leaf


# Languages that are docs/config, not source. A root-level file in one of these is not
# a "module" (otherwise README.md, pyproject.toml, Makefile pollute the module map).
_NONCODE_LANGS = frozenset(
    {
        "markdown",
        "toml",
        "yaml",
        "yml",
        "json",
        "ini",
        "cfg",
        "conf",
        "unknown",
        "make",
        "makefile",
        "dockerfile",
        "text",
        "plaintext",
        "rst",
        "csv",
        "lock",
        "env",
    }
)


def _module_inventory(top_chunks: list[RankedChunk]) -> list[dict]:
    """The authoritative module list, grounded in the real file tree.

    Every module that actually has code is represented exactly once — this is what stops
    real modules from silently vanishing when an agent's retrieval doesn't sample them.
    A module is kept if it is a directory package or holds real source (so root-level
    docs/config files like README.md don't masquerade as modules). Ordered by aggregate
    importance so the most central modules appear first.
    """
    root = _repo_root(top_chunks)
    groups: dict[str, dict] = {}
    for rc in top_chunks:
        path = str(rc.chunk.file_path)
        name, mpath = _module_of(path, root)
        is_dir = (
            len([s for s in _relpath(path, root).split("/") if s and s not in (".", "..")]) >= 2
        )
        is_code = rc.chunk.language.lower() not in _NONCODE_LANGS
        leaf = Path(path).name
        g = groups.setdefault(
            name,
            {"name": name, "path": mpath, "files": {}, "weight": 0.0, "dir": False, "code": False},
        )
        g["files"][leaf] = max(g["files"].get(leaf, 0.0), rc.importance_score)
        g["weight"] += rc.importance_score
        g["dir"] = g["dir"] or is_dir
        g["code"] = g["code"] or is_code
    inventory = []
    for g in sorted(groups.values(), key=lambda x: x["weight"], reverse=True):
        if not g["name"] or not (g["dir"] or g["code"]):
            continue  # root-level doc/config file, or unnamed — not a module
        key_files = [f for f, _ in sorted(g["files"].items(), key=lambda kv: kv[1], reverse=True)]
        inventory.append({"name": g["name"], "path": g["path"], "key_files": key_files[:3]})
    return inventory


def _evidence_chunks(top_chunks: list[RankedChunk], limit: int) -> list[RankedChunk]:
    """Pick evidence chunks that cover every module, then fill by importance.

    Taking only the global top-N starves modules whose code never reaches the top, so
    the LLM can't describe them. Seeding with the single most-important chunk per module
    guarantees every module has grounding before the remaining slots go to raw importance.
    """
    root = _repo_root(top_chunks)
    by_importance = sorted(top_chunks, key=lambda rc: rc.importance_score, reverse=True)
    picked: list[RankedChunk] = []
    seen_modules: set[str] = set()
    for rc in by_importance:
        mod, _ = _module_of(str(rc.chunk.file_path), root)
        if mod not in seen_modules:
            seen_modules.add(mod)
            picked.append(rc)
    picked_ids = {id(rc) for rc in picked}
    for rc in by_importance:
        if len(picked) >= limit:
            break
        if id(rc) not in picked_ids:
            picked.append(rc)
    return picked[:limit]


def _merge_and_dedupe(
    outputs: AgentOutputs, top_chunks: list[RankedChunk], inventory: list[dict]
) -> str:
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

    parts.append("\n## Authoritative module list — describe EVERY one")
    for inv in inventory:
        files = ", ".join(inv["key_files"]) or "(no files)"
        parts.append(f"- {inv['name']} ({inv['path']}/) — key files: {files}")

    limit = max(EVIDENCE_CHUNK_LIMIT, len(inventory) + 6)
    parts.append("\n## Evidence — representative code chunks (every module covered)")
    for rc in _evidence_chunks(top_chunks, limit):
        c = rc.chunk
        parts.append(
            f"- {c.file_path} :: {c.unit_type} {c.name} (importance={rc.importance_score:.4f})"
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
    data: dict,
    top_chunks: list[RankedChunk],
    repo_id: str,
    repo_name: str | None,
    inventory: list[dict],
) -> SynthesisResult:
    full, names = _valid_paths(top_chunks)
    name = repo_name or ""

    # Modules are grounded in the real file tree, not the LLM: every inventory module
    # appears exactly once (no real module dropped, no fake module invented). The LLM
    # only supplies the responsibility prose, matched back by module name.
    llm_responsibility: dict[str, str] = {}
    for m in data.get("modules", []) or []:
        nm = str(m.get("name", "")).strip().lower()
        resp = str(m.get("responsibility", "")).strip()
        if nm and resp:
            llm_responsibility[nm] = resp

    modules: list[SynthesisModule] = []
    for inv in inventory:
        resp = llm_responsibility.get(inv["name"].lower())
        if not resp:
            files = ", ".join(inv["key_files"])
            resp = f"Contains {files}." if files else "Part of the codebase."
        modules.append(SynthesisModule(name=inv["name"], path=inv["path"], responsibility=resp))

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
            log.warning("Dropping hallucinated entry-point file: %r", file)

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
        contributor_quickstart=[str(s) for s in (data.get("contributor_quickstart", []) or [])][:6],
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

    inventory = _module_inventory(top_chunks)
    user_message = _merge_and_dedupe(outputs, top_chunks, inventory)

    try:
        llm_result = await asyncio.wait_for(_call_llm(user_message), timeout=SYNTH_TIMEOUT)
    except TimeoutError:
        return Err(SynthesisError(reason=f"synthesis timed out after {SYNTH_TIMEOUT}s"))

    if not llm_result.is_ok():
        return Err(llm_result.error)

    return Ok(_build_result(llm_result.unwrap(), top_chunks, repo_id, repo_name, inventory))

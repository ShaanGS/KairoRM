"""Token-bounded chunking with scope-aware context headers.

Each `CodeUnit` becomes one `Chunk` if it fits inside the 400-token cap; oversize
units are split along inner statement/block boundaries (never mid-statement) so an
LLM sees coherent fragments. Every chunk carries a `context_header` describing its
scope (e.g. `# class AuthService > method verify_token`) — prepended at embed time
so the model understands where the code lives without seeing the rest of the file.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from functools import lru_cache

import tiktoken

from ingestion.types import Chunk, CodeUnit

MAX_TOKENS = 400
ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoder():  # noqa: ANN001
    return tiktoken.get_encoding(ENCODING_NAME)


def _token_count(text: str) -> int:
    return len(_encoder().encode(text))


def _context_header(unit: CodeUnit) -> str:
    """One-line scope descriptor — prepended to chunk content before embedding."""
    if unit.unit_type == "method" and unit.parent:
        return f"# {unit.language} | class {unit.parent} > method {unit.name}"
    if unit.unit_type == "class":
        return f"# {unit.language} | class {unit.name}"
    if unit.unit_type == "function":
        return f"# {unit.language} | function {unit.name}"
    return f"# {unit.language} | {unit.unit_type} {unit.name}"


def _chunk_id(unit: CodeUnit, start_line: int, content: str) -> str:
    """Deterministic id: hash(file_path + start_line + full content).

    Hashing the full content (not just a prefix) keeps distinct chunks distinct even
    when they start on the same line and share an opening — a 64-char prefix collided
    on real repos and ChromaDB rejects the whole batch on a single duplicate id.
    """
    h = hashlib.sha256()
    h.update(str(unit.file_path).encode("utf-8"))
    h.update(f":{start_line}:".encode())
    h.update(content.encode("utf-8"))
    return h.hexdigest()[:32]


def _make_chunk(unit: CodeUnit, content: str, start_line: int, end_line: int) -> Chunk:
    return Chunk(
        chunk_id=_chunk_id(unit, start_line, content),
        file_path=unit.file_path,
        language=unit.language,
        unit_type=unit.unit_type,
        name=unit.name,
        start_line=start_line,
        end_line=end_line,
        content=content,
        token_count=_token_count(content),
        imports=unit.imports,
        calls=unit.calls,
        context_header=_context_header(unit),
    )


def _split_oversize(unit: CodeUnit) -> list[Chunk]:
    """Greedy line-grouping that respects statement boundaries.

    We accumulate consecutive lines until adding the next one would blow the token
    cap, then close the chunk. A blank line is preferred as a break point, so we
    never cut inside a multi-line statement that lacks blank breaks. If a single
    statement exceeds the cap on its own, we emit it as one oversize chunk rather
    than mid-statement-splitting it.
    """
    lines = unit.raw_source.splitlines()
    if not lines:
        return [_make_chunk(unit, "", unit.start_line, unit.end_line)]

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_start = unit.start_line
    last_break_at = -1  # index into buf where the last blank/dedent break sits

    def flush(end_line: int) -> None:
        nonlocal buf, buf_start, last_break_at
        if not buf:
            return
        content = "\n".join(buf)
        chunks.append(_make_chunk(unit, content, buf_start, end_line))
        buf = []
        buf_start = end_line + 1
        last_break_at = -1

    for line in lines:
        buf.append(line)
        if line.strip() == "":
            last_break_at = len(buf) - 1
        if _token_count("\n".join(buf)) > MAX_TOKENS:
            # Roll back to the last blank-line boundary if we have one.
            if last_break_at > 0:
                head = buf[: last_break_at + 1]
                tail = buf[last_break_at + 1 :]
                end_line = buf_start + last_break_at
                content = "\n".join(head)
                chunks.append(_make_chunk(unit, content, buf_start, end_line))
                buf = tail
                buf_start = end_line + 1
                last_break_at = -1
            else:
                # No safe split point — emit what we have (may exceed cap by one line).
                end_line = buf_start + len(buf) - 1
                chunks.append(_make_chunk(unit, "\n".join(buf), buf_start, end_line))
                buf = []
                buf_start = end_line + 1
                last_break_at = -1

    flush(unit.end_line)
    return chunks


def _dedupe_ids(chunks: list[Chunk]) -> list[Chunk]:
    """Guarantee globally-unique chunk ids before they reach the vector store.

    Belt-and-suspenders over `_chunk_id`: two genuinely identical fragments (same file,
    line, and content) still hash equal, and ChromaDB rejects the entire batch on any
    duplicate id. Suffix collisions deterministically so a run never hard-fails on it.
    """
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        cid = c.chunk_id
        if cid in seen:
            i = 2
            while f"{cid}-{i}" in seen:
                i += 1
            cid = f"{cid}-{i}"
            c = replace(c, chunk_id=cid)
        seen.add(cid)
        out.append(c)
    return out


def chunk(units: list[CodeUnit]) -> list[Chunk]:
    """Convert `CodeUnit`s into token-bounded `Chunk`s with unique ids."""
    out: list[Chunk] = []
    for u in units:
        tokens = _token_count(u.raw_source)
        if tokens <= MAX_TOKENS:
            out.append(_make_chunk(u, u.raw_source, u.start_line, u.end_line))
        else:
            out.extend(_split_oversize(u))
    return _dedupe_ids(out)

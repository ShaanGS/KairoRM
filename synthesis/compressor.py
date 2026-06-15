"""Compress a `SynthesisResult` into a prompt-sized context package for Q&A.

The Q&A interface prepends this package to every question, so it must fit a tight
token budget. `architecture_summary` and `entry_points` are load-bearing and never
dropped; everything else is shed in priority order — least-important detail first —
until the serialized result fits, measured with the same `cl100k_base` tokenizer the
chunker uses.
"""

from __future__ import annotations

import json
from functools import lru_cache

import tiktoken

from ingestion.types import CompressedContext, SynthesisResult

ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoder():  # noqa: ANN202
    return tiktoken.get_encoding(ENCODING_NAME)


def _count(text: str) -> int:
    return len(_encoder().encode(text))


def _to_dict(result: SynthesisResult) -> dict:
    return {
        "repo_id": result.repo_id,
        "architecture_summary": result.architecture_summary,
        "modules": [
            {"name": m.name, "path": m.path, "responsibility": m.responsibility}
            for m in result.modules
        ],
        "key_dependencies": list(result.key_dependencies),
        "circular_risks": list(result.circular_risks),
        "entry_points": [
            {"name": e.name, "file": e.file, "description": e.description}
            for e in result.entry_points
        ],
        "contributor_quickstart": list(result.contributor_quickstart),
        "complexity_score": result.complexity_score,
        "generated_at": result.generated_at.isoformat(),
    }


def _top_modules(work: dict, n: int) -> None:
    # Longer names are a rough proxy for importance; keep the top n.
    work["modules"] = sorted(
        work.get("modules", []), key=lambda m: len(m.get("name", "")), reverse=True
    )[:n]


def _truncate(work: dict, key: str, n: int) -> None:
    work[key] = work.get(key, [])[:n]


# Priority-ordered truncations, applied cumulatively until under budget. Each step
# sheds the least-important surviving detail. architecture_summary and entry_points
# are intentionally never touched.
_TRUNCATIONS = [
    lambda w: _top_modules(w, 5),
    lambda w: _truncate(w, "contributor_quickstart", 3),
    lambda w: _truncate(w, "key_dependencies", 10),
    lambda w: _truncate(w, "circular_risks", 5),
    lambda w: _top_modules(w, 3),
    lambda w: _truncate(w, "contributor_quickstart", 1),
    lambda w: _truncate(w, "key_dependencies", 5),
    lambda w: _truncate(w, "circular_risks", 0),
    lambda w: _top_modules(w, 1),
    lambda w: _truncate(w, "key_dependencies", 0),
    lambda w: _truncate(w, "contributor_quickstart", 0),
    lambda w: _top_modules(w, 0),
]


def compress(result: SynthesisResult, *, token_budget: int = 3000) -> CompressedContext:
    """Serialize and, if needed, progressively truncate to fit `token_budget`."""
    work = _to_dict(result)
    content = json.dumps(work, default=str)
    tokens = _count(content)
    if tokens <= token_budget:
        return CompressedContext(content=content, token_count=tokens, truncated=False)

    for step in _TRUNCATIONS:
        step(work)
        content = json.dumps(work, default=str)
        tokens = _count(content)
        if tokens <= token_budget:
            return CompressedContext(content=content, token_count=tokens, truncated=True)

    # Exhausted every truncation and still over budget — return the smallest form.
    return CompressedContext(content=content, token_count=tokens, truncated=True)

from __future__ import annotations

import logging
from pathlib import Path

from ingestion.types import Chunk
from parsing.ranker import rank


def _chunk(
    *,
    chunk_id: str,
    name: str,
    calls: tuple[str, ...] = (),
    unit_type: str = "function",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        file_path=Path("/tmp/x.py"),
        language="python",
        unit_type=unit_type,
        name=name,
        start_line=1,
        end_line=2,
        content="...",
        token_count=10,
        imports=(),
        calls=calls,
        context_header="# python | function " + name,
    )


def test_linear_chain_ranks_terminal_node_highest() -> None:
    # A → B → C   (C is the most-called sink, so PageRank weights it highest)
    a = _chunk(chunk_id="a", name="a", calls=("b",))
    b = _chunk(chunk_id="b", name="b", calls=("c",))
    c = _chunk(chunk_id="c", name="c", calls=())
    result = rank([a, b, c])
    ranked = {r.chunk.chunk_id: r.importance_score for r in result.chunks}
    assert ranked["c"] > ranked["b"] > ranked["a"]
    assert result.cycles == []  # acyclic chain, nothing broken


def test_cycle_is_handled_without_exception(caplog) -> None:
    a = _chunk(chunk_id="a", name="a", calls=("b",))
    b = _chunk(chunk_id="b", name="b", calls=("a",))
    with caplog.at_level(logging.INFO, logger="parsing.ranker"):
        result = rank([a, b])
    assert len(result.chunks) == 2
    for r in result.chunks:
        assert r.importance_score > 0
    assert any("breaking cycle" in rec.message for rec in caplog.records)


def test_cycle_is_reported_in_result() -> None:
    # A ↔ B mutual call → exactly one back-edge broken and reported.
    # Cycles are now reported as (file_path, file_path); both chunks share x.py.
    a = _chunk(chunk_id="a", name="a", calls=("b",))
    b = _chunk(chunk_id="b", name="b", calls=("a",))
    result = rank([a, b])
    assert len(result.cycles) == 1
    caller, callee = result.cycles[0]
    assert caller == "/tmp/x.py" and callee == "/tmp/x.py"


def test_single_chunk_no_calls_gets_positive_score() -> None:
    a = _chunk(chunk_id="a", name="a", calls=())
    [r] = rank([a]).chunks
    assert r.importance_score > 0


def test_empty_input_returns_empty() -> None:
    result = rank([])
    assert result.chunks == []
    assert result.cycles == []


def test_unresolved_call_does_not_explode() -> None:
    # `a` calls `mystery` which isn't a chunk — should be ignored, no exception.
    a = _chunk(chunk_id="a", name="a", calls=("mystery",))
    [r] = rank([a]).chunks
    assert r.importance_score > 0

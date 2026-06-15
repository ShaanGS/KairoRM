from __future__ import annotations

from pathlib import Path

from ingestion.types import CodeUnit
from parsing.chunker import MAX_TOKENS, chunk


def _unit(
    *,
    name: str = "foo",
    unit_type: str = "function",
    source: str = "def foo():\n    pass\n",
    start: int = 1,
    end: int = 2,
    parent: str | None = None,
    calls: tuple[str, ...] = (),
    imports: tuple[str, ...] = (),
    file_path: Path = Path("/tmp/a.py"),
    language: str = "python",
) -> CodeUnit:
    return CodeUnit(
        file_path=file_path,
        language=language,
        unit_type=unit_type,
        name=name,
        start_line=start,
        end_line=end,
        raw_source=source,
        imports=imports,
        calls=calls,
        parent=parent,
    )


def test_small_unit_becomes_single_chunk() -> None:
    u = _unit(source="def foo(x):\n    return x + 1\n", start=1, end=2)
    chunks = chunk([u])
    assert len(chunks) == 1
    c = chunks[0]
    assert c.token_count > 0
    assert c.token_count <= MAX_TOKENS
    assert c.content == u.raw_source
    assert c.name == "foo"


def test_large_unit_is_split_under_cap() -> None:
    # Build a function that's well over 400 tokens with plenty of blank-line break points
    blocks = []
    for i in range(60):
        blocks.append(
            f"    # block {i} doing something nontrivial with a longish comment line\n"
            f"    x_{i} = compute_something_useful(arg_{i}, other_{i}, flag={i % 2 == 0})\n"
            f"    y_{i} = another_helper(x_{i}, mode='fast')\n"
            "\n"
        )
    big_source = "def huge():\n" + "".join(blocks)
    u = _unit(source=big_source, start=1, end=1 + big_source.count("\n"))

    chunks = chunk([u])
    assert len(chunks) > 1
    # Every chunk fits the cap (we allow a 1-line overshoot only when there is no
    # blank break-point; in this fixture there are plenty).
    for c in chunks:
        assert c.token_count <= MAX_TOKENS, (
            f"chunk {c.start_line}-{c.end_line} = {c.token_count} tokens"
        )
    # Line ranges cover the whole unit without gaps or overlaps.
    chunks_sorted = sorted(chunks, key=lambda c: c.start_line)
    for prev, curr in zip(chunks_sorted, chunks_sorted[1:], strict=False):
        assert curr.start_line == prev.end_line + 1


def test_context_header_for_method() -> None:
    u = _unit(name="verify", unit_type="method", parent="Auth")
    [c] = chunk([u])
    assert c.context_header == "# python | class Auth > method verify"


def test_context_header_for_function() -> None:
    u = _unit(name="foo", unit_type="function")
    [c] = chunk([u])
    assert c.context_header == "# python | function foo"


def test_context_header_for_class() -> None:
    u = _unit(name="Auth", unit_type="class", source="class Auth: ...\n", start=1, end=1)
    [c] = chunk([u])
    assert c.context_header == "# python | class Auth"


def test_chunk_id_is_deterministic() -> None:
    u = _unit()
    a = chunk([u])[0].chunk_id
    b = chunk([u])[0].chunk_id
    assert a == b
    assert len(a) == 32


def test_chunk_id_differs_per_file() -> None:
    u1 = _unit(file_path=Path("/tmp/a.py"))
    u2 = _unit(file_path=Path("/tmp/b.py"))
    assert chunk([u1])[0].chunk_id != chunk([u2])[0].chunk_id


def test_imports_and_calls_carry_through() -> None:
    u = _unit(imports=("import os",), calls=("bar", "baz"))
    [c] = chunk([u])
    assert c.imports == ("import os",)
    assert c.calls == ("bar", "baz")


def test_chunk_ids_are_unique_even_on_collision() -> None:
    # Two units that hash to the same id (same file/line/content) must still get
    # distinct chunk ids, or ChromaDB rejects the whole batch.
    from ingestion.types import CodeUnit
    from parsing.chunker import chunk

    u = CodeUnit(
        file_path=Path("/r/a.py"), language="python", unit_type="function", name="f",
        start_line=1, end_line=1, raw_source="x=1", imports=(), calls=(), parent=None,
    )
    chunks = chunk([u, u, u])  # identical units → would collide
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)) == 3
